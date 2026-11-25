"""Class-specific CAM grids and interpolation to input pixels."""

from functools import lru_cache

import cv2
import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d
from torch import nn

from .models import classify_features, target_features
from .timm_adapter import TimmClassifier
from .voc import IMAGE_SIZE, SCORING_SIZE

CAM_METHODS = ("gradcam", "gradcampp", "extendedcam")
CAM_ABLATIONS = {
    "gradcam-gaussian": "gradcam",
    "gradcampp-gaussian": "gradcampp",
    "extendedcam-bilinear": "extendedcam",
}
GRADCAMPP_FORMULAS = ("archive", "equation19")
FEATURE_SIZE = 14


def _gradcampp_weights(activations: torch.Tensor, gradients: torch.Tensor, formula: str) -> torch.Tensor:
    spatial = (-2, -1)
    positive = gradients.relu()
    if formula == "equation19":
        # Cancel g² analytically so small logit gradients keep their relative weights.
        # Pixels with a zero gradient or a singular denominator contribute zero.
        denominator = 2 + gradients * activations.sum(dim=spatial, keepdim=True)
        weights = positive / torch.where(denominator != 0, denominator, torch.inf)
    elif formula == "archive":
        squared = gradients.square()
        cubic_term = (activations * gradients.pow(3)).sum(dim=spatial, keepdim=True)
        denominator = 2 * squared + cubic_term
        denominator = torch.where(denominator != 0, denominator, 1.0)
        stabilized = denominator + 1e-7
        # A signed denominator can exactly cancel the historical epsilon in FP32.
        alpha = squared / torch.where(stabilized == 0, denominator, stabilized)
        weights = alpha * positive
    else:
        raise ValueError(f"Unknown Grad-CAM++ formula {formula!r}")
    # exp(logit) is a positive scale shared by every pixel of this CAM.
    return weights.sum(dim=spatial, keepdim=True)


def grid_cam(
    method: str, activations: torch.Tensor, gradients: torch.Tensor, *, gradcampp_formula: str = "archive",
) -> torch.Tensor:
    """Compute the rectified grid CAM from logit derivatives at the target layer."""
    if method == "gradcam":
        gradients = gradients.mean(dim=(-2, -1), keepdim=True)
    elif method == "gradcampp":
        gradients = _gradcampp_weights(activations, gradients, gradcampp_formula)
    elif method != "extendedcam":
        raise ValueError(f"Unknown CAM method {method!r}")
    return (activations * gradients).sum(dim=1).relu()


def class_grid_cams(
    model: nn.Module, images: torch.Tensor, image_indices: torch.Tensor,
    categories: torch.Tensor, methods: tuple[str, ...] = CAM_METHODS,
    *, gradcampp_formula: str = "archive",
) -> dict[str, torch.Tensor]:
    """Return a grid per (image index, target category) in the supplied order."""
    if model.training:
        raise ValueError("Class CAMs require an evaluation-mode classifier")
    if isinstance(model, TimmClassifier):
        features, gradients = _timm_gradients(model, images, image_indices, categories)
    else:
        with torch.no_grad():
            features = target_features(model, images).index_select(0, image_indices)
        if len(images) == 1 and len(categories) > 1:
            # Retain the archived batch-of-one backward pass for each cue.
            gradients = torch.cat([
                _class_gradients(model, target, category)
                for target, category in zip(features.split(1), categories.split(1), strict=True)
            ])
        else:
            gradients = _class_gradients(model, features, categories)
    return {method: grid_cam(method, features, gradients, gradcampp_formula=gradcampp_formula)
            for method in methods}


@torch.enable_grad()
def _class_gradients(model: nn.Module, features: torch.Tensor, categories: torch.Tensor) -> torch.Tensor:
    features = features.detach().requires_grad_(True)
    logits = classify_features(model, features)
    scores = logits.gather(1, categories[:, None]).sum()
    gradients, = torch.autograd.grad(scores, features)
    return gradients


@torch.enable_grad()
def _timm_gradients(
    model: TimmClassifier, images: torch.Tensor, image_indices: torch.Tensor, categories: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    # Each backward pass selects up to one class per image. Images with several
    # cues share the forward pass without mixing their class gradients.
    ranks = [0] * len(images)
    groups: list[list[int]] = []
    for cue, image in enumerate(image_indices.tolist()):
        rank = ranks[image]
        if rank == len(groups):
            groups.append([])
        groups[rank].append(cue)
        ranks[image] += 1
    logits, target = model.forward_with_target(images.detach().requires_grad_(True))
    gradients = target.new_empty((len(categories), *target.shape[1:]))
    for index, group in enumerate(groups, start=1):
        cues = image_indices.new_tensor(group)
        rows = image_indices[cues]
        scores = logits[rows].gather(1, categories[cues, None]).sum()
        gradient, = torch.autograd.grad(scores, target, retain_graph=index < len(groups))
        gradients.index_copy_(0, cues, gradient.index_select(0, rows))
    input_size = tuple(images.shape[-2:])
    activations = model.spatial(target.detach().index_select(0, image_indices), input_size=input_size)
    return activations, model.spatial(gradients, input_size=input_size)


@lru_cache(maxsize=16)
def gaussian_basis(
    sigma: float, device: torch.device, length: int = IMAGE_SIZE, cells: int = FEATURE_SIZE,
) -> torch.Tensor:
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("Gaussian sigma must be finite and positive")
    indices = np.arange(cells)
    centers = ((2 * indices + 1) * length - cells) // (2 * cells)
    impulses = np.zeros((length, cells), dtype=np.float64)
    impulses[centers, indices] = 1.0
    # Pixel centers give 7 + 16i for the archived 14-to-224 upsampling.
    kernels = gaussian_filter1d(impulses, sigma, axis=0, mode="constant", truncate=4.0)
    return torch.from_numpy(kernels).to(device)


def upsample(
    method: str, grids: torch.Tensor, sigma: float | None = None, *,
    output_size: tuple[int, int] = SCORING_SIZE,
) -> torch.Tensor:
    """Map rectified CAM grids onto the requested input pixel grid."""
    height, width = output_size
    if grids.ndim != 3 or not grids.numel():
        raise ValueError("Expected nonempty CAM grids with shape (batch, height, width)")
    rows, columns = grids.shape[-2:]
    if method in ("extendedcam", "gradcam-gaussian", "gradcampp-gaussian"):
        if sigma is None:
            raise ValueError("Gaussian upsampling requires a sigma in input pixels")
        maxima = grids.amax(dim=(-2, -1), keepdim=True)
        grids = grids / torch.where(maxima > 0, maxima, 1.0)
        if rows > height or columns > width:
            raise ValueError("Gaussian upsampling needs at least one output pixel per grid cell")
        row_basis = gaussian_basis(sigma, grids.device, height, rows)
        column_basis = gaussian_basis(sigma, grids.device, width, columns)
        return (row_basis @ grids.to(torch.float64)) @ column_basis.T
    if method not in CAM_METHODS and method not in CAM_ABLATIONS:
        raise ValueError(f"Unknown CAM method {method!r}")
    maps = [cv2.resize(grid, (width, height), interpolation=cv2.INTER_LINEAR)
            for grid in grids.detach().cpu().numpy()]
    return torch.from_numpy(np.stack(maps))


def maxima_points(maps: torch.Tensor) -> list[tuple[int, int]]:
    """Return the first maximum of each map in row-major order, including ties."""
    if not torch.isfinite(maps).all():
        raise ValueError("CAM contains non-finite values")
    return [divmod(index, maps.shape[-1]) for index in maps.flatten(1).argmax(1).tolist()]

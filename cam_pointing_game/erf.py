"""Fit the effective receptive field used by Gaussian upsampling."""

from collections.abc import Callable, Iterable, Sequence
from copy import deepcopy

import numpy as np
import torch
from scipy.optimize import curve_fit
from torch import nn
from torch.func import replace_all_batch_norm_modules_, vmap

from .models import SpatialClassifier, layer3_features, target_features
from .timm_adapter import TimmClassifier


def fit_gaussian(heatmap: np.ndarray) -> dict[str, float]:
    if not np.isfinite(heatmap).all() or heatmap.sum() <= 0 or heatmap.var() == 0:
        raise ValueError("ERF fitting requires a finite, nonconstant map with positive mass")
    density = heatmap / heatmap.sum()
    height, width = density.shape
    x, y = np.meshgrid(np.arange(width), np.arange(height))

    def gaussian(xy, amplitude, cx, cy, sigma_x, sigma_y):
        x, y = xy
        squared_distance = ((x - cx) / sigma_x) ** 2 + ((y - cy) / sigma_y) ** 2
        return (amplitude * np.exp(-0.5 * squared_distance) / (2 * np.pi * sigma_x * sigma_y)).ravel()

    initial = (2, 0.4 * (width - 1), 0.4 * (height - 1), (width - 1) / 15, (height - 1) / 9)
    parameters, _ = curve_fit(gaussian, (x, y), density.ravel(), p0=initial)
    _, center_x, center_y, sigma_x, sigma_y = parameters
    residual = density.ravel() - gaussian((x, y), *parameters)
    return {
        "center_x": float(center_x), "center_y": float(center_y),
        "sigma_x": float(abs(sigma_x)), "sigma_y": float(abs(sigma_y)),
        "sigma": float((abs(sigma_x) + abs(sigma_y)) / 2),
        "r_squared": float(1 - residual.var() / density.var()),
    }


@torch.inference_mode(False)
@torch.enable_grad()
def measure_erf(
    model: nn.Module, batches: Iterable[torch.Tensor | Sequence[torch.Tensor]], *,
    input_size: tuple[int, int] | None = None,
    progress: Callable[[int], None] | None = None,
) -> dict[str, float]:
    device = next(model.parameters()).device
    accumulated = None

    if isinstance(model, (SpatialClassifier, TimmClassifier)):
        if model.training:
            raise ValueError("ERF fitting for this classifier requires evaluation mode")

        def responses(images: torch.Tensor) -> torch.Tensor:
            return center_responses(target_features(model, images))
    else:
        # The archived run used training-mode BatchNorm one image at a time.
        features_model = replace_all_batch_norm_modules_(deepcopy(model)).train().requires_grad_(False)

        def center_response(image: torch.Tensor) -> torch.Tensor:
            features = layer3_features(features_model, image.unsqueeze(0))
            return center_responses(features)[0]

        responses = vmap(center_response)

    processed = 0
    for batch in batches:
        pixels = batch if isinstance(batch, torch.Tensor) else batch[0]
        if pixels.ndim != 4 or pixels.shape[1] != 3 or not pixels.numel():
            raise ValueError("ERF calibration batches need nonempty RGB shape (batch, 3, height, width)")
        if input_size is not None and pixels.shape[-2:] != input_size:
            raise ValueError(f"ERF calibration images must have height and width {input_size}")
        images = pixels.detach().to(device, dtype=torch.float32, non_blocking=True)
        if images.is_inference():
            images = images.clone()
        images.requires_grad_(True)
        with torch.autocast(device.type, enabled=False):
            gradients, = torch.autograd.grad(responses(images).sum(), images)
        positive_gradients = gradients.sum(dim=1).relu()
        batch_erf = positive_gradients.sum(dim=0, dtype=torch.float64)
        if accumulated is None:
            accumulated = batch_erf
        else:
            if accumulated.shape != batch_erf.shape:
                raise ValueError("ERF calibration images must use the same height and width")
            accumulated += batch_erf
        processed += len(pixels)
        if progress is not None:
            progress(processed)
    if accumulated is None:
        raise ValueError("ERF fitting requires at least one calibration batch")
    return fit_gaussian(accumulated.cpu().numpy())


def center_responses(features: torch.Tensor) -> torch.Tensor:
    height, width = features.shape[-2:]
    return features[:, :, height // 2, width // 2].sum(dim=1)

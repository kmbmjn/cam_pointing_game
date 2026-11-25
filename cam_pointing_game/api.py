"""Compute dense class activation maps in one call."""

from collections.abc import Iterable, Sequence

import torch
from torch import nn

from .cams import class_grid_cams, upsample
from .erf import measure_erf
from .timm_adapter import TimmClassifier

Targets = int | Sequence[int] | torch.Tensor
Calibration = torch.Tensor | Iterable[torch.Tensor | Sequence[torch.Tensor]]


def _prepare_model(model: nn.Module, target_layer: str | None) -> nn.Module:
    if isinstance(model, TimmClassifier):
        return model if target_layer is None else TimmClassifier(model.backbone, target_layer=target_layer)
    if target_layer is not None or hasattr(model, "pretrained_cfg") or type(model).__module__.startswith("timm."):
        return TimmClassifier(model, target_layer=target_layer)
    return model


def _cam(
    method: str, model: nn.Module, images: torch.Tensor, targets: Targets, *,
    sigma: float | None = None, gradcampp_formula: str = "archive",
) -> torch.Tensor:
    if images.ndim != 4 or images.shape[1] != 3 or not images.numel():
        raise ValueError("Images need nonempty RGB shape (batch, 3, height, width)")
    images = images.to(next(model.parameters()).device, dtype=torch.float32)
    if images.is_inference():
        images = images.clone()
    categories = torch.as_tensor(targets, device=images.device)
    if categories.is_inference():
        categories = categories.clone()
    if categories.ndim == 0:
        categories = categories.expand(len(images))
    if (categories.shape != (len(images),) or categories.is_floating_point()
            or categories.is_complex() or categories.dtype == torch.bool):
        raise ValueError("Supply one integer category per image or a single shared category")
    indices = torch.arange(len(images), device=images.device)
    with torch.autocast(images.device.type, enabled=False):
        grids = class_grid_cams(
            model, images, indices, categories.long(), (method,), gradcampp_formula=gradcampp_formula,
        )
    return upsample(method, grids[method], sigma, output_size=tuple(images.shape[-2:])).cpu()


@torch.inference_mode(False)
def grad_cam(
    model: nn.Module, images: torch.Tensor, targets: Targets, *, target_layer: str | None = None,
) -> torch.Tensor:
    """Return a Grad-CAM map per image at the input resolution."""
    return _cam("gradcam", _prepare_model(model, target_layer), images, targets)


@torch.inference_mode(False)
def grad_cam_pp(
    model: nn.Module, images: torch.Tensor, targets: Targets, *, formula: str = "equation19",
    target_layer: str | None = None,
) -> torch.Tensor:
    """Return Grad-CAM++ maps using Eq. (19), or the Table 4 weights with formula='archive'."""
    return _cam("gradcampp", _prepare_model(model, target_layer), images, targets, gradcampp_formula=formula)


@torch.inference_mode(False)
def extended_cam(
    model: nn.Module, images: torch.Tensor, targets: Targets, *,
    calibration: Calibration | None = None, sigma: float | None = None,
    target_layer: str | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Return Extended-CAM maps and the ERF fit, using calibration images to estimate sigma."""
    if sigma is not None and calibration is not None:
        raise ValueError("Supply calibration images or a fitted sigma")
    model = _prepare_model(model, target_layer)
    if sigma is None:
        batches = images if calibration is None else calibration
        if isinstance(batches, torch.Tensor):
            batches = (batches,)
        fit = measure_erf(model, batches, input_size=tuple(images.shape[-2:]))
    else:
        fit = {"sigma": sigma}
    return _cam("extendedcam", model, images, targets, sigma=fit["sigma"]), fit

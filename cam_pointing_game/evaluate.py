"""Run the model-free baseline and/or CAMs on each VOC image/category pair."""

from collections.abc import Callable, Iterable

import torch
from torch import nn

from .cams import CAM_ABLATIONS, CAM_METHODS, class_grid_cams, maxima_points, upsample
from .metric import CENTER, PointingGame, project_point
from .voc import Annotation

METHODS = ("center", *CAM_METHODS)


def evaluate(
    annotations: list[Annotation], methods: tuple[str, ...], *,
    difficult_set: str = "all-boxes", model: nn.Module | None = None,
    batches: Iterable[tuple[torch.Tensor, torch.Tensor]] | None = None,
    sigma: float | None = None, gradcampp_formula: str = "archive",
    progress: Callable[[int], None] | None = None,
) -> dict:
    """Score the annotated categories, with exactly one image per annotation for CAMs."""
    unknown_methods = set(methods).difference(METHODS, CAM_ABLATIONS)
    if unknown_methods:
        raise ValueError(f"Unknown evaluation methods {sorted(unknown_methods)}")
    games = {method: PointingGame(difficult_set) for method in methods}
    if "center" in games:
        for annotation in annotations:
            for category in annotation.categories:
                games["center"].update(annotation, category, CENTER)

    cam_methods = tuple(method for method in methods if method != "center")
    if not cam_methods:
        return {method: game.results() for method, game in games.items()}
    if model is None or batches is None:
        raise ValueError("Model-based evaluation requires a classifier and image batches")
    grid_methods = {method: CAM_ABLATIONS.get(method, method) for method in cam_methods}
    unique_grids = tuple(dict.fromkeys(grid_methods.values()))

    device = next(model.parameters()).device
    seen_indices: set[int] = set()
    for images, indices in batches:
        if indices.ndim != 1 or len(indices) != len(images):
            raise ValueError("Each batch needs one annotation index per image")
        batch_annotations = []
        for index in indices.tolist():
            if index < 0 or index >= len(annotations):
                raise ValueError(f"Invalid annotation index {index}")
            if index in seen_indices:
                raise ValueError(f"Image {annotations[index].image_id} was evaluated twice")
            seen_indices.add(index)
            batch_annotations.append(annotations[index])
        prompts = [(i, category) for i, annotation in enumerate(batch_annotations)
                   for category in annotation.categories]
        if prompts:
            input_size = tuple(images.shape[-2:])
            image_indices, categories = torch.tensor(prompts, device=device).T.contiguous()
            grids = class_grid_cams(
                model, images.to(device, dtype=torch.float32, non_blocking=True),
                image_indices, categories, unique_grids, gradcampp_formula=gradcampp_formula,
            )
            for method, grid in grid_methods.items():
                points = maxima_points(upsample(method, grids[grid], sigma, output_size=input_size))
                for (index, category), point in zip(prompts, points, strict=True):
                    games[method].update(batch_annotations[index], category, project_point(point, input_size))
        if progress is not None:
            progress(len(seen_indices))

    if len(seen_indices) != len(annotations):
        raise ValueError(f"Received {len(seen_indices)} of {len(annotations)} evaluation images")
    return {method: game.results() for method, game in games.items()}

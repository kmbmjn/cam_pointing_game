"""Render the same category cue and radius-15 hit test as the evaluator."""

from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torch import nn

from .cams import CAM_METHODS, class_grid_cams, maxima_points, upsample
from .metric import CENTER, RADIUS, is_hit, project_point, reflect_point
from .models import model_input_spec
from .voc import IMAGE_SIZE, SCORING_SIZE, VOC_CLASSES, Annotation, preprocess_image

LABELS = {"center": "Center", "gradcam": "Grad-CAM", "gradcampp": "Grad-CAM++", "extendedcam": "Extended-CAM"}
BOX_COLOR = "#72ed99"
POINT_COLOR = "#ffe55c"
RAW_MAX_COLOR = "#ffffff"


def render_pointing_game(
    model: nn.Module, image_path: Path, annotation: Annotation, category: int, sigma: float, *,
    gradcampp_formula: str = "archive",
) -> Image.Image:
    boxes = tuple(box for box in annotation.boxes if box.category == category)
    if not boxes:
        raise ValueError(f"{VOC_CLASSES[category]} is absent from {annotation.image_id}")
    input_spec = model_input_spec(model)
    pixels = preprocess_image(image_path, annotation, input_spec)
    device = next(model.parameters()).device
    images = torch.from_numpy(pixels).permute(2, 0, 1)[None].to(device, dtype=torch.float32)
    grids = class_grid_cams(
        model, images, torch.zeros(1, dtype=torch.long, device=device),
        torch.tensor([category], device=device), gradcampp_formula=gradcampp_formula,
    )
    canvas_pixels = pixels
    if input_spec.size != SCORING_SIZE:
        canvas_pixels = cv2.resize(pixels, SCORING_SIZE[::-1], interpolation=cv2.INTER_AREA)
    panels = [(LABELS["center"], canvas_pixels, CENTER)]
    for method in CAM_METHODS:
        maps = upsample(method, grids[method], sigma, output_size=input_spec.size)
        point = project_point(maxima_points(maps)[0], input_spec.size)
        panels.append((LABELS[method], heatmap_overlay(canvas_pixels, maps[0]), point))

    gap = 12
    header = 34
    caption = 34
    width = gap + (IMAGE_SIZE + gap) * len(panels)
    figure = Image.new("RGB", (width, header + IMAGE_SIZE + caption), "#14202a")
    draw = ImageDraw.Draw(figure)
    font = ImageFont.load_default(size=15)
    draw.text((gap, 8), f"VOC 2007   /   {annotation.image_id}   /   {VOC_CLASSES[category]}",
              font=font, fill="white")
    for index, (name, background, point) in enumerate(panels):
        panel = Image.fromarray(background)
        marks = ImageDraw.Draw(panel)
        for box in boxes:
            marks.rectangle((box.xmin, box.ymin, box.xmax, box.ymax), outline=BOX_COLOR, width=2)
        row, column = reflect_point(point, annotation)
        if (row, column) != point:
            raw_row, raw_col = point
            marks.line((raw_col - 3, raw_row - 3, raw_col + 3, raw_row + 3), fill=RAW_MAX_COLOR, width=2)
            marks.line((raw_col - 3, raw_row + 3, raw_col + 3, raw_row - 3), fill=RAW_MAX_COLOR, width=2)
        marks.ellipse((column - RADIUS, row - RADIUS, column + RADIUS, row + RADIUS),
                      outline=POINT_COLOR, width=2)
        marks.ellipse((column - 2, row - 2, column + 2, row + 2), fill=POINT_COLOR)
        hit = is_hit((row, column), boxes)
        left = gap + index * (IMAGE_SIZE + gap)
        figure.paste(panel, (left, header))
        draw.text((left, header + IMAGE_SIZE + 8), name, font=font, fill="white")
        draw.text((left + IMAGE_SIZE, header + IMAGE_SIZE + 8), "Hit" if hit else "Miss",
                  font=font, anchor="ra", fill=BOX_COLOR if hit else "#ffaeaa")
    return figure


def heatmap_overlay(pixels: np.ndarray, heatmap: torch.Tensor) -> np.ndarray:
    values = cv2.resize(heatmap.detach().cpu().numpy(), (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LINEAR)
    low, high = values.min(), values.max()
    if high <= low:
        return pixels
    weights = (values - low) / (high - low)
    colored = cv2.applyColorMap(np.uint8(255 * weights), cv2.COLORMAP_INFERNO)[:, :, ::-1]
    alpha = (0.65 * weights)[..., None]
    return np.uint8((1 - alpha) * pixels + alpha * colored)

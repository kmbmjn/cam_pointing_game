"""Class-wise hit rates for the Extended-CAM pointing game."""

import numpy as np
from shapely.geometry import Point, Polygon

from .voc import IMAGE_SIZE, VOC_CLASSES, Annotation, Box

RADIUS = 15
CENTER = (IMAGE_SIZE // 2, IMAGE_SIZE // 2)
DIFFICULT_SETS = ("all-boxes", "target-boxes")


def project_point(point: tuple[int, int], size: tuple[int, int]) -> tuple[int, int]:
    """Project a native input pixel center onto the 224 × 224 scoring grid."""
    row, column = point
    height, width = size
    return (2 * row + 1) * IMAGE_SIZE // (2 * height), (2 * column + 1) * IMAGE_SIZE // (2 * width)


def reflect_point(point: tuple[int, int], annotation: Annotation) -> tuple[int, int]:
    """Reflect an argmax in the padding back toward the content, in (row, column)."""
    row, column = point
    margin = annotation.mirror_margin
    near, far = margin, IMAGE_SIZE - margin

    def reflect(coordinate: int) -> int:
        if coordinate < near:
            return 2 * near - coordinate
        if coordinate > far:
            return 2 * far - coordinate
        return coordinate

    if annotation.height > annotation.width:
        return row, reflect(column)
    return reflect(row), column


def is_hit(point: tuple[int, int], boxes: tuple[Box, ...]) -> bool:
    # Use the same polygonal circle and (row, column) order as the experiment.
    circle = Point(*point).buffer(RADIUS, quad_segs=16)
    return any(circle.intersects(Polygon((
        (box.ymin, box.xmin), (box.ymax, box.xmin),
        (box.ymax, box.xmax), (box.ymin, box.xmax),
    ))) for box in boxes)


def is_difficult(annotation: Annotation, category: int, difficult_set: str) -> bool:
    if len(annotation.categories) < 2:
        return False
    boxes = annotation.boxes
    if difficult_set == "target-boxes":
        boxes = tuple(box for box in boxes if box.category == category)
    content_area = IMAGE_SIZE * (IMAGE_SIZE - 2 * annotation.mirror_margin)
    return 4 * sum(box.area for box in boxes) < content_area


class PointingGame:
    def __init__(self, difficult_set: str = "all-boxes"):
        if difficult_set not in DIFFICULT_SETS:
            raise ValueError(f"Unknown difficult-set rule {difficult_set!r}")
        self.difficult_set = difficult_set
        self.hits = np.zeros((2, len(VOC_CLASSES)), dtype=np.int64)
        self.totals = np.zeros_like(self.hits)

    def update(self, annotation: Annotation, category: int, point: tuple[int, int]) -> None:
        boxes = tuple(box for box in annotation.boxes if box.category == category)
        if not boxes:
            raise ValueError(f"Category {category} is absent from {annotation.image_id}")
        hit = is_hit(reflect_point(point, annotation), boxes)
        self.totals[0, category] += 1
        self.hits[0, category] += hit
        if is_difficult(annotation, category, self.difficult_set):
            self.totals[1, category] += 1
            self.hits[1, category] += hit

    def results(self) -> dict:
        results = {}
        for name, hits, totals in zip(("all", "difficult"), self.hits, self.totals):
            supported = totals > 0
            per_class = {}
            for label, hit, total in zip(VOC_CLASSES, hits.tolist(), totals.tolist()):
                per_class[label] = {
                    "hits": hit, "misses": total - hit,
                    "score_percent": 100 * hit / total if total else None,
                }
            results[name] = {
                "score_percent": float(100 * np.mean(hits[supported] / totals[supported]))
                if supported.any() else None,
                "hits": int(hits.sum()), "misses": int((totals - hits).sum()),
                "per_class": per_class,
            }
        return results

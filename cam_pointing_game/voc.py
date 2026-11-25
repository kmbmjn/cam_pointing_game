"""PASCAL VOC 2007 data and the preprocessing used for Table 4."""

from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

IMAGE_SIZE = 224
SCORING_SIZE = (IMAGE_SIZE, IMAGE_SIZE)
VOC_CLASSES = (
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
    "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor",
)
CLASS_INDEX = {name: index for index, name in enumerate(VOC_CLASSES)}
BOX_SOURCES = ("first-descendant", "object")


@dataclass(frozen=True)
class InputSpec:
    size: tuple[int, int] = SCORING_SIZE  # height, width
    interpolation: str = "bilinear"
    mean: tuple[float, ...] | None = None
    std: tuple[float, ...] | None = None

    def to_dict(self) -> dict:
        return {
            "size": list(self.size), "interpolation": self.interpolation,
            "mean": list(self.mean) if self.mean is not None else None,
            "std": list(self.std) if self.std is not None else None,
        }


TABLE4_INPUT = InputSpec()


@dataclass(frozen=True)
class Box:
    category: int
    xmin: int
    ymin: int
    xmax: int
    ymax: int

    @property
    def area(self) -> int:
        return (self.xmax - self.xmin) * (self.ymax - self.ymin)


@dataclass(frozen=True)
class Annotation:
    image_id: str
    width: int
    height: int
    boxes: tuple[Box, ...]

    @property
    def padding(self) -> tuple[int, int]:
        side = max(self.width, self.height)
        return (side - self.width) // 2, (side - self.height) // 2

    @property
    def mirror_margin(self) -> int:
        # The historical evaluator computed this from the unpadded long side.
        return int(max(self.padding) * IMAGE_SIZE / max(self.width, self.height))

    @property
    def categories(self) -> tuple[int, ...]:
        return tuple(sorted({box.category for box in self.boxes}))


def read_annotations(
    root: Path, split: str, limit: int | None = None, *, box_source: str = BOX_SOURCES[0],
) -> list[Annotation]:
    split_path = root / "ImageSets" / "Main" / f"{split}.txt"
    image_ids = split_path.read_text(encoding="ascii").split()
    if not image_ids:
        raise ValueError(f"Empty VOC split at {split_path}")
    return [read_annotation(root / "Annotations" / f"{image_id}.xml", box_source=box_source)
            for image_id in image_ids[:limit]]


def read_annotation(path: Path, *, box_source: str = BOX_SOURCES[0]) -> Annotation:
    if box_source not in BOX_SOURCES:
        raise ValueError(f"Unknown VOC box source {box_source!r}")
    xml = ElementTree.parse(path).getroot()
    width, height = (int(xml.findtext(f"size/{axis}", "0")) for axis in ("width", "height"))
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid image dimensions in {path}")
    image = Annotation(path.stem, width, height, ())
    pad_x, pad_y = image.padding
    scale_x = IMAGE_SIZE / (width + 2 * pad_x)
    scale_y = IMAGE_SIZE / (height + 2 * pad_y)

    # Preserve the VOC coordinates as stored, with the original integer truncation.
    boxes = []
    for obj in xml.findall("object"):
        # The archived parser can encounter a part box before the object box.
        box = obj.find(".//bndbox" if box_source == "first-descendant" else "bndbox")
        if box is None:
            raise ValueError(f"Missing {box_source} bounding box in {path}")
        x_min, y_min, x_max, y_max = (
            float(box.findtext(field, ""))
            for field in ("xmin", "ymin", "xmax", "ymax")
        )
        boxes.append(Box(
            CLASS_INDEX[obj.findtext("name")],
            int((x_min + pad_x) * scale_x), int((y_min + pad_y) * scale_y),
            int((x_max + pad_x) * scale_x), int((y_max + pad_y) * scale_y),
        ))
    return Annotation(path.stem, width, height, tuple(boxes))


def preprocess_image(
    path: Path, annotation: Annotation, input_spec: InputSpec = TABLE4_INPUT,
) -> np.ndarray:
    with Image.open(path) as image:
        if image.size != (annotation.width, annotation.height):
            raise ValueError(f"JPEG and XML dimensions differ for {path}")
        pixels = np.asarray(image.convert("RGB"))
    pad_x, pad_y = annotation.padding
    pixels = np.pad(pixels, ((pad_y, pad_y), (pad_x, pad_x), (0, 0)), mode="reflect")
    height, width = input_spec.size
    interpolation = Image.Resampling[input_spec.interpolation.upper()]
    return np.array(Image.fromarray(pixels).resize((width, height), interpolation))


class VOCImages(Dataset):
    def __init__(
        self, root: Path, annotations: list[Annotation], input_spec: InputSpec = TABLE4_INPUT,
    ):
        self.image_dir = root / "JPEGImages"
        self.annotations = annotations
        self.input_spec = input_spec

    def __len__(self) -> int:
        return len(self.annotations)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        annotation = self.annotations[index]
        path = self.image_dir / f"{annotation.image_id}.jpg"
        pixels = preprocess_image(path, annotation, self.input_spec)
        return torch.from_numpy(pixels).permute(2, 0, 1), index


def make_loader(
    root: Path, annotations: list[Annotation], batch_size: int, workers: int,
    device: torch.device, *, input_spec: InputSpec = TABLE4_INPUT,
    shuffle: bool = False, generator: torch.Generator | None = None,
) -> DataLoader:
    return DataLoader(
        VOCImages(root, annotations, input_spec), batch_size=batch_size, num_workers=workers,
        pin_memory=device.type == "cuda", persistent_workers=workers > 0,
        shuffle=shuffle, generator=generator,
    )


def multilabel_targets(annotations: list[Annotation]) -> torch.Tensor:
    labels = torch.zeros((len(annotations), len(VOC_CLASSES)), dtype=torch.float32)
    for index, annotation in enumerate(annotations):
        labels[index, list(annotation.categories)] = 1.0
    return labels

"""Dataset helpers for YOLO-formatted VisDrone labels."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from PIL import Image

from .constants import CAR_CLASS_IDS, CLASS_NAMES, HUMAN_CLASS_IDS, TARGET_CLASS_IDS

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

SPLITS = {
    "train": "VisDrone2019-DET-train",
    "val": "VisDrone2019-DET-val",
    "test-dev": "VisDrone2019-DET-test-dev",
    "test-challenge": "VisDrone2019-DET-test-challenge",
}


@dataclass(frozen=True)
class Detection:
    class_id: int
    x_center: float
    y_center: float
    width: float
    height: float
    confidence: float | None = None

    @property
    def class_name(self) -> str:
        return CLASS_NAMES.get(self.class_id, f"class_{self.class_id}")

    @property
    def target_name(self) -> str:
        if self.class_id in HUMAN_CLASS_IDS:
            return "human"
        if self.class_id in CAR_CLASS_IDS:
            return "car"
        return "other"

    @property
    def is_target(self) -> bool:
        return self.class_id in TARGET_CLASS_IDS

    @property
    def is_human(self) -> bool:
        return self.class_id in HUMAN_CLASS_IDS

    @property
    def is_car(self) -> bool:
        return self.class_id in CAR_CLASS_IDS

    @property
    def normalized_area(self) -> float:
        return self.width * self.height

    def to_xyxy(self, image_width: int, image_height: int) -> tuple[int, int, int, int]:
        x1 = int(round((self.x_center - self.width / 2) * image_width))
        y1 = int(round((self.y_center - self.height / 2) * image_height))
        x2 = int(round((self.x_center + self.width / 2) * image_width))
        y2 = int(round((self.y_center + self.height / 2) * image_height))
        x1 = max(0, min(image_width - 1, x1))
        y1 = max(0, min(image_height - 1, y1))
        x2 = max(0, min(image_width - 1, x2))
        y2 = max(0, min(image_height - 1, y2))
        return x1, y1, x2, y2


def dataset_root(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def split_dir(root: str | Path, split: str) -> Path:
    if split not in SPLITS:
        raise ValueError(f"Unknown split '{split}'. Expected one of: {', '.join(SPLITS)}")
    return dataset_root(root) / SPLITS[split]


def images_dir(root: str | Path, split: str) -> Path:
    return split_dir(root, split) / "images"


def labels_dir(root: str | Path, split: str) -> Path:
    return split_dir(root, split) / "labels"


def iter_images(root: str | Path, split: str) -> Iterable[Path]:
    directory = images_dir(root, split)
    if not directory.exists():
        return []
    return sorted(p for p in directory.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)


def label_path_for_image(image_path: str | Path) -> Path:
    image_path = Path(image_path)
    return image_path.parent.parent / "labels" / f"{image_path.stem}.txt"


def parse_yolo_label_line(line: str) -> Detection | None:
    parts = line.strip().split()
    if len(parts) < 5:
        return None
    try:
        class_id = int(float(parts[0]))
        x_center, y_center, width, height = (float(v) for v in parts[1:5])
    except ValueError:
        return None
    return Detection(class_id, x_center, y_center, width, height)


def read_yolo_labels(label_path: str | Path) -> list[Detection]:
    label_path = Path(label_path)
    if not label_path.exists():
        return []
    detections: list[Detection] = []
    for line in label_path.read_text().splitlines():
        detection = parse_yolo_label_line(line)
        if detection is not None:
            detections.append(detection)
    return detections


def filter_targets(detections: Iterable[Detection]) -> list[Detection]:
    return [d for d in detections if d.is_target]


def image_size(image_path: str | Path) -> tuple[int, int]:
    with Image.open(image_path) as image:
        return image.size


def find_labeled_images(root: str | Path, split: str, limit: int | None = None) -> list[Path]:
    selected: list[Path] = []
    for image_path in iter_images(root, split):
        if read_yolo_labels(label_path_for_image(image_path)):
            selected.append(image_path)
        if limit is not None and len(selected) >= limit:
            break
    return selected

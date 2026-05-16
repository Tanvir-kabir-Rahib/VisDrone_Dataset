#!/usr/bin/env python3
"""Run human/car detection and counting on images.

Without --weights, this script uses the existing YOLO labels as a local
ground-truth demo path. With --weights and Ultralytics installed, it runs model
inference and draws predictions.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from visdrone_project.constants import TARGET_CLASS_IDS
from visdrone_project.dataset import (
    Detection,
    filter_targets,
    find_labeled_images,
    iter_images,
    label_path_for_image,
    read_yolo_labels,
)
from visdrone_project.visualize import draw_detections


def source_images(dataset_root: Path, split: str | None, source: Path | None, limit: int) -> list[Path]:
    if source is not None:
        source = source.expanduser().resolve()
        if source.is_dir():
            images = sorted(
                p
                for p in source.iterdir()
                if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
            )
        else:
            images = [source]
    else:
        if split is None:
            split = "val"
        images = list(iter_images(dataset_root, split))
    return images[:limit] if limit > 0 else images


def detections_from_ultralytics(weights: Path, image_paths: list[Path], conf: float) -> dict[Path, list[Detection]]:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            "Ultralytics is not installed. Install requirements or run without --weights "
            "to use label-based demo outputs."
        ) from exc

    model = YOLO(str(weights))
    outputs: dict[Path, list[Detection]] = {}
    for image_path in image_paths:
        result = model.predict(source=str(image_path), conf=conf, verbose=False)[0]
        detections: list[Detection] = []
        image_height, image_width = result.orig_shape[:2]
        for box in result.boxes:
            class_id = int(box.cls.item())
            if class_id not in TARGET_CLASS_IDS:
                continue
            x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
            detections.append(
                Detection(
                    class_id=class_id,
                    x_center=((x1 + x2) / 2) / image_width,
                    y_center=((y1 + y2) / 2) / image_height,
                    width=(x2 - x1) / image_width,
                    height=(y2 - y1) / image_height,
                    confidence=float(box.conf.item()),
                )
            )
        outputs[image_path] = detections
    return outputs


def detections_from_labels(image_paths: list[Path]) -> dict[Path, list[Detection]]:
    outputs: dict[Path, list[Detection]] = {}
    for image_path in image_paths:
        outputs[image_path] = filter_targets(read_yolo_labels(label_path_for_image(image_path)))
    return outputs


def write_summary(rows: list[dict], output_dir: Path) -> None:
    json_path = output_dir / "count_summary.json"
    csv_path = output_dir / "count_summary.csv"
    json_path.write_text(json.dumps(rows, indent=2))
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["image", "humans", "cars", "drawn_boxes", "total_boxes", "output"],
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", default=".", type=Path)
    parser.add_argument("--split", default="val")
    parser.add_argument("--source", type=Path)
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--output-dir", default=Path("outputs/predictions"), type=Path)
    parser.add_argument("--limit", default=8, type=int)
    parser.add_argument("--conf", default=0.25, type=float)
    parser.add_argument("--use-labels", action="store_true", help="Force label-based demo mode.")
    args = parser.parse_args()

    dataset_root = args.dataset_root.expanduser().resolve()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    images = source_images(dataset_root, args.split, args.source, args.limit)
    if not images:
        images = find_labeled_images(dataset_root, args.split, limit=args.limit)
    if not images:
        print("No input images found.", file=sys.stderr)
        return 2

    if args.weights and not args.use_labels:
        detections_by_image = detections_from_ultralytics(args.weights, images, args.conf)
        title = "YOLO prediction"
    else:
        detections_by_image = detections_from_labels(images)
        title = "Ground-truth label demo"

    rows: list[dict] = []
    for image_path, detections in detections_by_image.items():
        output_path = output_dir / f"{image_path.stem}_count.jpg"
        counts = draw_detections(image_path, detections, output_path, title=title)
        rows.append(
            {
                "image": str(image_path),
                "humans": counts["humans"],
                "cars": counts["cars"],
                "drawn_boxes": counts["drawn_boxes"],
                "total_boxes": counts["total_boxes"],
                "output": str(output_path),
            }
        )
    write_summary(rows, output_dir)
    print(f"Processed {len(rows)} images into {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Evaluate a trained YOLO model on VisDrone validation or test-dev images."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from scripts.train_yolo import validate_dataset, write_resolved_yaml


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", default=".", type=Path)
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--split", default="val", choices=["val", "test"])
    parser.add_argument("--imgsz", default=960, type=int)
    parser.add_argument("--conf", default=0.25, type=float)
    args = parser.parse_args()

    dataset_root = args.dataset_root.expanduser().resolve()
    validate_dataset(dataset_root)
    resolved_yaml = write_resolved_yaml(dataset_root, Path("outputs/evaluation/visdrone_resolved.yaml"))

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit(
            "Ultralytics/PyTorch is not installed. Install requirements before evaluating model weights."
        ) from exc

    model = YOLO(str(args.weights))
    metrics = model.val(
        data=str(resolved_yaml),
        split=args.split,
        imgsz=args.imgsz,
        conf=args.conf,
        project="outputs/evaluation",
        name=f"{args.weights.stem}-{args.split}",
    )
    print(metrics)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

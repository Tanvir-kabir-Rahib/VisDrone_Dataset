#!/usr/bin/env python3
"""Train/fine-tune a YOLO detector on the VisDrone dataset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from visdrone_project.constants import CLASS_NAMES
from visdrone_project.dataset import images_dir


def parse_cache(value: str) -> bool | str:
    normalized = value.lower().strip()
    if normalized in {"false", "0", "none", "no", "off"}:
        return False
    if normalized in {"true", "1", "yes", "on"}:
        return True
    if normalized in {"ram", "disk"}:
        return normalized
    raise argparse.ArgumentTypeError("cache must be one of: false, true, ram, disk")


def write_resolved_yaml(dataset_root: Path, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "path": str(dataset_root),
        "train": "VisDrone2019-DET-train/images",
        "val": "VisDrone2019-DET-val/images",
        "test": "VisDrone2019-DET-test-dev/images",
        "nc": len(CLASS_NAMES),
        "names": CLASS_NAMES,
    }
    output_path.write_text(yaml.safe_dump(data, sort_keys=False))
    return output_path


def validate_dataset(dataset_root: Path) -> None:
    missing = []
    for split in ["train", "val", "test-dev"]:
        directory = images_dir(dataset_root, split)
        if not directory.exists():
            missing.append(str(directory))
    if missing:
        raise FileNotFoundError("Missing dataset directories: " + ", ".join(missing))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", default=".", type=Path)
    parser.add_argument("--model", default="yolo11n.pt", help="Ultralytics model checkpoint.")
    parser.add_argument("--epochs", default=50, type=int)
    parser.add_argument("--imgsz", default=640, type=int)
    parser.add_argument("--batch", default=16, type=int)
    parser.add_argument("--device", default=None)
    parser.add_argument("--project", default="outputs/training")
    parser.add_argument("--name", default="visdrone-yolo")
    parser.add_argument("--workers", default=8, type=int)
    parser.add_argument("--cache", default=False, type=parse_cache)
    parser.add_argument("--patience", default=10, type=int)
    parser.add_argument("--fraction", default=1.0, type=float, help="Use a subset of data for faster experiments.")
    parser.add_argument("--freeze", default=None, type=int, help="Freeze first N model layers for faster fine-tuning.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    dataset_root = args.dataset_root.expanduser().resolve()
    validate_dataset(dataset_root)
    resolved_yaml = write_resolved_yaml(dataset_root, Path("outputs/training/visdrone_resolved.yaml"))

    if args.dry_run:
        print("Dataset OK.")
        print(f"Resolved data YAML: {resolved_yaml}")
        print(
            "Training command: "
            f"python scripts/train_yolo.py --model {args.model} --epochs {args.epochs} "
            f"--imgsz {args.imgsz} --batch {args.batch} --workers {args.workers} "
            f"--cache {args.cache}"
        )
        print("Classes of interest for the assessment: pedestrian + people = human, car = car.")
        return 0

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit(
            "Ultralytics/PyTorch is not installed. Run `python3 -m pip install -r requirements.txt` "
            "or execute this script in a GPU/Colab environment."
        ) from exc

    model = YOLO(args.model)
    train_kwargs = {
        "data": str(resolved_yaml),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "project": args.project,
        "name": args.name,
        "patience": args.patience,
        "cache": args.cache,
        "workers": args.workers,
    }
    if args.device is not None:
        train_kwargs["device"] = args.device
    if args.fraction < 1.0:
        train_kwargs["fraction"] = args.fraction
    if args.freeze is not None:
        train_kwargs["freeze"] = args.freeze
    model.train(**train_kwargs)
    print(f"Training complete. Check {Path(args.project) / args.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Create dataset summary statistics and sample visualizations."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
CACHE_ROOT = Path(tempfile.gettempdir())
os.environ.setdefault("MPLCONFIGDIR", str(CACHE_ROOT / "matplotlib-codex-cache"))
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_ROOT / "codex-xdg-cache"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)

import matplotlib.pyplot as plt
import numpy as np

from visdrone_project.constants import CLASS_NAMES
from visdrone_project.dataset import (
    SPLITS,
    filter_targets,
    find_labeled_images,
    image_size,
    iter_images,
    label_path_for_image,
    read_yolo_labels,
)
from visdrone_project.visualize import draw_detections, make_grid


def summarize_split(dataset_root: Path, split: str) -> dict:
    images = list(iter_images(dataset_root, split))
    class_counts: Counter[int] = Counter()
    target_counts: Counter[str] = Counter()
    normalized_areas: list[float] = []
    labeled_images = 0
    empty_images = 0

    for image_path in images:
        detections = read_yolo_labels(label_path_for_image(image_path))
        if detections:
            labeled_images += 1
        else:
            empty_images += 1
        for detection in detections:
            class_counts[detection.class_id] += 1
            normalized_areas.append(detection.normalized_area)
            if detection.is_human:
                target_counts["human"] += 1
            elif detection.is_car:
                target_counts["car"] += 1

    return {
        "images": len(images),
        "labeled_images": labeled_images,
        "empty_images": empty_images,
        "boxes": int(sum(class_counts.values())),
        "target_boxes": dict(target_counts),
        "class_counts": {CLASS_NAMES.get(k, str(k)): int(v) for k, v in sorted(class_counts.items())},
        "normalized_box_area": {
            "mean": float(np.mean(normalized_areas)) if normalized_areas else 0.0,
            "median": float(np.median(normalized_areas)) if normalized_areas else 0.0,
            "p10": float(np.percentile(normalized_areas, 10)) if normalized_areas else 0.0,
            "p90": float(np.percentile(normalized_areas, 90)) if normalized_areas else 0.0,
        },
    }


def plot_class_distribution(summary: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    splits = [s for s in ["train", "val", "test-dev"] if s in summary["splits"]]
    classes = list(CLASS_NAMES.values())
    x = np.arange(len(classes))
    width = 0.25

    fig, ax = plt.subplots(figsize=(13, 6))
    for index, split in enumerate(splits):
        counts = [summary["splits"][split]["class_counts"].get(name, 0) for name in classes]
        ax.bar(x + (index - 1) * width, counts, width=width, label=split)

    ax.set_title("VisDrone class distribution")
    ax.set_ylabel("Bounding boxes")
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=35, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_target_distribution(summary: dict, output_path: Path) -> None:
    splits = [s for s in ["train", "val", "test-dev"] if s in summary["splits"]]
    labels = ["human", "car"]
    x = np.arange(len(labels))
    width = 0.25
    fig, ax = plt.subplots(figsize=(7, 5))
    for index, split in enumerate(splits):
        counts = [summary["splits"][split]["target_boxes"].get(label, 0) for label in labels]
        ax.bar(x + (index - 1) * width, counts, width=width, label=split)
    ax.set_title("Assessment target classes")
    ax.set_ylabel("Bounding boxes")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def collect_image_sizes(dataset_root: Path, split: str, limit: int) -> list[tuple[int, int]]:
    sizes = []
    for image_path in list(iter_images(dataset_root, split))[:limit]:
        sizes.append(image_size(image_path))
    return sizes


def plot_image_sizes(sizes_by_split: dict[str, list[tuple[int, int]]], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    for split, sizes in sizes_by_split.items():
        if not sizes:
            continue
        widths = [w for w, _ in sizes]
        heights = [h for _, h in sizes]
        ax.scatter(widths, heights, s=18, alpha=0.55, label=split)
    ax.set_title("Image size spread")
    ax.set_xlabel("width")
    ax.set_ylabel("height")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def create_sample_outputs(dataset_root: Path, output_dir: Path, split: str, limit: int) -> list[str]:
    sample_dir = output_dir / "sample_annotations"
    sample_dir.mkdir(parents=True, exist_ok=True)
    image_paths = find_labeled_images(dataset_root, split, limit=limit)
    annotated_paths: list[str] = []

    for image_path in image_paths:
        detections = filter_targets(read_yolo_labels(label_path_for_image(image_path)))
        output_path = sample_dir / f"{image_path.stem}_targets.jpg"
        draw_detections(
            image_path,
            detections,
            output_path,
            title="Ground-truth targets",
            include_other_classes=False,
        )
        annotated_paths.append(str(output_path))

    if annotated_paths:
        make_grid(annotated_paths, output_dir / "sample_grid.jpg", columns=2, tile_width=620)
    return annotated_paths


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", default=".", type=Path)
    parser.add_argument("--output-dir", default=Path("outputs/dataset"), type=Path)
    parser.add_argument("--sample-split", default="val", choices=SPLITS.keys())
    parser.add_argument("--sample-limit", default=6, type=int)
    parser.add_argument("--size-sample-limit", default=500, type=int)
    args = parser.parse_args()

    dataset_root = args.dataset_root.expanduser().resolve()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "dataset_root": str(dataset_root),
        "splits": {},
    }
    for split in ["train", "val", "test-dev", "test-challenge"]:
        summary["splits"][split] = summarize_split(dataset_root, split)

    sizes_by_split = {
        split: collect_image_sizes(dataset_root, split, args.size_sample_limit)
        for split in ["train", "val", "test-dev"]
    }
    summary["image_size_samples"] = {
        split: {
            "count": len(sizes),
            "unique_sizes": sorted({f"{w}x{h}" for w, h in sizes})[:20],
        }
        for split, sizes in sizes_by_split.items()
    }

    plot_class_distribution(summary, output_dir / "class_distribution.png")
    plot_target_distribution(summary, output_dir / "target_distribution.png")
    plot_image_sizes(sizes_by_split, output_dir / "image_size_spread.png")
    summary["sample_outputs"] = create_sample_outputs(
        dataset_root, output_dir, args.sample_split, args.sample_limit
    )

    summary_path = output_dir / "dataset_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"Wrote {summary_path}")
    print(f"Wrote visualizations to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

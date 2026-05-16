#!/usr/bin/env python3
"""Create a silent GIF slideshow from generated assessment outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def font(size: int) -> ImageFont.ImageFont:
    for candidate in [
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial.ttf",
    ]:
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def fit_image(image: Image.Image, max_width: int, max_height: int) -> Image.Image:
    scale = min(max_width / image.width, max_height / image.height)
    size = (int(image.width * scale), int(image.height * scale))
    return image.resize(size, Image.Resampling.LANCZOS)


def make_slide(title: str, subtitle: str, image_path: Path | None, size: tuple[int, int]) -> Image.Image:
    width, height = size
    canvas = Image.new("RGB", size, (246, 248, 251))
    draw = ImageDraw.Draw(canvas)
    title_font = font(42)
    subtitle_font = font(24)

    draw.rectangle((0, 0, width, 108), fill=(22, 28, 38))
    draw.text((42, 26), title, fill=(255, 255, 255), font=title_font)
    draw.text((44, 76), subtitle, fill=(202, 211, 224), font=subtitle_font)

    if image_path is not None and image_path.exists():
        image = Image.open(image_path).convert("RGB")
        fitted = fit_image(image, width - 84, height - 150)
        x = (width - fitted.width) // 2
        y = 128 + (height - 150 - fitted.height) // 2
        canvas.paste(fitted, (x, y))
    return canvas


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=Path("outputs/demo/visdrone_demo.gif"), type=Path)
    parser.add_argument("--duration-ms", default=2200, type=int)
    args = parser.parse_args()

    slides = [
        (
            "VisDrone Human and Car Detection",
            "Dataset understanding, visualization, counting, and YOLO training pipeline",
            None,
        ),
        (
            "Class Distribution",
            "All VisDrone classes are retained during training",
            Path("outputs/dataset/class_distribution.png"),
        ),
        (
            "Assessment Targets",
            "pedestrian + people = human, car = car",
            Path("outputs/dataset/target_distribution.png"),
        ),
        (
            "Sample Ground Truth",
            "Bounding boxes validate parsing and target mapping",
            Path("outputs/dataset/sample_grid.jpg"),
        ),
    ]

    prediction_paths = sorted(Path("outputs/predictions").glob("*_count.jpg"))[:4]
    for index, path in enumerate(prediction_paths, start=1):
        slides.append(
            (
                f"Counting Output {index}",
                "Top banner displays total humans and cars",
                path,
            )
        )

    frames = [make_slide(title, subtitle, image_path, (1280, 720)) for title, subtitle, image_path in slides]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        args.output,
        save_all=True,
        append_images=frames[1:],
        duration=args.duration_ms,
        loop=0,
        optimize=True,
    )
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

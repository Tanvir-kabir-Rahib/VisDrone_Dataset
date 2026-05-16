"""Visualization helpers for VisDrone detection and counting outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw, ImageFont

from .constants import DRAW_COLORS
from .dataset import Detection


def _font(size: int = 14) -> ImageFont.ImageFont:
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial.ttf",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _text_box(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> tuple[int, int]:
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def draw_detections(
    image_path: str | Path,
    detections: Iterable[Detection],
    output_path: str | Path,
    title: str = "Detection result",
    include_other_classes: bool = False,
) -> dict[str, int]:
    image_path = Path(image_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    label_font = _font(14)
    title_font = _font(18)

    detections = list(detections)
    drawable = [d for d in detections if include_other_classes or d.is_target]
    human_count = sum(1 for d in detections if d.is_human)
    car_count = sum(1 for d in detections if d.is_car)

    for detection in drawable:
        label = detection.target_name if detection.is_target else detection.class_name
        color = DRAW_COLORS.get(label, DRAW_COLORS["other"])
        x1, y1, x2, y2 = detection.to_xyxy(*image.size)
        line_width = max(2, round(min(image.size) / 420))
        draw.rectangle((x1, y1, x2, y2), outline=color, width=line_width)
        confidence = "" if detection.confidence is None else f" {detection.confidence:.2f}"
        label_text = f"{label}{confidence}"
        text_w, text_h = _text_box(draw, label_text, label_font)
        text_y = max(0, y1 - text_h - 4)
        draw.rectangle((x1, text_y, x1 + text_w + 8, text_y + text_h + 4), fill=color)
        draw.text((x1 + 4, text_y + 2), label_text, fill=(255, 255, 255), font=label_font)

    banner = f"{title} | humans: {human_count} | cars: {car_count}"
    banner_w, banner_h = _text_box(draw, banner, title_font)
    draw.rectangle((0, 0, min(image.width, banner_w + 24), banner_h + 18), fill=(20, 24, 32))
    draw.text((12, 8), banner, fill=(255, 255, 255), font=title_font)

    image.save(output_path, quality=92)
    return {
        "humans": human_count,
        "cars": car_count,
        "drawn_boxes": len(drawable),
        "total_boxes": len(detections),
    }


def make_grid(
    image_paths: list[str | Path],
    output_path: str | Path,
    columns: int = 2,
    tile_width: int = 720,
    background: tuple[int, int, int] = (245, 247, 250),
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    images = [Image.open(p).convert("RGB") for p in image_paths]
    if not images:
        raise ValueError("No images were provided for the grid.")

    resized = []
    for image in images:
        scale = tile_width / image.width
        tile_height = int(round(image.height * scale))
        resized.append(image.resize((tile_width, tile_height)))

    rows = (len(resized) + columns - 1) // columns
    row_heights = []
    for row in range(rows):
        row_images = resized[row * columns : (row + 1) * columns]
        row_heights.append(max(img.height for img in row_images))

    padding = 18
    grid_width = columns * tile_width + (columns + 1) * padding
    grid_height = sum(row_heights) + (rows + 1) * padding
    grid = Image.new("RGB", (grid_width, grid_height), background)

    y = padding
    for row in range(rows):
        x = padding
        for image in resized[row * columns : (row + 1) * columns]:
            grid.paste(image, (x, y))
            x += tile_width + padding
        y += row_heights[row] + padding

    grid.save(output_path, quality=92)

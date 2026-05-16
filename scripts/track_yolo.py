#!/usr/bin/env python3
"""Optional ByteTrack/BoT-SORT tracking wrapper for trained YOLO weights."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--tracker", default="bytetrack.yaml", help="bytetrack.yaml or botsort.yaml")
    parser.add_argument("--conf", default=0.25, type=float)
    parser.add_argument("--project", default="outputs/tracking")
    parser.add_argument("--name", default="visdrone-tracking")
    args = parser.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit(
            "Ultralytics/PyTorch is not installed. Install requirements before running tracking."
        ) from exc

    model = YOLO(str(args.weights))
    model.track(
        source=str(args.source),
        tracker=args.tracker,
        conf=args.conf,
        save=True,
        persist=True,
        project=args.project,
        name=args.name,
    )
    print(f"Tracking outputs saved under {Path(args.project) / args.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

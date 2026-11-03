#!/usr/bin/env python3
"""Explicitly download an official Ultralytics YOLOv8 model into models/."""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "model",
        nargs="?",
        default="yolov8n-seg.pt",
        choices=("yolov8n.pt", "yolov8n-seg.pt"),
    )
    parser.add_argument(
        "--accept-agpl",
        action="store_true",
        help="confirm that the project will comply with Ultralytics AGPL-3.0 or has a commercial license",
    )
    args = parser.parse_args()
    if not args.accept_agpl:
        parser.error(
            "Ultralytics code and pretrained models are AGPL-3.0 by default. "
            "Review the project licensing requirements, then pass --accept-agpl."
        )
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit("ultralytics is not installed") from exc
    model_dir = Path(__file__).resolve().parents[1] / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    previous = Path.cwd()
    try:
        os.chdir(model_dir)
        YOLO(args.model)
    finally:
        os.chdir(previous)
    output = model_dir / args.model
    if not output.exists():
        raise SystemExit(f"download completed but model was not found at {output}")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


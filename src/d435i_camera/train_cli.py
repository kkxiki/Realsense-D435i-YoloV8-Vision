"""Command-line entry point for custom object training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

from .training import TrainingConfig, YoloTrainingService


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="XuK YOLO custom object training")
    parser.add_argument("--data", required=True, help="YOLO data.yaml")
    parser.add_argument("--model", default="models/yolov8n.pt", help="pretrained .pt model")
    parser.add_argument("--task", choices=["detect", "segment"], default="detect")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--project", default="training_runs")
    parser.add_argument("--name", default="custom_object")
    parser.add_argument("--cache", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    root = Path(__file__).resolve().parents[2]
    service = YoloTrainingService(root)
    config = TrainingConfig(
        dataset_yaml=args.data,
        base_model=args.model,
        task=args.task,
        epochs=args.epochs,
        image_size=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        patience=args.patience,
        project=args.project,
        name=args.name,
        cache=args.cache,
    )
    try:
        summary = service.validate(config)
        print(json.dumps({"validation": summary.as_dict()}, ensure_ascii=False, indent=2), flush=True)
        if args.validate_only:
            return 0

        def progress(event):  # type: ignore[no-untyped-def]
            print("XUK_TRAIN_PROGRESS=" + json.dumps(event, ensure_ascii=False), flush=True)

        result = service.train(config, progress=progress)
        print("XUK_TRAIN_RESULT=" + json.dumps(result.as_dict(), ensure_ascii=False), flush=True)
        return 0
    except Exception as exc:
        print("XUK_TRAIN_ERROR=" + str(exc), flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

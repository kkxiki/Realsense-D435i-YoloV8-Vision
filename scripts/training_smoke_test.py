#!/usr/bin/env python3
"""One-epoch synthetic smoke test for the custom training interface."""

from __future__ import annotations

import tempfile
from pathlib import Path

import cv2
import numpy as np
import yaml

from d435i_camera.training import TrainingConfig, YoloTrainingService


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory(prefix="xuk_training_smoke_") as directory:
        root = Path(directory)
        for split in ("train", "val"):
            (root / "images" / split).mkdir(parents=True)
            (root / "labels" / split).mkdir(parents=True)
            for index in range(2):
                image = np.full((160, 160, 3), 35, dtype=np.uint8)
                cv2.rectangle(image, (48, 30), (112, 130), (50, 160, 230), -1)
                stem = f"{split}_{index}"
                cv2.imwrite(str(root / "images" / split / f"{stem}.jpg"), image)
                (root / "labels" / split / f"{stem}.txt").write_text(
                    "0 0.5 0.5 0.4 0.625\n", encoding="utf-8"
                )
        dataset = root / "data.yaml"
        dataset.write_text(
            yaml.safe_dump(
                {
                    "path": ".",
                    "train": "images/train",
                    "val": "images/val",
                    "names": {0: "smoke_object"},
                },
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        service = YoloTrainingService(project_root)
        result = service.train(
            TrainingConfig(
                dataset_yaml=str(dataset),
                base_model=str(project_root / "models" / "yolov8n.pt"),
                epochs=1,
                image_size=160,
                batch=2,
                device="cpu",
                workers=0,
                patience=1,
                project=str(root / "runs"),
                name="smoke",
            )
        )
        if not result.best_model.is_file():
            raise RuntimeError("training smoke test did not produce best.pt")
        print(f"TRAINING_SMOKE_OK best={result.best_model}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

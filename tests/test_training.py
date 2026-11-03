from pathlib import Path

import pytest

from d435i_camera.training import TrainingConfig, YoloTrainingService, validate_yolo_dataset


def _dataset(root: Path, bad_coordinate: bool = False) -> Path:
    for split in ("train", "val"):
        (root / "images" / split).mkdir(parents=True)
        (root / "labels" / split).mkdir(parents=True)
        (root / "images" / split / f"{split}.jpg").write_bytes(b"test-image")
        coordinate = "1.2" if bad_coordinate else "0.5"
        (root / "labels" / split / f"{split}.txt").write_text(
            f"0 {coordinate} 0.5 0.2 0.3\n", encoding="utf-8"
        )
    yaml_path = root / "data.yaml"
    yaml_path.write_text(
        "path: .\ntrain: images/train\nval: images/val\nnames:\n  0: custom_part\n",
        encoding="utf-8",
    )
    return yaml_path


def test_training_dataset_validator_reports_counts(tmp_path: Path):
    summary = validate_yolo_dataset(_dataset(tmp_path))
    assert summary.class_names == {0: "custom_part"}
    assert summary.train_images == 1
    assert summary.val_images == 1
    assert summary.label_files == 2
    assert summary.labeled_objects == 2


def test_training_dataset_validator_rejects_non_normalized_coordinates(tmp_path: Path):
    with pytest.raises(ValueError, match="归一化坐标"):
        validate_yolo_dataset(_dataset(tmp_path, bad_coordinate=True))


def test_training_service_validates_local_base_model(tmp_path: Path):
    dataset = _dataset(tmp_path / "dataset")
    model = tmp_path / "base.pt"
    model.write_bytes(b"placeholder")
    service = YoloTrainingService(tmp_path)
    summary = service.validate(
        TrainingConfig(dataset_yaml=str(dataset), base_model=str(model), epochs=2)
    )
    assert summary.labeled_objects == 2

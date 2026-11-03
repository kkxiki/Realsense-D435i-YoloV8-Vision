from pathlib import Path

import cv2
import numpy as np
import pytest

from d435i_camera.annotation import (
    YoloAnnotationProject,
    YoloBox,
    initialize_detection_dataset,
)


def test_mouse_corners_are_saved_as_normalized_yolo_box(tmp_path: Path):
    project = initialize_detection_dataset(tmp_path / "data.yaml", ["水瓶", "杯子"])
    source = tmp_path / "source.jpg"
    assert cv2.imwrite(str(source), np.zeros((480, 848, 3), dtype=np.uint8))
    image = project.import_images([source], "train")[0]

    box = YoloBox.from_corners(0, 0.25, 0.20, 0.75, 0.80)
    label = project.save_boxes(image, [box])

    assert label.relative_to(tmp_path).as_posix() == "labels/train/source.txt"
    assert label.read_text(encoding="utf-8") == "0 0.500000 0.500000 0.500000 0.600000\n"
    restored = project.read_boxes(image)[0]
    assert restored.class_id == box.class_id
    assert restored.x_center == pytest.approx(box.x_center, abs=1e-6)
    assert restored.y_center == pytest.approx(box.y_center, abs=1e-6)
    assert restored.width == pytest.approx(box.width, abs=1e-6)
    assert restored.height == pytest.approx(box.height, abs=1e-6)


def test_import_names_are_unique_and_existing_class_ids_are_protected(tmp_path: Path):
    project = initialize_detection_dataset(tmp_path / "data.yaml", ["水瓶"])
    source = tmp_path / "same.png"
    assert cv2.imwrite(str(source), np.zeros((20, 20, 3), dtype=np.uint8))
    first, second = project.import_images([source, source], "val")
    assert first.name == "same.png"
    assert second.name == "same_1.png"

    project.save_boxes(first, [YoloBox.from_corners(0, 0.1, 0.1, 0.9, 0.9)])
    project.set_class_names(["水瓶", "手机"])
    assert project.class_names == {0: "水瓶", 1: "手机"}
    with pytest.raises(ValueError, match="不能删除、改名"):
        project.set_class_names(["瓶子", "手机"])


def test_invalid_tiny_or_out_of_bounds_boxes_are_rejected():
    with pytest.raises(ValueError):
        YoloBox(0, 0.5, 0.5, 0.0, 0.2)
    with pytest.raises(ValueError):
        YoloBox(0, 0.1, 0.5, 0.4, 0.2)

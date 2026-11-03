from pathlib import Path

import pytest

from d435i_camera.config import load_config
from d435i_camera.service import CameraModule


def test_model_loader_rejects_directory_without_changing_detector(tmp_path: Path):
    config = load_config(Path(__file__).parents[1] / "config" / "default.yaml")
    config.camera.backend = "mock"
    module = CameraModule(config)
    original = module.detector
    original_path = module.config.detector.model_path
    with pytest.raises(ValueError, match="选择的是目录"):
        module.load_detector(str(tmp_path))
    assert module.detector is original
    assert module.config.detector.model_path == original_path

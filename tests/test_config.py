from pathlib import Path

from d435i_camera.config import load_config, save_config


def test_default_board_is_user_confirmed_8_by_11_15mm():
    config = load_config(Path(__file__).parents[1] / "config" / "default.yaml")
    board = config.calibration.board
    assert (board.inner_corners_cols, board.inner_corners_rows) == (8, 11)
    assert board.square_size_mm == 15.0
    assert board.confirmed is True
    assert config.detector.classes == []  # Empty filter means all 80 COCO classes.
    assert config.gui.show_top_left_panel is False
    assert config.detector.model_path.endswith("yolov8n.pt")
    assert config.calibration.eye_to_hand_method == "SHAH"


def test_stream_config_can_be_persisted_and_reloaded(tmp_path):
    config = load_config(Path(__file__).parents[1] / "config" / "default.yaml")
    config.camera.width = 640
    config.camera.height = 480
    config.camera.fps = 15
    config.camera.depth_width = 848
    config.camera.depth_height = 480
    config.camera.depth_fps = 60
    destination = save_config(config, tmp_path / "runtime.yaml")
    loaded = load_config(destination)
    assert (loaded.camera.width, loaded.camera.height, loaded.camera.fps) == (640, 480, 15)
    assert (
        loaded.camera.depth_width,
        loaded.camera.depth_height,
        loaded.camera.depth_fps,
    ) == (848, 480, 60)

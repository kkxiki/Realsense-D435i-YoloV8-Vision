from pathlib import Path

import numpy as np

from d435i_camera.config import load_config
from d435i_camera.service import CameraModule


def test_mock_pipeline_exposes_rgb_depth_detection_pose_and_snapshot(tmp_path: Path):
    config_path = Path(__file__).parents[1] / "config" / "default.yaml"
    config = load_config(config_path)
    config.camera.backend = "mock"
    config.camera.serial = ""
    module = CameraModule(config)
    try:
        module.start()
        result = module.process_once()
        assert result.frame.color_bgr.shape == (config.camera.height, config.camera.width, 3)
        assert result.frame.depth_m is not None
        assert len(result.observations) == 2
        assert all(item.pose is not None for item in result.observations)
        assert any(item.detection.class_name == "bottle" for item in result.observations)
        profiles = module.list_stream_profiles()
        assert any((item.width, item.height, item.fps) == (640, 480, 15) for item in profiles)
        assert any(item.width == 1920 for item in module.list_color_profiles())
        assert any(item.fps == 60 for item in module.list_depth_profiles())
        configured = module.configure_camera(
            width=640,
            height=480,
            fps=15,
            enable_imu=False,
            restart_if_running=True,
        )
        assert configured["running"] is True
        assert (configured["width"], configured["height"], configured["fps"]) == (640, 480, 15)
        assert configured["color"] == configured["depth"]
        configured = module.configure_camera(
            color_width=640,
            color_height=480,
            color_fps=30,
            depth_width=848,
            depth_height=480,
            depth_fps=60,
            restart_if_running=True,
        )
        assert configured["color"] == {"width": 640, "height": 480, "fps": 30}
        assert configured["depth"] == {"width": 848, "height": 480, "fps": 60}
        result = module.process_once()
        assert result.frame.color_bgr.shape == (480, 640, 3)
        points, colors = module.point_cloud(stride=12)
        assert points.ndim == 2 and points.shape[1] == 3
        assert colors.shape == points.shape
        depth = module.depth_at(config.camera.width * 0.42, config.camera.height * 0.55)
        assert depth is not None and 0.5 < depth < 0.7
        xyz = module.deproject(config.camera.width * 0.42, config.camera.height * 0.55)
        assert xyz.shape == (3,)
        ply = module.save_point_cloud_ply(tmp_path / "cloud.ply", stride=20)
        assert ply.exists() and ply.read_text(encoding="ascii").startswith("ply\n")
        paths = module.capture_snapshot(tmp_path)
        assert paths.color.exists()
        assert paths.depth and paths.depth.exists()
        assert paths.metadata.exists()
    finally:
        module.close()

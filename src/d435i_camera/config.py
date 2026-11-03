"""Typed YAML configuration with strict validation and project-relative paths."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class FilterConfig:
    spatial: bool = True
    temporal: bool = True
    hole_filling: bool = False


@dataclass
class CameraConfig:
    backend: str = "realsense"
    serial: str = ""
    width: int = 848
    height: int = 480
    fps: int = 30
    depth_width: int = 848
    depth_height: int = 480
    depth_fps: int = 30
    timeout_ms: int = 3000
    enable_imu: bool = False
    align_depth_to_color: bool = True
    filters: FilterConfig = field(default_factory=FilterConfig)


@dataclass
class DepthConfig:
    min_m: float = 0.15
    max_m: float = 1.5
    roi_radius_px: int = 5
    object_band_m: float = 0.08
    point_sample_step: int = 3
    min_pose_points: int = 50


@dataclass
class DetectorConfig:
    enabled: bool = True
    model_path: str = ""
    confidence: float = 0.55
    iou: float = 0.45
    image_size: int = 640
    device: str = "cpu"
    classes: List[int] = field(default_factory=list)
    frame_interval: int = 1


@dataclass
class PoseConfig:
    method: str = "depth_pca"
    output_frame: str = "camera_color_optical_frame"
    camera_mount: str = "eye_in_hand"
    smoothing_alpha: float = 0.35
    track_max_center_distance_px: float = 80.0
    axis_length_m: float = 0.06


@dataclass
class BoardConfig:
    type: str = "chessboard"
    inner_corners_cols: int = 8
    inner_corners_rows: int = 11
    square_size_mm: float = 15.0
    confirmed: bool = True

    def require_confirmed(self) -> None:
        if not self.confirmed:
            raise ValueError("标定板参数尚未确认；请填写实测内角点和方格尺寸并勾选确认")
        if self.inner_corners_cols < 2 or self.inner_corners_rows < 2:
            raise ValueError("棋盘内角点列数和行数必须至少为 2")
        if self.square_size_mm <= 0:
            raise ValueError("棋盘方格尺寸必须大于 0 mm")


@dataclass
class CalibrationConfig:
    board: BoardConfig = field(default_factory=BoardConfig)
    min_intrinsic_samples: int = 12
    min_handeye_samples: int = 12
    handeye_method: str = "PARK"
    min_eye_to_hand_samples: int = 12
    eye_to_hand_method: str = "SHAH"
    intrinsic_file: str = "data/calibration/intrinsics.json"
    handeye_file: str = "data/calibration/tool_T_camera.json"
    eye_to_hand_file: str = "data/calibration/base_T_camera.json"


@dataclass
class PathConfig:
    capture_dir: str = "data/captures"
    log_dir: str = "data/logs"


@dataclass
class GuiConfig:
    language: str = "zh_CN"
    show_depth_colormap: bool = False
    show_pose_axes: bool = True
    show_top_left_panel: bool = True


@dataclass
class AppConfig:
    camera: CameraConfig = field(default_factory=CameraConfig)
    depth: DepthConfig = field(default_factory=DepthConfig)
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    pose: PoseConfig = field(default_factory=PoseConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    paths: PathConfig = field(default_factory=PathConfig)
    gui: GuiConfig = field(default_factory=GuiConfig)
    config_path: Optional[Path] = field(default=None, repr=False)
    project_root: Path = field(default_factory=lambda: Path.cwd(), repr=False)

    def resolve(self, value: str) -> Path:
        path = Path(value).expanduser()
        return path if path.is_absolute() else (self.project_root / path).resolve()

    def validate(self) -> None:
        if self.camera.backend not in {"realsense", "mock"}:
            raise ValueError("camera.backend must be 'realsense' or 'mock'")
        if min(
            self.camera.width,
            self.camera.height,
            self.camera.fps,
            self.camera.depth_width,
            self.camera.depth_height,
            self.camera.depth_fps,
        ) <= 0:
            raise ValueError("color/depth resolution and FPS must be positive")
        if not 0 < self.depth.min_m < self.depth.max_m:
            raise ValueError("depth range is invalid")
        if not 0.0 <= self.detector.confidence <= 1.0:
            raise ValueError("detector confidence must be in [0, 1]")
        if not 0.0 <= self.detector.iou <= 1.0:
            raise ValueError("detector IoU must be in [0, 1]")
        if self.detector.frame_interval < 1:
            raise ValueError("detector frame_interval must be at least 1")
        if not 0.0 < self.pose.smoothing_alpha <= 1.0:
            raise ValueError("pose smoothing alpha must be in (0, 1]")
        if self.pose.camera_mount not in {"eye_in_hand", "eye_to_hand"}:
            raise ValueError("pose.camera_mount must be 'eye_in_hand' or 'eye_to_hand'")


def _construct(cls: Any, values: Optional[Dict[str, Any]]) -> Any:
    values = values or {}
    allowed = set(cls.__dataclass_fields__)
    unknown = set(values) - allowed
    if unknown:
        raise ValueError(f"unknown {cls.__name__} settings: {sorted(unknown)}")
    return cls(**values)


def load_config(path: Optional[str | Path] = None) -> AppConfig:
    if path is None:
        project_root = Path(__file__).resolve().parents[2]
        candidate = project_root / "config" / "default.yaml"
    else:
        candidate = Path(path).expanduser().resolve()
        project_root = candidate.parent.parent if candidate.parent.name == "config" else candidate.parent

    raw: Dict[str, Any] = {}
    if candidate.exists():
        with candidate.open("r", encoding="utf-8") as stream:
            raw = yaml.safe_load(stream) or {}

    known_sections = {"camera", "depth", "detector", "pose", "calibration", "paths", "gui"}
    unknown = set(raw) - known_sections
    if unknown:
        raise ValueError(f"unknown configuration sections: {sorted(unknown)}")

    camera_raw = dict(raw.get("camera") or {})
    camera_filters = _construct(FilterConfig, camera_raw.pop("filters", None))
    calibration_raw = dict(raw.get("calibration") or {})
    board = _construct(BoardConfig, calibration_raw.pop("board", None))
    config = AppConfig(
        camera=_construct(CameraConfig, {**camera_raw, "filters": camera_filters}),
        depth=_construct(DepthConfig, raw.get("depth")),
        detector=_construct(DetectorConfig, raw.get("detector")),
        pose=_construct(PoseConfig, raw.get("pose")),
        calibration=_construct(CalibrationConfig, {**calibration_raw, "board": board}),
        paths=_construct(PathConfig, raw.get("paths")),
        gui=_construct(GuiConfig, raw.get("gui")),
        config_path=candidate,
        project_root=project_root,
    )
    config.validate()
    return config


def save_config(config: AppConfig, path: Optional[str | Path] = None) -> Path:
    """Persist the editable runtime settings without serializing internal path state."""
    destination = (
        Path(path).expanduser().resolve()
        if path is not None
        else (config.config_path or (config.project_root / "config" / "default.yaml"))
    )
    payload = {
        "camera": asdict(config.camera),
        "depth": asdict(config.depth),
        "detector": asdict(config.detector),
        "pose": asdict(config.pose),
        "calibration": asdict(config.calibration),
        "paths": asdict(config.paths),
        "gui": asdict(config.gui),
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="\n") as stream:
        yaml.safe_dump(payload, stream, allow_unicode=True, sort_keys=False)
    config.config_path = destination
    return destination

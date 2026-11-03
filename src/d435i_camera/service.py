"""Facade exposing every camera/perception/calibration function to GUI or robot code."""

from __future__ import annotations

import json
import threading
import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .calibration import (
    EyeToHandCalibrationSession,
    HandEyeCalibrationSession,
    IntrinsicCalibrationSession,
    load_eye_to_hand,
    load_handeye,
)
from .camera import CameraBackend, MockCamera, RealSenseCamera
from .config import AppConfig, save_config
from .detection import NullDetector, ObjectDetector, SyntheticDetector, YoloDetector
from .geometry import deproject_pixel, invert_transform, robust_depth
from .models import (
    CameraDeviceInfo,
    Detection2D,
    FrameBundle,
    ObjectObservation,
    PerceptionFrame,
    SnapshotPaths,
    StreamProfile,
)
from .overlay import draw_overlay
from .pose import ObjectPoseEstimator, transform_pose
from .robot_pose import RobotPoseProvider


class CameraModule:
    """Thread-safe, synchronous module used by the GUI worker and external applications.

    Public operations intentionally separate acquisition, detection and pose estimation so
    downstream robot code can use only the layer it needs.
    """

    def __init__(
        self,
        config: AppConfig,
        camera: Optional[CameraBackend] = None,
        detector: Optional[ObjectDetector] = None,
        robot_pose_provider: Optional[RobotPoseProvider] = None,
    ) -> None:
        self.config = config
        self._lock = threading.RLock()
        self.camera = camera or self._make_camera()
        self.detector = detector or self._make_detector()
        self.pose_estimator = ObjectPoseEstimator(config.depth, config.pose)
        self.robot_pose_provider = robot_pose_provider
        self._tool_T_camera: Optional[np.ndarray] = None
        self._base_T_camera: Optional[np.ndarray] = None
        self._latest: Optional[PerceptionFrame] = None
        self._latest_raw: Optional[FrameBundle] = None
        self._sequence = 0
        self._last_detections: List[Detection2D] = []
        self._load_handeye_if_available()
        self._load_eye_to_hand_if_available()

    def _make_camera(self) -> CameraBackend:
        if self.config.camera.backend == "mock":
            return MockCamera(self.config.camera)
        return RealSenseCamera(self.config.camera)

    def _make_detector(self) -> ObjectDetector:
        if self.config.camera.backend == "mock":
            return SyntheticDetector()
        if not self.config.detector.enabled or not self.config.detector.model_path:
            return NullDetector()
        model = self.config.detector.model_path
        model_path = Path(model).expanduser()
        if not model_path.is_absolute():
            model_path = self.config.resolve(model)
        detector_config = self.config.detector
        detector_config.model_path = str(model_path)
        return YoloDetector(detector_config)

    def _load_handeye_if_available(self) -> None:
        path = self.config.resolve(self.config.calibration.handeye_file)
        if path.exists():
            self._tool_T_camera = load_handeye(path)

    def _load_eye_to_hand_if_available(self) -> None:
        path = self.config.resolve(self.config.calibration.eye_to_hand_file)
        if path.exists():
            self._base_T_camera = load_eye_to_hand(path)

    @property
    def running(self) -> bool:
        return self.camera.running

    @property
    def latest(self) -> Optional[PerceptionFrame]:
        with self._lock:
            return self._latest

    @property
    def latest_raw(self) -> Optional[FrameBundle]:
        with self._lock:
            return self._latest_raw

    @property
    def latest_imu(self):
        frame = self.latest_raw
        return frame.imu if frame is not None else None

    @property
    def tool_T_camera(self) -> Optional[np.ndarray]:
        return None if self._tool_T_camera is None else self._tool_T_camera.copy()

    def set_tool_T_camera(self, matrix: np.ndarray) -> None:
        from .geometry import validate_transform

        self._tool_T_camera = validate_transform(matrix).copy()

    @property
    def base_T_camera(self) -> Optional[np.ndarray]:
        """Fixed head-camera transform (camera coordinates -> robot base coordinates)."""

        return None if self._base_T_camera is None else self._base_T_camera.copy()

    def set_base_T_camera(self, matrix: np.ndarray) -> None:
        from .geometry import validate_transform

        self._base_T_camera = validate_transform(matrix).copy()

    def set_robot_pose_provider(self, provider: Optional[RobotPoseProvider]) -> None:
        self.robot_pose_provider = provider

    def list_devices(self) -> List[CameraDeviceInfo]:
        return type(self.camera).list_devices()

    def list_stream_profiles(self) -> List[StreamProfile]:
        """Compatibility API: profiles with identical RGB and depth size/FPS."""
        return self.camera.list_stream_profiles()

    def list_color_profiles(self) -> List[StreamProfile]:
        """List native RGB BGR8 profiles independently from depth."""

        return self.camera.list_color_profiles()

    def list_depth_profiles(self) -> List[StreamProfile]:
        """List native depth Z16 profiles independently from RGB."""

        return self.camera.list_depth_profiles()

    def get_stream_config(self) -> Dict[str, Any]:
        camera = self.config.camera
        return {
            "width": camera.width,
            "height": camera.height,
            "fps": camera.fps,
            "color": {"width": camera.width, "height": camera.height, "fps": camera.fps},
            "depth": {
                "width": camera.depth_width,
                "height": camera.depth_height,
                "fps": camera.depth_fps,
            },
            "enable_imu": camera.enable_imu,
            "align_depth_to_color": camera.align_depth_to_color,
            "running": self.running,
            "serial": camera.serial,
        }

    def configure_camera(
        self,
        *,
        width: Optional[int] = None,
        height: Optional[int] = None,
        fps: Optional[int] = None,
        color_width: Optional[int] = None,
        color_height: Optional[int] = None,
        color_fps: Optional[int] = None,
        depth_width: Optional[int] = None,
        depth_height: Optional[int] = None,
        depth_fps: Optional[int] = None,
        enable_imu: Optional[bool] = None,
        controls: Optional[Dict[str, float]] = None,
        restart_if_running: bool = True,
        validate_streams: bool = True,
        persist: bool = False,
    ) -> Dict[str, Any]:
        """Apply stream and hardware settings through one public interface.

        RGB and depth profiles can be selected independently. Legacy ``width/height/fps``
        arguments set both streams and remain available for existing robot code. Any stream
        change requires a pipeline restart; an unsupported RGB/depth combination is rolled back.
        By default, a stopped camera is opened briefly to validate the selected pair before saving.
        """
        def selected(specific: Optional[int], legacy: Optional[int], current: int) -> int:
            return int(specific if specific is not None else legacy if legacy is not None else current)

        color_requested = (
            selected(color_width, width, self.config.camera.width),
            selected(color_height, height, self.config.camera.height),
            selected(color_fps, fps, self.config.camera.fps),
        )
        depth_requested = (
            selected(depth_width, width, self.config.camera.depth_width),
            selected(depth_height, height, self.config.camera.depth_height),
            selected(depth_fps, fps, self.config.camera.depth_fps),
        )
        color_available = {
            (item.width, item.height, item.fps) for item in self.list_color_profiles()
        }
        depth_available = {
            (item.width, item.height, item.fps) for item in self.list_depth_profiles()
        }
        if color_requested not in color_available:
            raise ValueError(
                f"相机不支持 RGB 配置 {color_requested[0]}×{color_requested[1]}@{color_requested[2]}"
            )
        if depth_requested not in depth_available:
            raise ValueError(
                f"相机不支持 Depth 配置 {depth_requested[0]}×{depth_requested[1]}@{depth_requested[2]}"
            )

        with self._lock:
            old = (
                self.config.camera.width,
                self.config.camera.height,
                self.config.camera.fps,
                self.config.camera.depth_width,
                self.config.camera.depth_height,
                self.config.camera.depth_fps,
                self.config.camera.enable_imu,
            )
            was_running = self.running
            self.camera.stop()
            self.config.camera.width = color_requested[0]
            self.config.camera.height = color_requested[1]
            self.config.camera.fps = color_requested[2]
            self.config.camera.depth_width = depth_requested[0]
            self.config.camera.depth_height = depth_requested[1]
            self.config.camera.depth_fps = depth_requested[2]
            if enable_imu is not None:
                self.config.camera.enable_imu = bool(enable_imu)
            self.camera = self._make_camera()
            self._latest = None
            self._latest_raw = None
            self._last_detections = []
            self.pose_estimator.smoother.reset()
            restarted_device: Optional[CameraDeviceInfo] = None
            started_for_validation = False
            try:
                if was_running and restart_if_running:
                    restarted_device = self.camera.start()
                elif validate_streams:
                    restarted_device = self.camera.start()
                    started_for_validation = True
                if controls:
                    if not self.camera.running:
                        raise ValueError("设置曝光/增益等硬件参数时，相机必须处于运行状态")
                    for name, value in controls.items():
                        self.camera.set_control(name, value)
                if started_for_validation:
                    self.camera.stop()
                saved_path = str(save_config(self.config)) if persist else None
            except Exception:
                self.camera.stop()
                (
                    self.config.camera.width,
                    self.config.camera.height,
                    self.config.camera.fps,
                    self.config.camera.depth_width,
                    self.config.camera.depth_height,
                    self.config.camera.depth_fps,
                    self.config.camera.enable_imu,
                ) = old
                self.camera = self._make_camera()
                if was_running and restart_if_running:
                    try:
                        self.camera.start()
                    except Exception:
                        pass
                raise
            result = self.get_stream_config()
            result["saved_config"] = saved_path
            result["device"] = vars(restarted_device) if restarted_device else None
            return result

    def set_controls(self, values: Dict[str, float]) -> Dict[str, Dict[str, Any]]:
        """Apply multiple exposure/depth controls and return the refreshed values."""
        for name, value in values.items():
            self.camera.set_control(name, value)
        return self.camera.get_controls()

    def start(self) -> CameraDeviceInfo:
        with self._lock:
            info = self.camera.start()
            self.pose_estimator.smoother.reset()
            self._sequence = 0
            return info

    def stop(self) -> None:
        with self._lock:
            self.camera.stop()

    def close(self) -> None:
        self.stop()
        provider = self.robot_pose_provider
        if provider is not None and hasattr(provider, "close"):
            provider.close()  # type: ignore[attr-defined]

    def read_frame(self, timeout_ms: Optional[int] = None) -> FrameBundle:
        frame = self.camera.get_frame(timeout_ms)
        with self._lock:
            self._latest_raw = frame
        return frame

    def detect(self, frame: FrameBundle) -> Tuple[List[Detection2D], float]:
        return self.detector.detect(frame)

    def estimate_objects(
        self, frame: FrameBundle, detections: Sequence[Detection2D]
    ) -> Tuple[List[ObjectObservation], Optional[np.ndarray]]:
        camera_observations = self.pose_estimator.estimate_all(
            detections, frame, self._sequence
        )
        target_T_camera: Optional[np.ndarray] = None
        desired_frame = self.config.pose.output_frame
        if desired_frame != frame.camera_frame:
            if self.config.pose.camera_mount == "eye_to_hand":
                target_T_camera = self._base_T_camera
            else:
                provider = self.robot_pose_provider
                robot_pose = provider.latest(max_age_s=0.5) if provider else None
                if robot_pose is not None and self._tool_T_camera is not None:
                    target_T_camera = robot_pose.base_T_tool @ self._tool_T_camera
            if target_T_camera is not None:
                output: List[ObjectObservation] = []
                for item in camera_observations:
                    transformed = (
                        transform_pose(item.pose, target_T_camera, desired_frame)
                        if item.pose is not None
                        else None
                    )
                    output.append(
                        ObjectObservation(
                            detection=item.detection,
                            pose=transformed,
                            valid=item.valid,
                            message=item.message,
                            depth_m=item.depth_m,
                        )
                    )
                return output, target_T_camera
            # Preserve measurable camera-frame output and explicitly report why base is absent.
            missing = (
                "fixed base_T_camera missing"
                if self.config.pose.camera_mount == "eye_to_hand"
                else "robot pose or tool_T_camera missing"
            )
            camera_observations = [
                ObjectObservation(
                    item.detection,
                    item.pose,
                    False,
                    (item.message + "; " if item.message else "")
                    + f"base transform unavailable ({missing})",
                    item.depth_m,
                )
                for item in camera_observations
            ]
        return camera_observations, target_T_camera

    def process_once(self, timeout_ms: Optional[int] = None) -> PerceptionFrame:
        started = time.perf_counter()
        frame = self.read_frame(timeout_ms)
        run_detector = self._sequence % self.config.detector.frame_interval == 0
        if run_detector:
            detections, inference_ms = self.detect(frame)
            self._last_detections = detections
        else:
            detections, inference_ms = list(self._last_detections), 0.0
        observations, target_T_camera = self.estimate_objects(frame, detections)
        processing_ms = (time.perf_counter() - started) * 1000.0
        annotated = draw_overlay(
            frame,
            observations,
            processing_ms=processing_ms,
            show_panel=self.config.gui.show_top_left_panel,
            show_axes=self.config.gui.show_pose_axes,
            axis_length_m=self.config.pose.axis_length_m,
            camera_T_pose_frame=(
                invert_transform(target_T_camera) if target_T_camera is not None else None
            ),
        )
        result = PerceptionFrame(
            frame=frame,
            observations=observations,
            annotated_bgr=annotated,
            processing_ms=processing_ms,
            inference_ms=inference_ms,
            sequence=self._sequence,
        )
        with self._lock:
            self._latest = result
            self._sequence += 1
        return result

    def load_detector(self, model_path: str) -> None:
        value = str(model_path).strip()
        if not value:
            raise ValueError("请选择 YOLO 模型文件（.pt、.onnx 或 .engine）")
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = self.config.resolve(value)
        path = path.resolve()
        if not path.is_file():
            raise ValueError(f"YOLO 模型文件不存在或选择的是目录：{path}")
        if path.suffix.lower() not in {".pt", ".onnx", ".engine"}:
            raise ValueError(f"不支持的模型格式：{path.suffix or '无扩展名'}")
        detector_config = replace(
            self.config.detector,
            model_path=str(path),
            enabled=True,
        )
        detector = YoloDetector(detector_config)
        with self._lock:
            self.config.detector = detector_config
            self.detector = detector

    def unload_detector(self) -> None:
        self.detector = NullDetector()
        self.config.detector.enabled = False

    def get_controls(self) -> Dict[str, Dict[str, Any]]:
        return self.camera.get_controls()

    def set_control(self, name: str, value: float) -> None:
        self.camera.set_control(name, value)

    def health(self) -> Dict[str, Any]:
        latest = self.latest
        return {
            "running": self.running,
            "backend": self.config.camera.backend,
            "serial": self.config.camera.serial,
            "detector_ready": self.detector.ready,
            "model_path": getattr(self.detector, "model_path", ""),
            "handeye_loaded": self._tool_T_camera is not None,
            "eye_to_hand_loaded": self._base_T_camera is not None,
            "camera_mount": self.config.pose.camera_mount,
            "robot_pose_available": bool(
                self.robot_pose_provider and self.robot_pose_provider.latest(0.5)
            ),
            "last_frame_number": latest.frame.frame_number if latest else None,
            "last_processing_ms": latest.processing_ms if latest else None,
            "objects": len(latest.observations) if latest else 0,
        }

    def point_cloud(
        self, frame: Optional[FrameBundle] = None, stride: int = 4
    ) -> Tuple[np.ndarray, np.ndarray]:
        frame = frame or self.latest_raw
        if frame is None or frame.depth_m is None:
            raise ValueError("没有可用的对齐深度帧")
        stride = max(1, int(stride))
        depth = frame.depth_m[::stride, ::stride]
        color = frame.color_bgr[::stride, ::stride]
        ys, xs = np.mgrid[0 : frame.intrinsics.height : stride, 0 : frame.intrinsics.width : stride]
        valid = (
            np.isfinite(depth)
            & (depth >= self.config.depth.min_m)
            & (depth <= self.config.depth.max_m)
        )
        z = depth[valid].astype(np.float64)
        x = (xs[valid] - frame.intrinsics.cx) * z / frame.intrinsics.fx
        y = (ys[valid] - frame.intrinsics.cy) * z / frame.intrinsics.fy
        return np.column_stack((x, y, z)), color[valid][:, ::-1].copy()

    def depth_at(
        self,
        u: float,
        v: float,
        frame: Optional[FrameBundle] = None,
        radius_px: Optional[int] = None,
    ) -> Optional[float]:
        frame = frame or self.latest_raw
        if frame is None or frame.depth_m is None:
            return None
        return robust_depth(
            frame.depth_m,
            u,
            v,
            radius=(self.config.depth.roi_radius_px if radius_px is None else radius_px),
            min_m=self.config.depth.min_m,
            max_m=self.config.depth.max_m,
        )

    def deproject(
        self,
        u: float,
        v: float,
        depth_m: Optional[float] = None,
        frame: Optional[FrameBundle] = None,
    ) -> np.ndarray:
        frame = frame or self.latest_raw
        if frame is None:
            raise ValueError("没有可用帧")
        value = depth_m if depth_m is not None else self.depth_at(u, v, frame)
        if value is None:
            raise ValueError(f"像素 ({u:.1f}, {v:.1f}) 周围没有有效深度")
        return deproject_pixel(u, v, value, frame.intrinsics)

    def save_point_cloud_ply(
        self,
        path: str | Path,
        frame: Optional[FrameBundle] = None,
        stride: int = 4,
    ) -> Path:
        points, colors = self.point_cloud(frame, stride)
        destination = Path(path).expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", encoding="ascii", newline="\n") as stream:
            stream.write("ply\nformat ascii 1.0\n")
            stream.write(f"element vertex {len(points)}\n")
            stream.write("property float x\nproperty float y\nproperty float z\n")
            stream.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
            stream.write("end_header\n")
            for point, color in zip(points, colors):
                stream.write(
                    f"{point[0]:.7f} {point[1]:.7f} {point[2]:.7f} "
                    f"{int(color[0])} {int(color[1])} {int(color[2])}\n"
                )
        return destination

    def capture_snapshot(
        self,
        directory: Optional[str | Path] = None,
        prefix: str = "capture",
    ) -> SnapshotPaths:
        result = self.latest
        frame = result.frame if result is not None else self.read_frame()
        root = (
            Path(directory).expanduser().resolve()
            if directory is not None
            else self.config.resolve(self.config.paths.capture_dir)
        )
        root.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        base = root / f"{prefix}_{stamp}"
        color_path = base.with_name(base.name + "_color.png")
        if not cv2.imwrite(str(color_path), frame.color_bgr):
            raise IOError(f"无法保存彩色图：{color_path}")
        depth_path: Optional[Path] = None
        if frame.depth_m is not None:
            depth_path = base.with_name(base.name + "_depth_m.npy")
            np.save(depth_path, frame.depth_m)
        metadata_path = base.with_name(base.name + "_metadata.json")
        payload = {
            "timestamp_s": frame.timestamp_s,
            "frame_number": frame.frame_number,
            "camera_frame": frame.camera_frame,
            "intrinsics": frame.intrinsics.as_dict(),
            "objects": [item.as_dict() for item in result.observations] if result else [],
            "depth_file": depth_path.name if depth_path else None,
            "color_file": color_path.name,
        }
        metadata_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return SnapshotPaths(color_path, depth_path, metadata_path)

    def export_factory_intrinsics(self, path: Optional[str | Path] = None) -> Path:
        intrinsics = self.camera.intrinsics
        if intrinsics is None:
            raise ValueError("相机未启动，无法读取出厂内参")
        destination = (
            Path(path).expanduser().resolve()
            if path is not None
            else self.config.resolve(self.config.calibration.intrinsic_file)
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema": "xuk.camera_intrinsics/v1",
            "source": "realsense_factory",
            "camera": intrinsics.as_dict(),
        }
        destination.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return destination

    def new_intrinsic_session(self) -> IntrinsicCalibrationSession:
        return IntrinsicCalibrationSession(
            self.config.calibration.board,
            self.config.calibration.min_intrinsic_samples,
        )

    def new_handeye_session(self) -> HandEyeCalibrationSession:
        intrinsics = self.camera.intrinsics
        if intrinsics is None:
            raise ValueError("相机未启动，无法创建手眼标定会话")
        return HandEyeCalibrationSession(
            self.config.calibration.board,
            intrinsics,
            self.config.calibration.min_handeye_samples,
            self.config.calibration.handeye_method,
        )

    def new_eye_to_hand_session(self) -> EyeToHandCalibrationSession:
        """Create calibration for a fixed head camera and board rigidly attached to tool."""

        intrinsics = self.camera.intrinsics
        if intrinsics is None:
            raise ValueError("相机未启动，无法创建眼在手外标定会话")
        return EyeToHandCalibrationSession(
            self.config.calibration.board,
            intrinsics,
            self.config.calibration.min_eye_to_hand_samples,
            self.config.calibration.eye_to_hand_method,
        )

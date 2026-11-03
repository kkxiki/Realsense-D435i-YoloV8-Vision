"""Shared, dependency-light data models used by every public interface."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class Intrinsics:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    distortion_model: str = "none"
    coeffs: Tuple[float, ...] = ()

    def validate(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("intrinsic image size must be positive")
        if not np.isfinite([self.fx, self.fy, self.cx, self.cy]).all():
            raise ValueError("intrinsics must be finite")
        if self.fx <= 0 or self.fy <= 0:
            raise ValueError("focal lengths must be positive")

    @property
    def matrix(self) -> np.ndarray:
        self.validate()
        return np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )

    @property
    def distortion(self) -> np.ndarray:
        return np.asarray(self.coeffs or (0.0, 0.0, 0.0, 0.0, 0.0), dtype=np.float64)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "width": self.width,
            "height": self.height,
            "fx": self.fx,
            "fy": self.fy,
            "cx": self.cx,
            "cy": self.cy,
            "distortion_model": self.distortion_model,
            "coeffs": list(self.coeffs),
        }


@dataclass(frozen=True)
class ImuSample:
    timestamp_s: float
    accel_m_s2: Optional[Tuple[float, float, float]] = None
    gyro_rad_s: Optional[Tuple[float, float, float]] = None


@dataclass
class FrameBundle:
    color_bgr: np.ndarray
    depth_m: Optional[np.ndarray]
    intrinsics: Intrinsics
    timestamp_s: float
    frame_number: int
    camera_frame: str = "camera_color_optical_frame"
    imu: Optional[ImuSample] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if self.color_bgr.ndim != 3 or self.color_bgr.shape[2] != 3:
            raise ValueError("color image must be HxWx3 BGR")
        height, width = self.color_bgr.shape[:2]
        if self.depth_m is not None and self.depth_m.shape != (height, width):
            raise ValueError("aligned depth dimensions must match color dimensions")
        if (width, height) != (self.intrinsics.width, self.intrinsics.height):
            raise ValueError("frame dimensions do not match intrinsics")
        self.intrinsics.validate()


@dataclass(frozen=True)
class CameraDeviceInfo:
    name: str
    serial: str
    firmware: str = ""
    usb_type: str = ""
    product_line: str = ""
    supports_imu: bool = False
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, order=True)
class StreamProfile:
    width: int
    height: int
    fps: int
    color_format: str = "bgr8"
    depth_format: str = "z16"

    @property
    def label(self) -> str:
        return f"{self.width} × {self.height} @ {self.fps} FPS"

    def as_dict(self) -> Dict[str, Any]:
        return {
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "color_format": self.color_format,
            "depth_format": self.depth_format,
        }


@dataclass(frozen=True)
class Detection2D:
    class_id: int
    class_name: str
    confidence: float
    bbox_xyxy: Tuple[float, float, float, float]
    track_id: Optional[int] = None
    mask: Optional[np.ndarray] = field(default=None, compare=False, repr=False)

    @property
    def center(self) -> Tuple[float, float]:
        x1, y1, x2, y2 = self.bbox_xyxy
        return ((x1 + x2) * 0.5, (y1 + y2) * 0.5)

    @property
    def size(self) -> Tuple[float, float]:
        x1, y1, x2, y2 = self.bbox_xyxy
        return (max(0.0, x2 - x1), max(0.0, y2 - y1))

    def clipped(self, width: int, height: int) -> "Detection2D":
        x1, y1, x2, y2 = self.bbox_xyxy
        return Detection2D(
            class_id=self.class_id,
            class_name=self.class_name,
            confidence=self.confidence,
            bbox_xyxy=(
                float(np.clip(x1, 0, width - 1)),
                float(np.clip(y1, 0, height - 1)),
                float(np.clip(x2, 0, width - 1)),
                float(np.clip(y2, 0, height - 1)),
            ),
            track_id=self.track_id,
            mask=self.mask,
        )


@dataclass(frozen=True)
class Pose6D:
    translation_m: Tuple[float, float, float]
    rotation_matrix: np.ndarray = field(compare=False, repr=False)
    quaternion_xyzw: Tuple[float, float, float, float]
    rpy_deg: Tuple[float, float, float]
    frame_id: str
    method: str
    quality: float
    dimensions_m: Tuple[float, float, float] = (0.0, 0.0, 0.0)

    @property
    def matrix(self) -> np.ndarray:
        matrix = np.eye(4, dtype=np.float64)
        matrix[:3, :3] = np.asarray(self.rotation_matrix, dtype=np.float64)
        matrix[:3, 3] = np.asarray(self.translation_m, dtype=np.float64)
        return matrix

    def as_dict(self) -> Dict[str, Any]:
        return {
            "translation_m": list(self.translation_m),
            "quaternion_xyzw": list(self.quaternion_xyzw),
            "rpy_deg": list(self.rpy_deg),
            "frame_id": self.frame_id,
            "method": self.method,
            "quality": self.quality,
            "dimensions_m": list(self.dimensions_m),
        }


@dataclass(frozen=True)
class ObjectObservation:
    detection: Detection2D
    pose: Optional[Pose6D]
    valid: bool
    message: str = ""
    depth_m: Optional[float] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "class_id": self.detection.class_id,
            "class_name": self.detection.class_name,
            "confidence": self.detection.confidence,
            "bbox_xyxy": list(self.detection.bbox_xyxy),
            "track_id": self.detection.track_id,
            "valid": self.valid,
            "message": self.message,
            "depth_m": self.depth_m,
            "pose": self.pose.as_dict() if self.pose else None,
        }


@dataclass
class PerceptionFrame:
    frame: FrameBundle
    observations: List[ObjectObservation]
    annotated_bgr: np.ndarray
    processing_ms: float
    inference_ms: float
    sequence: int

    def as_dict(self) -> Dict[str, Any]:
        return {
            "timestamp_s": self.frame.timestamp_s,
            "frame_number": self.frame.frame_number,
            "processing_ms": self.processing_ms,
            "inference_ms": self.inference_ms,
            "sequence": self.sequence,
            "objects": [item.as_dict() for item in self.observations],
        }


@dataclass(frozen=True)
class SnapshotPaths:
    color: Path
    depth: Optional[Path]
    metadata: Path

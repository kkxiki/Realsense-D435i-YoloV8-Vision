"""Deterministic RGB-D scene used for GUI demos and regression tests."""

from __future__ import annotations

import math
import threading
import time
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from ..config import CameraConfig
from ..models import CameraDeviceInfo, FrameBundle, Intrinsics, StreamProfile
from .base import CameraBackend, CameraNotStartedError


class MockCamera(CameraBackend):
    def __init__(self, config: CameraConfig) -> None:
        self.config = config
        self._running = False
        self._frame_number = 0
        self._lock = threading.RLock()
        self._intrinsics = Intrinsics(
            width=config.width,
            height=config.height,
            fx=config.width * 0.82,
            fy=config.width * 0.82,
            cx=(config.width - 1) / 2.0,
            cy=(config.height - 1) / 2.0,
        )

    @classmethod
    def list_devices(cls) -> List[CameraDeviceInfo]:
        return [
            CameraDeviceInfo(
                name="Synthetic RGB-D Camera",
                serial="MOCK-D435I",
                firmware="0.1",
                usb_type="virtual",
                product_line="mock",
                supports_imu=False,
            )
        ]

    @property
    def running(self) -> bool:
        return self._running

    @property
    def intrinsics(self) -> Optional[Intrinsics]:
        return self._intrinsics

    def start(self) -> CameraDeviceInfo:
        with self._lock:
            self._running = True
            return self.list_devices()[0]

    def stop(self) -> None:
        with self._lock:
            self._running = False

    def get_frame(self, timeout_ms: Optional[int] = None) -> FrameBundle:
        del timeout_ms
        with self._lock:
            if not self._running:
                raise CameraNotStartedError("模拟相机尚未启动")
            width, height = self.config.width, self.config.height
            t = self._frame_number / max(1, self.config.fps)
            image = np.zeros((height, width, 3), dtype=np.uint8)
            image[:] = (30, 35, 42)
            # A mild grid makes dropped frames and motion easy to see.
            for x in range(0, width, 60):
                cv2.line(image, (x, 0), (x, height), (42, 48, 56), 1)
            for y in range(0, height, 60):
                cv2.line(image, (0, y), (width, y), (42, 48, 56), 1)

            bottle_w = max(42, width // 13)
            bottle_h = max(120, height // 3)
            center_x = int(width * 0.42 + math.sin(t * 0.8) * width * 0.12)
            center_y = int(height * 0.55 + math.cos(t * 0.6) * height * 0.05)
            bottle_box = (
                center_x - bottle_w // 2,
                center_y - bottle_h // 2,
                center_x + bottle_w // 2,
                center_y + bottle_h // 2,
            )
            x1, y1, x2, y2 = bottle_box
            cv2.rectangle(image, (x1, y1 + 18), (x2, y2), (210, 145, 40), -1)
            cv2.rectangle(
                image,
                (center_x - bottle_w // 4, y1),
                (center_x + bottle_w // 4, y1 + 25),
                (225, 165, 55),
                -1,
            )

            cup_r = max(35, height // 10)
            cup_center = (int(width * 0.70), int(height * 0.62))
            cup_box = (
                cup_center[0] - cup_r,
                cup_center[1] - cup_r,
                cup_center[0] + cup_r,
                cup_center[1] + cup_r,
            )
            cv2.circle(image, cup_center, cup_r, (70, 175, 100), -1)

            depth = np.full((height, width), 1.15, dtype=np.float32)
            depth[y1:y2, x1:x2] = 0.58 + np.linspace(-0.025, 0.025, x2 - x1)[None, :]
            cx1, cy1, cx2, cy2 = cup_box
            yy, xx = np.ogrid[:height, :width]
            cup_mask = (xx - cup_center[0]) ** 2 + (yy - cup_center[1]) ** 2 <= cup_r**2
            depth[cup_mask] = 0.73
            noise = np.random.default_rng(self._frame_number).normal(0, 0.0015, depth.shape)
            depth = (depth + noise).astype(np.float32)

            objects = [
                {"class_id": 39, "class_name": "bottle", "confidence": 0.94, "bbox": bottle_box},
                {"class_id": 41, "class_name": "cup", "confidence": 0.88, "bbox": cup_box},
            ]
            bundle = FrameBundle(
                color_bgr=image,
                depth_m=depth,
                intrinsics=self._intrinsics,
                timestamp_s=time.time(),
                frame_number=self._frame_number,
                metadata={
                    "serial": "MOCK-D435I",
                    "depth_aligned_to_color": True,
                    "mock_objects": objects,
                },
            )
            self._frame_number += 1
            return bundle

    def list_stream_profiles(self) -> List[StreamProfile]:
        return [
            StreamProfile(640, 480, 15),
            StreamProfile(640, 480, 30),
            StreamProfile(848, 480, 15),
            StreamProfile(848, 480, 30),
            StreamProfile(1280, 720, 15),
            StreamProfile(1280, 720, 30),
        ]

    def list_color_profiles(self) -> List[StreamProfile]:
        return self.list_stream_profiles() + [
            StreamProfile(1920, 1080, 15),
            StreamProfile(1920, 1080, 30),
        ]

    def list_depth_profiles(self) -> List[StreamProfile]:
        return self.list_stream_profiles() + [
            StreamProfile(424, 240, 30),
            StreamProfile(848, 480, 60),
        ]

    def get_controls(self) -> Dict[str, Dict[str, Any]]:
        result = {}
        for name, value, maximum in (
            ("color.exposure", 166.0, 10000.0),
            ("depth.exposure", 8500.0, 165000.0),
        ):
            result[name] = {
                "value": value,
                "min": 1.0,
                "max": maximum,
                "step": 1.0,
                "default": value,
                "read_only": False,
                "hardware_read_only": False,
                "protected": False,
                "sensor": "mock",
            }
        return result

    def set_control(self, name: str, value: float) -> None:
        if name not in {"color.exposure", "depth.exposure"}:
            raise ValueError(f"mock camera has no control {name}")
        del value

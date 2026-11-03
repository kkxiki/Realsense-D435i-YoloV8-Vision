"""Robot-pose provider contract used by eye-in-hand calibration and base transforms."""

from __future__ import annotations

import json
import socket
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import numpy as np

from .geometry import make_transform, pose_matrix_from_rpy, validate_transform


@dataclass(frozen=True)
class TimedRobotPose:
    base_T_tool: np.ndarray
    timestamp_s: float
    source: str


class RobotPoseProvider(ABC):
    @abstractmethod
    def latest(self, max_age_s: float = 0.5) -> Optional[TimedRobotPose]:
        raise NotImplementedError


class ManualRobotPoseProvider(RobotPoseProvider):
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._pose: Optional[TimedRobotPose] = None

    def update_matrix(
        self, base_T_tool: np.ndarray, timestamp_s: Optional[float] = None
    ) -> TimedRobotPose:
        pose = TimedRobotPose(
            validate_transform(base_T_tool).copy(),
            time.time() if timestamp_s is None else float(timestamp_s),
            "manual",
        )
        with self._lock:
            self._pose = pose
        return pose

    def update_rpy(
        self,
        xyz: Sequence[float],
        rpy_deg: Sequence[float],
        translation_unit: str = "mm",
    ) -> TimedRobotPose:
        if translation_unit not in {"m", "mm"}:
            raise ValueError("translation_unit must be m or mm")
        scale = 0.001 if translation_unit == "mm" else 1.0
        return self.update_matrix(pose_matrix_from_rpy(xyz, rpy_deg, scale))

    def latest(self, max_age_s: float = 0.5) -> Optional[TimedRobotPose]:
        with self._lock:
            pose = self._pose
        if pose is None or time.time() - pose.timestamp_s > max_age_s:
            return None
        return pose


class CallbackRobotPoseProvider(RobotPoseProvider):
    def __init__(self, callback: Callable[[], Optional[TimedRobotPose]]) -> None:
        self._callback = callback

    def latest(self, max_age_s: float = 0.5) -> Optional[TimedRobotPose]:
        pose = self._callback()
        if pose is None or time.time() - pose.timestamp_s > max_age_s:
            return None
        validate_transform(pose.base_T_tool)
        return pose


class UdpRobotPoseProvider(RobotPoseProvider):
    """Receive a small JSON pose packet from a ROS-Python bridge.

    This keeps the Python 3.10 GUI independent from ROS Foxy's Python 3.8 runtime.
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 17661) -> None:
        self.host = host
        self.port = int(port)
        self._manual = ManualRobotPoseProvider()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._socket: Optional[socket.socket] = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._listen, daemon=True, name="robot-pose-udp")
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._socket is not None:
            self._socket.close()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self._thread = None
        self._socket = None

    def _listen(self) -> None:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._socket = sock
        sock.settimeout(0.5)
        sock.bind((self.host, self.port))
        while not self._stop_event.is_set():
            try:
                payload, _address = sock.recvfrom(8192)
                data = json.loads(payload.decode("utf-8"))
                translation = np.asarray(data["translation"], dtype=np.float64).reshape(3)
                if data.get("translation_unit", "m") == "mm":
                    translation *= 0.001
                rotation_type = data.get("rotation_type", "rpy")
                if rotation_type == "matrix":
                    matrix = make_transform(np.asarray(data["rotation"]), translation)
                elif rotation_type == "rpy":
                    rpy = np.asarray(data["rotation"], dtype=np.float64).reshape(3)
                    if data.get("angle_unit", "rad") == "rad":
                        rpy = np.rad2deg(rpy)
                    matrix = pose_matrix_from_rpy(translation, rpy, translation_scale=1.0)
                else:
                    raise ValueError(f"unsupported rotation_type {rotation_type}")
                self._manual.update_matrix(matrix, data.get("timestamp_s", time.time()))
            except (OSError, ValueError, KeyError, json.JSONDecodeError):
                continue

    def latest(self, max_age_s: float = 0.5) -> Optional[TimedRobotPose]:
        pose = self._manual.latest(max_age_s)
        if pose is None:
            return None
        return TimedRobotPose(pose.base_T_tool, pose.timestamp_s, "ros_udp_bridge")

    def close(self) -> None:
        self.stop()


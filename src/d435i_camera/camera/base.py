"""Backend contract for hardware, recordings and deterministic test cameras."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from ..models import CameraDeviceInfo, FrameBundle, Intrinsics, StreamProfile


class CameraError(RuntimeError):
    pass


class CameraNotStartedError(CameraError):
    pass


class CameraBackend(ABC):
    @classmethod
    @abstractmethod
    def list_devices(cls) -> List[CameraDeviceInfo]:
        raise NotImplementedError

    @property
    @abstractmethod
    def running(self) -> bool:
        raise NotImplementedError

    @property
    @abstractmethod
    def intrinsics(self) -> Optional[Intrinsics]:
        raise NotImplementedError

    @abstractmethod
    def start(self) -> CameraDeviceInfo:
        raise NotImplementedError

    @abstractmethod
    def stop(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def get_frame(self, timeout_ms: Optional[int] = None) -> FrameBundle:
        raise NotImplementedError

    @abstractmethod
    def list_stream_profiles(self) -> List[StreamProfile]:
        """Return color/depth profile combinations supported at the same size and FPS."""

    def list_color_profiles(self) -> List[StreamProfile]:
        """Return native color profiles; backends may override for independent streams."""

        return self.list_stream_profiles()

    def list_depth_profiles(self) -> List[StreamProfile]:
        """Return native depth profiles; backends may override for independent streams."""

        return self.list_stream_profiles()

    @abstractmethod
    def get_controls(self) -> Dict[str, Dict[str, Any]]:
        """Return current value, supported range and sensor for adjustable controls."""

    @abstractmethod
    def set_control(self, name: str, value: float) -> None:
        raise NotImplementedError

    def close(self) -> None:
        self.stop()

    def __enter__(self) -> "CameraBackend":
        self.start()
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

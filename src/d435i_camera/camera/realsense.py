"""Intel RealSense D435i implementation using aligned native depth."""

from __future__ import annotations

import re
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..config import CameraConfig
from ..models import CameraDeviceInfo, FrameBundle, ImuSample, Intrinsics, StreamProfile
from .base import CameraBackend, CameraError, CameraNotStartedError


def _rs_module() -> Any:
    try:
        import pyrealsense2 as rs
    except ImportError as exc:
        raise CameraError(
            "未安装 pyrealsense2。请运行 scripts/install.sh，或执行 "
            "python3 -m pip install pyrealsense2。"
        ) from exc
    return rs


class RealSenseCamera(CameraBackend):
    # These options are useful for diagnostics but can change timing, calibration units,
    # synchronization or firmware sequencing. Show them in the GUI but do not edit them
    # through the ordinary operator interface.
    PROTECTED_OPTIONS = {
        "frames_queue_size",
        "error_polling_enabled",
        "output_trigger_enabled",
        "depth_units",
        "stereo_baseline",
        "inter_cam_sync_mode",
        "emitter_on_off",
        "global_time_enabled",
        "emitter_always_on",
        "hdr_enabled",
        "sequence_name",
        "sequence_size",
        "sequence_id",
        "asic_temperature",
        "projector_temperature",
    }

    def __init__(self, config: CameraConfig) -> None:
        self.config = config
        self._lock = threading.RLock()
        self._pipeline: Any = None
        self._profile: Any = None
        self._align: Any = None
        self._depth_scale = 0.001
        self._intrinsics: Optional[Intrinsics] = None
        self._device_info: Optional[CameraDeviceInfo] = None
        self._filters: List[Any] = []
        self._last_imu: Optional[ImuSample] = None

    @classmethod
    def list_devices(cls) -> List[CameraDeviceInfo]:
        rs = _rs_module()
        devices: List[CameraDeviceInfo] = []
        for device in rs.context().query_devices():
            def info(key: Any) -> str:
                try:
                    return str(device.get_info(key)) if device.supports(key) else ""
                except RuntimeError:
                    return ""

            name = info(rs.camera_info.name)
            devices.append(
                CameraDeviceInfo(
                    name=name,
                    serial=info(rs.camera_info.serial_number),
                    firmware=info(rs.camera_info.firmware_version),
                    usb_type=info(rs.camera_info.usb_type_descriptor),
                    product_line=info(rs.camera_info.product_line),
                    supports_imu=("435i" in name.lower() or "455" in name.lower()),
                )
            )
        return devices

    @property
    def running(self) -> bool:
        return self._pipeline is not None

    @property
    def intrinsics(self) -> Optional[Intrinsics]:
        return self._intrinsics

    @property
    def device_info(self) -> Optional[CameraDeviceInfo]:
        return self._device_info

    @staticmethod
    def _motion_frequencies(rs: Any, serial: str) -> Tuple[int, int]:
        accel_rates: List[int] = []
        gyro_rates: List[int] = []
        for device in rs.context().query_devices():
            try:
                if device.get_info(rs.camera_info.serial_number) != serial:
                    continue
            except RuntimeError:
                continue
            for sensor in device.query_sensors():
                for profile in sensor.get_stream_profiles():
                    try:
                        if profile.format() != rs.format.motion_xyz32f:
                            continue
                        if profile.stream_type() == rs.stream.accel:
                            accel_rates.append(int(profile.fps()))
                        elif profile.stream_type() == rs.stream.gyro:
                            gyro_rates.append(int(profile.fps()))
                    except RuntimeError:
                        continue
        if not accel_rates or not gyro_rates:
            raise CameraError("D435i 未公布可用的加速度计/陀螺仪 profile")
        return min(accel_rates), min(gyro_rates)

    def start(self) -> CameraDeviceInfo:
        with self._lock:
            if self.running:
                assert self._device_info is not None
                return self._device_info
            rs = _rs_module()
            devices = self.list_devices()
            if not devices:
                raise CameraError("未发现 Intel RealSense 相机")
            if self.config.serial:
                matched = [item for item in devices if item.serial == self.config.serial]
                if not matched:
                    found = ", ".join(item.serial for item in devices)
                    raise CameraError(
                        f"未找到配置的相机序列号 {self.config.serial}；当前设备：{found}"
                    )
                selected = matched[0]
            else:
                selected = devices[0]

            pipeline = rs.pipeline()
            pipeline_config = rs.config()
            pipeline_config.enable_device(selected.serial)
            pipeline_config.enable_stream(
                rs.stream.color,
                self.config.width,
                self.config.height,
                rs.format.bgr8,
                self.config.fps,
            )
            pipeline_config.enable_stream(
                rs.stream.depth,
                self.config.depth_width,
                self.config.depth_height,
                rs.format.z16,
                self.config.depth_fps,
            )
            if self.config.enable_imu and selected.supports_imu:
                # Select from profiles reported by this exact device instead of assuming
                # a frequency (D435i firmware variants commonly differ).
                accel_fps, gyro_fps = self._motion_frequencies(rs, selected.serial)
                pipeline_config.enable_stream(
                    rs.stream.accel, rs.format.motion_xyz32f, accel_fps
                )
                pipeline_config.enable_stream(
                    rs.stream.gyro, rs.format.motion_xyz32f, gyro_fps
                )

            try:
                profile = pipeline.start(pipeline_config)
            except RuntimeError as exc:
                raise CameraError(f"RealSense 启动失败：{exc}") from exc

            device = profile.get_device()
            try:
                self._depth_scale = float(device.first_depth_sensor().get_depth_scale())
            except RuntimeError:
                self._depth_scale = 0.001
            color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
            raw_intr = color_profile.get_intrinsics()
            self._intrinsics = Intrinsics(
                width=int(raw_intr.width),
                height=int(raw_intr.height),
                fx=float(raw_intr.fx),
                fy=float(raw_intr.fy),
                cx=float(raw_intr.ppx),
                cy=float(raw_intr.ppy),
                distortion_model=str(raw_intr.model),
                coeffs=tuple(float(value) for value in raw_intr.coeffs),
            )
            self._pipeline = pipeline
            self._profile = profile
            self._align = rs.align(rs.stream.color) if self.config.align_depth_to_color else None
            self._filters = []
            if self.config.filters.spatial:
                self._filters.append(rs.spatial_filter())
            if self.config.filters.temporal:
                self._filters.append(rs.temporal_filter())
            if self.config.filters.hole_filling:
                self._filters.append(rs.hole_filling_filter())
            self._device_info = selected

            # Auto exposure needs several frames to settle before consumers calibrate/capture.
            for _ in range(8):
                try:
                    pipeline.wait_for_frames(1000)
                except RuntimeError:
                    break
            return selected

    def _list_native_profiles(self, wanted_stream: Any, wanted_format: Any) -> List[StreamProfile]:
        rs = _rs_module()
        selected_serial = self.config.serial
        profiles = set()
        for device in rs.context().query_devices():
            try:
                serial = device.get_info(rs.camera_info.serial_number)
            except RuntimeError:
                continue
            if selected_serial and serial != selected_serial:
                continue
            for sensor in device.query_sensors():
                for profile in sensor.get_stream_profiles():
                    try:
                        stream_type = profile.stream_type()
                        pixel_format = profile.format()
                        if stream_type != wanted_stream or pixel_format != wanted_format:
                            continue
                        video = profile.as_video_stream_profile()
                        key = (int(video.width()), int(video.height()), int(profile.fps()))
                        profiles.add(key)
                    except RuntimeError:
                        continue
            if selected_serial:
                break
        return [
            StreamProfile(width, height, fps)
            for width, height, fps in sorted(
                profiles, key=lambda item: (item[0] * item[1], item[2])
            )
        ]

    def list_color_profiles(self) -> List[StreamProfile]:
        rs = _rs_module()
        return self._list_native_profiles(rs.stream.color, rs.format.bgr8)

    def list_depth_profiles(self) -> List[StreamProfile]:
        rs = _rs_module()
        return self._list_native_profiles(rs.stream.depth, rs.format.z16)

    def list_stream_profiles(self) -> List[StreamProfile]:
        color = {(item.width, item.height, item.fps) for item in self.list_color_profiles()}
        depth = {(item.width, item.height, item.fps) for item in self.list_depth_profiles()}
        return [StreamProfile(*values) for values in sorted(color & depth)]

    def stop(self) -> None:
        with self._lock:
            if self._pipeline is not None:
                try:
                    self._pipeline.stop()
                except RuntimeError:
                    pass
            self._pipeline = None
            self._profile = None
            self._align = None
            self._filters = []

    def _motion_tuple(self, frames: Any, stream: Any) -> Optional[Tuple[float, float, float]]:
        try:
            frame = frames.first_or_default(stream)
            if not frame:
                return None
            data = frame.as_motion_frame().get_motion_data()
            return float(data.x), float(data.y), float(data.z)
        except (RuntimeError, AttributeError):
            return None

    def get_frame(self, timeout_ms: Optional[int] = None) -> FrameBundle:
        with self._lock:
            if not self.running:
                raise CameraNotStartedError("RealSense 相机尚未启动")
            assert self._pipeline is not None and self._intrinsics is not None
            rs = _rs_module()
            try:
                frames = self._pipeline.wait_for_frames(
                    int(timeout_ms if timeout_ms is not None else self.config.timeout_ms)
                )
                accel = self._motion_tuple(frames, rs.stream.accel)
                gyro = self._motion_tuple(frames, rs.stream.gyro)
                if accel is not None or gyro is not None:
                    self._last_imu = ImuSample(time.time(), accel, gyro)
                video_frames = self._align.process(frames) if self._align is not None else frames
                color_frame = video_frames.get_color_frame()
                depth_frame = video_frames.get_depth_frame()
                if not color_frame or not depth_frame:
                    raise CameraError("本次 RealSense 帧缺少彩色图或深度图")
                for filter_object in self._filters:
                    depth_frame = filter_object.process(depth_frame)
                color = np.asanyarray(color_frame.get_data()).copy()
                depth = (
                    np.asanyarray(depth_frame.get_data()).astype(np.float32, copy=False)
                    * self._depth_scale
                )
                bundle = FrameBundle(
                    color_bgr=color,
                    depth_m=depth,
                    intrinsics=self._intrinsics,
                    # Public timestamps use host Unix time so they can be compared with
                    # ROS/state data. The hardware clock is preserved separately below.
                    timestamp_s=time.time(),
                    frame_number=int(color_frame.get_frame_number()),
                    imu=self._last_imu,
                    metadata={
                        "serial": self._device_info.serial if self._device_info else "",
                        "depth_scale": self._depth_scale,
                        "depth_aligned_to_color": self._align is not None,
                        "device_timestamp_ms": float(color_frame.get_timestamp()),
                        "timestamp_domain": str(color_frame.get_frame_timestamp_domain()),
                    },
                )
                bundle.validate()
                return bundle
            except RuntimeError as exc:
                raise CameraError(f"读取 RealSense 帧失败：{exc}") from exc

    @staticmethod
    def _option_key(sensor_name: str, option: Any) -> Tuple[str, str]:
        sensor_lower = sensor_name.lower()
        if "rgb" in sensor_lower or "color" in sensor_lower:
            prefix = "color"
        elif "stereo" in sensor_lower or "depth" in sensor_lower:
            prefix = "depth"
        elif "motion" in sensor_lower:
            prefix = "motion"
        else:
            prefix = re.sub(r"[^a-z0-9]+", "_", sensor_lower).strip("_") or "sensor"
        option_name = re.sub(r"[^a-z0-9]+", "_", str(option).lower()).strip("_")
        return f"{prefix}.{option_name}", option_name

    def _option_map(self) -> Dict[str, Tuple[Any, Any, str, bool]]:
        if not self.running or self._profile is None:
            raise CameraNotStartedError("RealSense 相机尚未启动")
        rs = _rs_module()
        result: Dict[str, Tuple[Any, Any, str, bool]] = {}
        for sensor in self._profile.get_device().query_sensors():
            sensor_name = "unknown"
            try:
                sensor_name = sensor.get_info(rs.camera_info.name)
            except RuntimeError:
                pass
            for option in sensor.get_supported_options():
                try:
                    if sensor.supports(option):
                        key, option_name = self._option_key(sensor_name, option)
                        result[key] = (
                            sensor,
                            option,
                            sensor_name,
                            option_name in self.PROTECTED_OPTIONS,
                        )
                except RuntimeError:
                    continue
        return result

    def get_controls(self) -> Dict[str, Dict[str, Any]]:
        controls: Dict[str, Dict[str, Any]] = {}
        with self._lock:
            for name, (sensor, option, sensor_name, protected) in self._option_map().items():
                try:
                    value_range = sensor.get_option_range(option)
                    hardware_read_only = bool(sensor.is_option_read_only(option))
                    controls[name] = {
                        "value": float(sensor.get_option(option)),
                        "min": float(value_range.min),
                        "max": float(value_range.max),
                        "step": float(value_range.step),
                        "default": float(value_range.default),
                        "read_only": hardware_read_only or protected,
                        "hardware_read_only": hardware_read_only,
                        "protected": protected,
                        "sensor": str(sensor_name),
                    }
                except RuntimeError:
                    continue
        return controls

    def set_control(self, name: str, value: float) -> None:
        with self._lock:
            options = self._option_map()
            if name not in options:
                raise CameraError(f"当前 D435i 不支持参数：{name}")
            sensor, option, _sensor_name, protected = options[name]
            if protected:
                raise CameraError(f"参数 {name} 属于高级保护项，普通接口禁止修改")
            try:
                value_range = sensor.get_option_range(option)
                clipped = min(float(value_range.max), max(float(value_range.min), float(value)))
                sensor.set_option(option, clipped)
            except RuntimeError as exc:
                raise CameraError(f"设置 {name} 失败：{exc}") from exc

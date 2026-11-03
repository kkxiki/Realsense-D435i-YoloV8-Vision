"""Headless Python 3.10 camera gateway for the ROS2 Foxy Python 3.8 bridge."""

from __future__ import annotations

import argparse
import json
import signal
import socketserver
import threading
import traceback
import zlib
from typing import Any, Dict, Optional

import cv2
import numpy as np

from .bridge_protocol import BridgePacket, encode_packet
from .config import load_config
from .geometry import make_transform
from .labels import chinese_class_name
from .service import CameraModule


class LatestPacket:
    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._sequence = -1
        self._payload: Optional[bytes] = None

    def publish(self, sequence: int, payload: bytes) -> None:
        with self._condition:
            self._sequence = int(sequence)
            self._payload = payload
            self._condition.notify_all()

    def wait_after(self, sequence: int, timeout_s: float = 2.0) -> tuple[int, Optional[bytes]]:
        with self._condition:
            self._condition.wait_for(
                lambda: self._sequence > sequence and self._payload is not None,
                timeout=float(timeout_s),
            )
            return self._sequence, self._payload if self._sequence > sequence else None


class CameraGateway:
    def __init__(self, config_path: str, jpeg_quality: int = 85, mock: bool = False) -> None:
        self.config = load_config(config_path)
        if mock:
            self.config.camera.backend = "mock"
        self.module = CameraModule(self.config)
        self.jpeg_quality = int(np.clip(jpeg_quality, 40, 95))
        self.latest_packet = LatestPacket()
        self.operation_lock = threading.RLock()
        self.stop_event = threading.Event()
        self.worker = threading.Thread(target=self._process_loop, name="camera-process", daemon=True)
        self.last_error = ""
        self.calibration_sessions: Dict[str, Any] = {}
        self._bridge_sequence = 0

    def start(self, start_camera: bool = True) -> None:
        if start_camera:
            with self.operation_lock:
                self.module.start()
        self.worker.start()

    def close(self) -> None:
        self.stop_event.set()
        self.worker.join(timeout=3.0)
        with self.operation_lock:
            self.module.close()

    def _encode_result(self, result: Any, bridge_sequence: int) -> bytes:
        color_ok, color = cv2.imencode(
            ".jpg", result.frame.color_bgr, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality]
        )
        annotated_ok, annotated = cv2.imencode(
            ".jpg", result.annotated_bgr, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality]
        )
        if not color_ok or not annotated_ok:
            raise RuntimeError("JPEG encoding failed")
        depth = result.frame.depth_m
        if depth is None:
            depth_bytes = b""
            depth_shape = []
        else:
            depth_array = np.ascontiguousarray(depth, dtype=np.float32)
            depth_bytes = zlib.compress(depth_array.tobytes(), level=1)
            depth_shape = list(depth_array.shape)
        objects = []
        for observation in result.observations:
            item = observation.as_dict()
            item["class_name_zh"] = chinese_class_name(observation.detection.class_name)
            objects.append(item)
        metadata = {
            "schema": "xuk.d435i.bridge/v1",
            "timestamp_s": result.frame.timestamp_s,
            "frame_number": result.frame.frame_number,
            "sequence": int(bridge_sequence),
            "perception_sequence": result.sequence,
            "camera_frame": result.frame.camera_frame,
            "intrinsics": result.frame.intrinsics.as_dict(),
            "depth_shape": depth_shape,
            "depth_dtype": "float32",
            "depth_encoding": "32FC1",
            "objects": objects,
            "processing_ms": result.processing_ms,
            "inference_ms": result.inference_ms,
        }
        return encode_packet(
            BridgePacket(metadata, color.tobytes(), depth_bytes, annotated.tobytes())
        )

    def _process_loop(self) -> None:
        while not self.stop_event.is_set():
            if not self.module.running:
                self.stop_event.wait(0.05)
                continue
            try:
                with self.operation_lock:
                    result = self.module.process_once()
                bridge_sequence = self._bridge_sequence
                self._bridge_sequence += 1
                self.latest_packet.publish(
                    bridge_sequence, self._encode_result(result, bridge_sequence)
                )
                self.last_error = ""
            except Exception as exc:
                self.last_error = f"{exc}"
                self.stop_event.wait(0.1)

    @staticmethod
    def _pose_matrix(payload: Dict[str, Any]) -> np.ndarray:
        position = payload.get("position") or {}
        orientation = payload.get("orientation") or {}
        x, y, z, w = [
            float(orientation.get(key, default))
            for key, default in (("x", 0.0), ("y", 0.0), ("z", 0.0), ("w", 1.0))
        ]
        norm = np.linalg.norm([x, y, z, w])
        if norm < 1e-12:
            raise ValueError("base_T_tool quaternion has zero norm")
        x, y, z, w = np.asarray([x, y, z, w], dtype=np.float64) / norm
        rotation = np.array(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
            ]
        )
        return make_transform(
            rotation,
            [float(position.get("x", 0.0)), float(position.get("y", 0.0)), float(position.get("z", 0.0))],
        )

    def _calibration(self, request: Dict[str, Any]) -> Dict[str, Any]:
        calibration_type = str(request.get("calibration_type", "")).strip().lower()
        command = str(request.get("calibration_command", "")).strip().lower()
        if calibration_type not in {"intrinsic", "eye_in_hand", "eye_to_hand"}:
            raise ValueError("calibration_type must be intrinsic, eye_in_hand or eye_to_hand")
        if command == "reset":
            creators = {
                "intrinsic": self.module.new_intrinsic_session,
                "eye_in_hand": self.module.new_handeye_session,
                "eye_to_hand": self.module.new_eye_to_hand_session,
            }
            self.calibration_sessions[calibration_type] = creators[calibration_type]()
            return {"message": f"{calibration_type} session reset", "sample_count": 0}
        session = self.calibration_sessions.get(calibration_type)
        if session is None:
            raise ValueError("calibration session does not exist; call reset first")
        if command == "capture":
            frame = self.module.latest_raw
            if frame is None:
                raise ValueError("no camera frame available")
            if calibration_type == "intrinsic":
                accepted, message, _preview = session.add_frame(frame.color_bgr)
            else:
                base_T_tool = self._pose_matrix(request.get("base_to_tool") or {})
                accepted, message, _preview = session.add_frame(
                    frame.color_bgr, base_T_tool, frame.timestamp_s
                )
            return {
                "message": message,
                "accepted": bool(accepted),
                "sample_count": int(session.sample_count),
            }
        if command == "solve":
            result = session.solve()
            config = self.module.config.calibration
            path_by_type = {
                "intrinsic": config.intrinsic_file,
                "eye_in_hand": config.handeye_file,
                "eye_to_hand": config.eye_to_hand_file,
            }
            path = result.save(self.module.config.resolve(path_by_type[calibration_type]))
            if calibration_type == "eye_in_hand":
                self.module.set_tool_T_camera(result.tool_T_camera)
            elif calibration_type == "eye_to_hand":
                self.module.set_base_T_camera(result.base_T_camera)
            return {"message": "calibration solved", "result_path": str(path), "result": result.as_dict()}
        raise ValueError("calibration_command must be reset, capture or solve")

    def handle_command(self, request: Dict[str, Any]) -> Dict[str, Any]:
        command = str(request.get("command", "")).strip().lower()
        try:
            with self.operation_lock:
                if command == "status":
                    result = self.module.health()
                    result["gateway_error"] = self.last_error
                elif command == "get_profiles":
                    result = {
                        "color": [item.as_dict() for item in self.module.list_color_profiles()],
                        "depth": [item.as_dict() for item in self.module.list_depth_profiles()],
                    }
                elif command == "get_controls":
                    result = self.module.get_controls()
                elif command == "set_control":
                    self.module.set_control(str(request["name"]), float(request["value"]))
                    result = self.module.get_controls().get(str(request["name"]), {})
                elif command == "set_stream":
                    result = self.module.configure_camera(
                        color_width=int(request["color_width"]),
                        color_height=int(request["color_height"]),
                        color_fps=int(request["color_fps"]),
                        depth_width=int(request["depth_width"]),
                        depth_height=int(request["depth_height"]),
                        depth_fps=int(request["depth_fps"]),
                        enable_imu=bool(request.get("enable_imu", False)),
                        restart_if_running=True,
                        validate_streams=True,
                        persist=bool(request.get("persist", True)),
                    )
                elif command == "start":
                    result = self.module.health() if self.module.running else vars(self.module.start())
                elif command == "stop":
                    self.module.stop()
                    result = {"running": False}
                elif command == "snapshot":
                    paths = self.module.capture_snapshot()
                    result = {
                        "color": str(paths.color),
                        "depth": str(paths.depth) if paths.depth else "",
                        "metadata": str(paths.metadata),
                    }
                elif command == "calibration":
                    result = self._calibration(request)
                else:
                    raise ValueError(f"unknown gateway command: {command}")
            return {"success": True, "result": result}
        except Exception as exc:
            return {
                "success": False,
                "message": str(exc),
                "traceback": traceback.format_exc(limit=8),
            }


class ControlHandler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        line = self.rfile.readline(4 * 1024 * 1024)
        if not line:
            return
        try:
            request = json.loads(line.decode("utf-8"))
            response = self.server.gateway.handle_command(request)  # type: ignore[attr-defined]
        except Exception as exc:
            response = {"success": False, "message": str(exc)}
        self.wfile.write(json.dumps(response, ensure_ascii=False).encode("utf-8") + b"\n")


class StreamHandler(socketserver.BaseRequestHandler):
    def handle(self) -> None:
        sequence = -1
        gateway = self.server.gateway  # type: ignore[attr-defined]
        while not gateway.stop_event.is_set():
            sequence, payload = gateway.latest_packet.wait_after(sequence)
            if payload is None:
                continue
            try:
                self.request.sendall(payload)
            except (BrokenPipeError, ConnectionResetError, OSError):
                return


class ThreadedServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="D435i ROS2 Foxy TCP gateway")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--control-port", type=int, default=17630)
    parser.add_argument("--stream-port", type=int, default=17631)
    parser.add_argument("--jpeg-quality", type=int, default=85)
    parser.add_argument("--no-auto-start", action="store_true")
    parser.add_argument("--mock", action="store_true", help="use synthetic RGB-D and detections")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    gateway = CameraGateway(args.config, args.jpeg_quality, mock=args.mock)
    control = ThreadedServer((args.host, args.control_port), ControlHandler)
    stream = ThreadedServer((args.host, args.stream_port), StreamHandler)
    control.gateway = gateway  # type: ignore[attr-defined]
    stream.gateway = gateway  # type: ignore[attr-defined]
    threads = [
        threading.Thread(target=control.serve_forever, name="control-server", daemon=True),
        threading.Thread(target=stream.serve_forever, name="stream-server", daemon=True),
    ]
    gateway.start(start_camera=not args.no_auto_start)
    for thread in threads:
        thread.start()
    stop = threading.Event()

    def request_stop(_signum: int, _frame: Any) -> None:
        stop.set()

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    print(
        f"D435i gateway ready: control={args.host}:{args.control_port}, "
        f"stream={args.host}:{args.stream_port}",
        flush=True,
    )
    try:
        while not stop.wait(0.5):
            pass
    finally:
        control.shutdown()
        stream.shutdown()
        control.server_close()
        stream.server_close()
        gateway.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

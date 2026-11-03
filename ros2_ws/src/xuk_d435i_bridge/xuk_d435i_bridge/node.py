"""ROS2 Foxy publishers and services for the XuK D435i camera gateway."""

import json
import socket
import threading
import time
import zlib

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image
from std_srvs.srv import Trigger
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose

from xuk_d435i_interfaces.msg import Object6D, Object6DArray
from xuk_d435i_interfaces.srv import (
    CalibrationCommand,
    GetJson,
    SetControl,
    SetStream,
)

from .protocol import recv_packet, request_json


class D435iBridge(Node):
    def __init__(self):
        super().__init__("xuk_d435i_bridge")
        defaults = {
            "gateway_host": "127.0.0.1",
            "control_port": 17630,
            "stream_port": 17631,
            "publish_rate_hz": 30.0,
            "reconnect_delay_s": 1.0,
            "publish_legacy_yolo": True,
            "color_topic": "/d435i/color/image_raw",
            "depth_topic": "/d435i/aligned_depth_to_color/image_raw",
            "camera_info_topic": "/d435i/color/camera_info",
            "annotated_topic": "/d435i/annotated/image_raw",
            "objects_topic": "/d435i/objects_6d",
            "legacy_detection_topic": "/yolov8/detections",
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)
        value = lambda name: self.get_parameter(name).value
        self.gateway_host = str(value("gateway_host"))
        self.control_port = int(value("control_port"))
        self.stream_port = int(value("stream_port"))
        self.reconnect_delay_s = float(value("reconnect_delay_s"))
        self.publish_legacy_yolo = bool(value("publish_legacy_yolo"))

        self.color_pub = self.create_publisher(Image, str(value("color_topic")), qos_profile_sensor_data)
        self.depth_pub = self.create_publisher(Image, str(value("depth_topic")), qos_profile_sensor_data)
        self.info_pub = self.create_publisher(
            CameraInfo, str(value("camera_info_topic")), qos_profile_sensor_data
        )
        self.annotated_pub = self.create_publisher(
            Image, str(value("annotated_topic")), qos_profile_sensor_data
        )
        self.objects_pub = self.create_publisher(
            Object6DArray, str(value("objects_topic")), qos_profile_sensor_data
        )
        self.legacy_pub = self.create_publisher(
            Detection2DArray, str(value("legacy_detection_topic")), qos_profile_sensor_data
        )

        self.create_service(Trigger, "/d435i/start", self._start)
        self.create_service(Trigger, "/d435i/stop", self._stop)
        self.create_service(Trigger, "/d435i/snapshot", self._snapshot)
        self.create_service(GetJson, "/d435i/get_json", self._get_json)
        self.create_service(SetControl, "/d435i/set_control", self._set_control)
        self.create_service(SetStream, "/d435i/set_stream", self._set_stream)
        self.create_service(CalibrationCommand, "/d435i/calibration", self._calibration)

        self._packet_lock = threading.Lock()
        self._latest_packet = None
        self._latest_sequence = -1
        self._published_sequence = -1
        self._stop_event = threading.Event()
        self._last_stream_error = ""
        self._stream_thread = threading.Thread(
            target=self._stream_loop, name="d435i-gateway-stream", daemon=True
        )
        self._stream_thread.start()
        publish_rate = max(1.0, float(value("publish_rate_hz")))
        self.create_timer(1.0 / publish_rate, self._publish_latest)
        self.get_logger().info(
            "D435i Foxy bridge ready; gateway=%s:%d/%d"
            % (self.gateway_host, self.control_port, self.stream_port)
        )

    def _stream_loop(self):
        while not self._stop_event.is_set():
            try:
                with socket.create_connection(
                    (self.gateway_host, self.stream_port), timeout=3.0
                ) as stream:
                    stream.settimeout(3.0)
                    self._last_stream_error = ""
                    while not self._stop_event.is_set():
                        packet = recv_packet(stream)
                        sequence = int(packet["metadata"].get("sequence", -1))
                        with self._packet_lock:
                            self._latest_packet = packet
                            self._latest_sequence = sequence
            except Exception as exc:
                message = str(exc)
                if message != self._last_stream_error:
                    self.get_logger().warn("等待相机网关连接: " + message)
                    self._last_stream_error = message
                self._stop_event.wait(self.reconnect_delay_s)

    @staticmethod
    def _image_message(image, encoding, header):
        message = Image()
        message.header = header
        message.height = int(image.shape[0])
        message.width = int(image.shape[1])
        message.encoding = encoding
        message.is_bigendian = False
        message.step = int(image.strides[0])
        message.data = np.ascontiguousarray(image).tobytes()
        return message

    @staticmethod
    def _fill_pose(target, pose):
        if not pose:
            target.orientation.w = 1.0
            return
        xyz = pose.get("translation_m") or [0.0, 0.0, 0.0]
        quat = pose.get("quaternion_xyzw") or [0.0, 0.0, 0.0, 1.0]
        target.position.x, target.position.y, target.position.z = [float(v) for v in xyz]
        target.orientation.x, target.orientation.y, target.orientation.z, target.orientation.w = [
            float(v) for v in quat
        ]

    def _publish_latest(self):
        with self._packet_lock:
            if self._latest_packet is None or self._latest_sequence == self._published_sequence:
                return
            packet = self._latest_packet
            sequence = self._latest_sequence
        try:
            metadata = packet["metadata"]
            color = cv2.imdecode(
                np.frombuffer(packet["color_jpeg"], dtype=np.uint8), cv2.IMREAD_COLOR
            )
            annotated = cv2.imdecode(
                np.frombuffer(packet["annotated_jpeg"], dtype=np.uint8), cv2.IMREAD_COLOR
            )
            if color is None or annotated is None:
                raise ValueError("cannot decode gateway JPEG")
            shape = metadata.get("depth_shape") or []
            depth = None
            if packet["depth_zlib"] and len(shape) == 2:
                raw = zlib.decompress(packet["depth_zlib"])
                depth = np.frombuffer(raw, dtype=np.float32).reshape(
                    (int(shape[0]), int(shape[1]))
                )
            stamp = self.get_clock().now().to_msg()
            frame_id = str(metadata.get("camera_frame") or "camera_color_optical_frame")
            header = self._header(stamp, frame_id)
            self.color_pub.publish(self._image_message(color, "bgr8", header))
            self.annotated_pub.publish(self._image_message(annotated, "bgr8", header))
            if depth is not None:
                self.depth_pub.publish(self._image_message(depth, "32FC1", header))
            self.info_pub.publish(self._camera_info(metadata.get("intrinsics") or {}, header))
            self.objects_pub.publish(self._objects_message(metadata, header))
            if self.publish_legacy_yolo:
                self.legacy_pub.publish(
                    self._legacy_message(metadata, header, color.shape[1], color.shape[0])
                )
            self._published_sequence = sequence
        except Exception as exc:
            self.get_logger().error("发布 D435i 数据失败: " + str(exc))

    @staticmethod
    def _header(stamp, frame_id):
        from std_msgs.msg import Header

        header = Header()
        header.stamp = stamp
        header.frame_id = frame_id
        return header

    @staticmethod
    def _camera_info(intrinsics, header):
        info = CameraInfo()
        info.header = header
        info.width = int(intrinsics.get("width", 0))
        info.height = int(intrinsics.get("height", 0))
        fx = float(intrinsics.get("fx", 0.0))
        fy = float(intrinsics.get("fy", 0.0))
        cx = float(intrinsics.get("cx", 0.0))
        cy = float(intrinsics.get("cy", 0.0))
        model = str(intrinsics.get("distortion_model", "none"))
        info.distortion_model = "plumb_bob" if model != "none" else ""
        info.d = [float(v) for v in intrinsics.get("coeffs", [])]
        info.k = [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]
        info.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        info.p = [fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0]
        return info

    def _objects_message(self, metadata, header):
        message = Object6DArray()
        message.header = header
        for item in metadata.get("objects", []):
            obj = Object6D()
            obj.class_name = str(item.get("class_name", ""))
            obj.class_name_zh = str(item.get("class_name_zh", obj.class_name))
            obj.class_id = int(item.get("class_id", -1))
            obj.confidence = float(item.get("confidence", 0.0))
            obj.bbox_xyxy_px = [float(v) for v in item.get("bbox_xyxy", [0, 0, 0, 0])]
            pose = item.get("pose")
            self._fill_pose(obj.pose, pose)
            dimensions = (pose or {}).get("dimensions_m") or [0.0, 0.0, 0.0]
            obj.dimensions_m.x, obj.dimensions_m.y, obj.dimensions_m.z = [
                float(v) for v in dimensions
            ]
            rpy = (pose or {}).get("rpy_deg") or [0.0, 0.0, 0.0]
            obj.rx_deg, obj.ry_deg, obj.rz_deg = [float(v) for v in rpy]
            obj.frame_id = str((pose or {}).get("frame_id", header.frame_id))
            obj.valid = bool(item.get("valid", False))
            obj.message = str(item.get("message", ""))
            depth = item.get("depth_m")
            obj.depth_m = float(depth) if depth is not None else float("nan")
            message.objects.append(obj)
        return message

    def _legacy_message(self, metadata, header, width, height):
        array = Detection2DArray()
        array.header = header
        width = max(1.0, float(width))
        height = max(1.0, float(height))
        for item in metadata.get("objects", []):
            x1, y1, x2, y2 = [float(v) for v in item.get("bbox_xyxy", [0, 0, 0, 0])]
            detection = Detection2D()
            detection.header = header
            detection.bbox.center.x = ((x1 + x2) * 0.5) / width
            detection.bbox.center.y = ((y1 + y2) * 0.5) / height
            detection.bbox.center.theta = 0.0
            detection.bbox.size_x = max(0.0, x2 - x1) / width
            detection.bbox.size_y = max(0.0, y2 - y1) / height
            hypothesis = ObjectHypothesisWithPose()
            hypothesis.id = str(item.get("class_id", -1))
            hypothesis.score = float(item.get("confidence", 0.0))
            self._fill_pose(hypothesis.pose.pose, item.get("pose"))
            detection.results.append(hypothesis)
            array.detections.append(detection)
        return array

    def _gateway(self, request):
        return request_json(self.gateway_host, self.control_port, request)

    @staticmethod
    def _result_text(reply):
        if reply.get("success"):
            return True, "ok", json.dumps(reply.get("result", {}), ensure_ascii=False)
        return False, str(reply.get("message", "gateway command failed")), ""

    def _trigger(self, command, response):
        try:
            success, message, result = self._result_text(self._gateway({"command": command}))
            response.success = success
            response.message = result if success else message
        except Exception as exc:
            response.success = False
            response.message = str(exc)
        return response

    def _start(self, _request, response):
        return self._trigger("start", response)

    def _stop(self, _request, response):
        return self._trigger("stop", response)

    def _snapshot(self, _request, response):
        return self._trigger("snapshot", response)

    def _get_json(self, request, response):
        command = {"profiles": "get_profiles", "controls": "get_controls"}.get(
            request.operation.strip().lower(), "status"
        )
        try:
            response.success, response.message, response.json = self._result_text(
                self._gateway({"command": command})
            )
        except Exception as exc:
            response.success, response.message, response.json = False, str(exc), ""
        return response

    def _set_control(self, request, response):
        try:
            response.success, response.message, response.control_json = self._result_text(
                self._gateway(
                    {"command": "set_control", "name": request.name, "value": request.value}
                )
            )
        except Exception as exc:
            response.success, response.message, response.control_json = False, str(exc), ""
        return response

    def _set_stream(self, request, response):
        payload = {
            "command": "set_stream",
            "color_width": request.color_width,
            "color_height": request.color_height,
            "color_fps": request.color_fps,
            "depth_width": request.depth_width,
            "depth_height": request.depth_height,
            "depth_fps": request.depth_fps,
            "enable_imu": request.enable_imu,
            "persist": request.persist,
        }
        try:
            response.success, response.message, response.config_json = self._result_text(
                self._gateway(payload)
            )
        except Exception as exc:
            response.success, response.message, response.config_json = False, str(exc), ""
        return response

    def _calibration(self, request, response):
        pose = request.base_to_tool
        payload = {
            "command": "calibration",
            "calibration_type": request.calibration_type,
            "calibration_command": request.command,
            "base_to_tool": {
                "position": {"x": pose.position.x, "y": pose.position.y, "z": pose.position.z},
                "orientation": {
                    "x": pose.orientation.x,
                    "y": pose.orientation.y,
                    "z": pose.orientation.z,
                    "w": pose.orientation.w,
                },
            },
        }
        try:
            reply = self._gateway(payload)
            result = reply.get("result", {})
            response.success = bool(reply.get("success", False))
            response.message = str(result.get("message", reply.get("message", "")))
            response.sample_count = int(result.get("sample_count", 0))
            response.result_path = str(result.get("result_path", ""))
            response.result_json = json.dumps(result.get("result", result), ensure_ascii=False)
        except Exception as exc:
            response.success = False
            response.message = str(exc)
        return response

    def destroy_node(self):
        self._stop_event.set()
        self._stream_thread.join(timeout=2.0)
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = D435iBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

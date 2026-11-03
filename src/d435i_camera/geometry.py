"""Coordinate transforms, depth sampling and rotation conversions."""

from __future__ import annotations

import math
from typing import Iterable, Optional, Sequence, Tuple

import cv2
import numpy as np

from .models import Intrinsics


def robust_depth(
    depth_m: np.ndarray,
    u: float,
    v: float,
    radius: int = 4,
    min_m: float = 0.15,
    max_m: float = 2.0,
) -> Optional[float]:
    if depth_m.ndim != 2:
        raise ValueError("depth image must be two-dimensional")
    if radius < 0:
        raise ValueError("depth radius cannot be negative")
    height, width = depth_m.shape
    ui, vi = int(round(u)), int(round(v))
    if not (0 <= ui < width and 0 <= vi < height):
        return None
    values = depth_m[
        max(0, vi - radius) : min(height, vi + radius + 1),
        max(0, ui - radius) : min(width, ui + radius + 1),
    ].astype(np.float64, copy=False)
    valid = values[np.isfinite(values) & (values >= min_m) & (values <= max_m)]
    return float(np.median(valid)) if valid.size else None


def deproject_pixel(u: float, v: float, depth_m: float, intrinsics: Intrinsics) -> np.ndarray:
    intrinsics.validate()
    if not np.isfinite(depth_m) or depth_m <= 0:
        raise ValueError("depth must be positive and finite")
    return np.array(
        [
            (float(u) - intrinsics.cx) * depth_m / intrinsics.fx,
            (float(v) - intrinsics.cy) * depth_m / intrinsics.fy,
            depth_m,
        ],
        dtype=np.float64,
    )


def project_point(point_xyz: Sequence[float], intrinsics: Intrinsics) -> Optional[Tuple[int, int]]:
    point = np.asarray(point_xyz, dtype=np.float64).reshape(3)
    if not np.isfinite(point).all() or point[2] <= 1e-8:
        return None
    u = intrinsics.fx * point[0] / point[2] + intrinsics.cx
    v = intrinsics.fy * point[1] / point[2] + intrinsics.cy
    return int(round(u)), int(round(v))


def make_transform(rotation: np.ndarray, translation: Iterable[float]) -> np.ndarray:
    rotation = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    translation = np.asarray(tuple(translation), dtype=np.float64).reshape(3)
    if not np.isfinite(rotation).all() or not np.isfinite(translation).all():
        raise ValueError("transform values must be finite")
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = translation
    return matrix


def validate_transform(matrix: np.ndarray, atol: float = 1e-5) -> np.ndarray:
    value = np.asarray(matrix, dtype=np.float64)
    if value.shape != (4, 4) or not np.isfinite(value).all():
        raise ValueError("transform must be a finite 4x4 matrix")
    if not np.allclose(value[3], [0, 0, 0, 1], atol=atol):
        raise ValueError("invalid homogeneous transform last row")
    rotation = value[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=atol):
        raise ValueError("rotation is not orthonormal")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=atol):
        raise ValueError("rotation determinant must be +1")
    return value


def invert_transform(matrix: np.ndarray) -> np.ndarray:
    value = validate_transform(matrix)
    rotation = value[:3, :3]
    translation = value[:3, 3]
    return make_transform(rotation.T, -(rotation.T @ translation))


def transform_point(matrix: np.ndarray, point_xyz: Sequence[float]) -> np.ndarray:
    value = validate_transform(matrix)
    point = np.asarray(point_xyz, dtype=np.float64).reshape(3)
    return value[:3, :3] @ point + value[:3, 3]


def rpy_deg_to_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    rx, ry, rz = np.deg2rad([roll, pitch, yaw])
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    return np.array(
        [
            [cz * cy, cz * sy * sx - sz * cx, cz * sy * cx + sz * sx],
            [sz * cy, sz * sy * sx + cz * cx, sz * sy * cx - cz * sx],
            [-sy, cy * sx, cy * cx],
        ],
        dtype=np.float64,
    )


def matrix_to_rpy_deg(rotation: np.ndarray) -> Tuple[float, float, float]:
    r = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    sy = math.hypot(r[0, 0], r[1, 0])
    singular = sy < 1e-8
    if not singular:
        roll = math.atan2(r[2, 1], r[2, 2])
        pitch = math.atan2(-r[2, 0], sy)
        yaw = math.atan2(r[1, 0], r[0, 0])
    else:
        roll = math.atan2(-r[1, 2], r[1, 1])
        pitch = math.atan2(-r[2, 0], sy)
        yaw = 0.0
    return tuple(float(value) for value in np.rad2deg([roll, pitch, yaw]))


def matrix_to_quaternion_xyzw(rotation: np.ndarray) -> Tuple[float, float, float, float]:
    r = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    # OpenCV Rodrigues is stable near pi; convert through the equivalent axis-angle.
    vector, _ = cv2.Rodrigues(r)
    angle = float(np.linalg.norm(vector))
    if angle < 1e-12:
        return 0.0, 0.0, 0.0, 1.0
    axis = vector.reshape(3) / angle
    scale = math.sin(angle * 0.5)
    quat = np.r_[axis * scale, math.cos(angle * 0.5)]
    if quat[3] < 0:
        quat = -quat
    return tuple(float(value) for value in quat)


def rotation_angle_deg(rotation: np.ndarray) -> float:
    vector, _ = cv2.Rodrigues(np.asarray(rotation, dtype=np.float64).reshape(3, 3))
    return float(np.rad2deg(np.linalg.norm(vector)))


def nearest_rotation(matrix: np.ndarray) -> np.ndarray:
    u, _s, vt = np.linalg.svd(np.asarray(matrix, dtype=np.float64).reshape(3, 3))
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vt
    return rotation


def pose_matrix_from_rpy(
    xyz: Sequence[float], rpy_deg: Sequence[float], translation_scale: float = 1.0
) -> np.ndarray:
    return make_transform(
        rpy_deg_to_matrix(*[float(v) for v in rpy_deg]),
        np.asarray(xyz, dtype=np.float64) * float(translation_scale),
    )

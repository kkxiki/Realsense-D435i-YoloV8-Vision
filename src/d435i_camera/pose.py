"""Metric 3D and depth-PCA 6D pose estimation for detected objects."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .config import DepthConfig, PoseConfig
from .geometry import (
    deproject_pixel,
    make_transform,
    matrix_to_quaternion_xyzw,
    matrix_to_rpy_deg,
    nearest_rotation,
    robust_depth,
    validate_transform,
)
from .models import Detection2D, FrameBundle, ObjectObservation, Pose6D


@dataclass
class _TrackState:
    class_id: int
    center: Tuple[float, float]
    pose: Pose6D
    last_sequence: int


class PoseSmoother:
    def __init__(self, alpha: float, max_center_distance_px: float) -> None:
        self.alpha = float(alpha)
        self.max_center_distance_px = float(max_center_distance_px)
        self._tracks: Dict[int, _TrackState] = {}
        self._next_id = 1

    def reset(self) -> None:
        self._tracks.clear()
        self._next_id = 1

    def _match(self, detection: Detection2D, sequence: int) -> Optional[int]:
        if detection.track_id is not None:
            return int(detection.track_id)
        best: Optional[Tuple[float, int]] = None
        center = np.asarray(detection.center)
        for track_id, state in self._tracks.items():
            if sequence - state.last_sequence > 12:
                continue
            if state.class_id != detection.class_id:
                continue
            distance = float(np.linalg.norm(center - np.asarray(state.center)))
            if distance <= self.max_center_distance_px and (best is None or distance < best[0]):
                best = (distance, track_id)
        return best[1] if best else None

    def update(self, detection: Detection2D, pose: Pose6D, sequence: int) -> Pose6D:
        track_id = self._match(detection, sequence)
        if track_id is None:
            track_id = self._next_id
            self._next_id += 1
            smoothed = pose
        else:
            previous = self._tracks.get(track_id)
            if previous is None or previous.pose.frame_id != pose.frame_id:
                smoothed = pose
            else:
                alpha = self.alpha
                translation = (
                    alpha * np.asarray(pose.translation_m)
                    + (1.0 - alpha) * np.asarray(previous.pose.translation_m)
                )
                rotation = nearest_rotation(
                    alpha * pose.rotation_matrix
                    + (1.0 - alpha) * previous.pose.rotation_matrix
                )
                dimensions = (
                    alpha * np.asarray(pose.dimensions_m)
                    + (1.0 - alpha) * np.asarray(previous.pose.dimensions_m)
                )
                smoothed = _make_pose(
                    translation,
                    rotation,
                    pose.frame_id,
                    pose.method + "+ema",
                    min(pose.quality, previous.pose.quality) * 0.25
                    + max(pose.quality, previous.pose.quality) * 0.75,
                    dimensions,
                )
        self._tracks[track_id] = _TrackState(
            detection.class_id, detection.center, smoothed, sequence
        )
        stale = [key for key, state in self._tracks.items() if sequence - state.last_sequence > 30]
        for key in stale:
            self._tracks.pop(key, None)
        return smoothed


def _make_pose(
    translation: Sequence[float],
    rotation: np.ndarray,
    frame_id: str,
    method: str,
    quality: float,
    dimensions_m: Sequence[float] = (0.0, 0.0, 0.0),
) -> Pose6D:
    rotation = nearest_rotation(rotation)
    return Pose6D(
        translation_m=tuple(float(value) for value in np.asarray(translation).reshape(3)),
        rotation_matrix=rotation,
        quaternion_xyzw=matrix_to_quaternion_xyzw(rotation),
        rpy_deg=matrix_to_rpy_deg(rotation),
        frame_id=frame_id,
        method=method,
        quality=float(np.clip(quality, 0.0, 1.0)),
        dimensions_m=tuple(
            float(max(0.0, value)) for value in np.asarray(dimensions_m).reshape(3)
        ),
    )


def bbox_point_cloud(
    detection: Detection2D,
    frame: FrameBundle,
    depth_config: DepthConfig,
) -> Tuple[np.ndarray, Optional[float], int]:
    if frame.depth_m is None:
        return np.empty((0, 3), dtype=np.float64), None, 0
    height, width = frame.depth_m.shape
    clipped = detection.clipped(width, height)
    x1, y1, x2, y2 = [int(round(value)) for value in clipped.bbox_xyxy]
    if x2 <= x1 or y2 <= y1:
        return np.empty((0, 3), dtype=np.float64), None, 0
    u, v = clipped.center
    center_depth = robust_depth(
        frame.depth_m,
        u,
        v,
        radius=depth_config.roi_radius_px,
        min_m=depth_config.min_m,
        max_m=depth_config.max_m,
    )
    roi = frame.depth_m[y1 : y2 + 1, x1 : x2 + 1]
    if center_depth is None:
        valid_roi = roi[
            np.isfinite(roi) & (roi >= depth_config.min_m) & (roi <= depth_config.max_m)
        ]
        center_depth = float(np.median(valid_roi)) if valid_roi.size else None
    if center_depth is None:
        return np.empty((0, 3), dtype=np.float64), None, int(roi.size)

    step = max(1, int(depth_config.point_sample_step))
    ys = np.arange(y1, y2 + 1, step)
    xs = np.arange(x1, x2 + 1, step)
    grid_x, grid_y = np.meshgrid(xs, ys)
    depths = frame.depth_m[grid_y, grid_x]
    valid = (
        np.isfinite(depths)
        & (depths >= depth_config.min_m)
        & (depths <= depth_config.max_m)
        & (np.abs(depths - center_depth) <= depth_config.object_band_m)
    )
    if detection.mask is not None:
        mask = detection.mask
        if mask.shape != frame.depth_m.shape:
            import cv2

            mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
        valid &= mask[grid_y, grid_x] > 0.5
    z = depths[valid].astype(np.float64)
    px = grid_x[valid].astype(np.float64)
    py = grid_y[valid].astype(np.float64)
    points = np.column_stack(
        (
            (px - frame.intrinsics.cx) * z / frame.intrinsics.fx,
            (py - frame.intrinsics.cy) * z / frame.intrinsics.fy,
            z,
        )
    )
    return points, center_depth, int(grid_x.size)


def pca_object_rotation(points: np.ndarray) -> Tuple[np.ndarray, float]:
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 3:
        raise ValueError("at least three XYZ points are required")
    centered = points - np.mean(points, axis=0)
    covariance = centered.T @ centered / max(1, len(points) - 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    values = np.maximum(eigenvalues[order], 1e-12)
    vectors = eigenvectors[:, order]

    z_axis = vectors[:, 0]  # longest object axis
    centroid = np.mean(points, axis=0)
    normal = vectors[:, 2]  # thinnest surface/object direction
    if np.dot(z_axis, np.array([0.0, -1.0, 0.0])) < 0:
        z_axis = -z_axis
    if np.dot(normal, -centroid) < 0:
        normal = -normal
    x_axis = np.cross(normal, z_axis)
    norm = np.linalg.norm(x_axis)
    if norm < 1e-8:
        x_axis = vectors[:, 1]
    else:
        x_axis /= norm
    y_axis = np.cross(z_axis, x_axis)
    y_axis /= max(np.linalg.norm(y_axis), 1e-12)
    if np.dot(x_axis, np.array([1.0, 0.0, 0.0])) < 0:
        x_axis = -x_axis
        y_axis = -y_axis
    rotation = nearest_rotation(np.column_stack((x_axis, y_axis, z_axis)))
    linearity = float(np.clip((values[0] - values[1]) / values[0], 0.0, 1.0))
    planarity = float(np.clip((values[1] - values[2]) / values[0], 0.0, 1.0))
    quality = 0.65 * linearity + 0.35 * planarity
    return rotation, quality


def oriented_box_from_points(
    points: np.ndarray, rotation: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Return a robust oriented-box center and XYZ dimensions in the pose axes."""

    points = np.asarray(points, dtype=np.float64)
    rotation = nearest_rotation(rotation)
    reference = np.median(points, axis=0)
    local = (points - reference) @ rotation
    lower, upper = np.percentile(local, [2.0, 98.0], axis=0)
    dimensions = np.maximum(upper - lower, 0.0)
    # A single RGB-D view mainly observes the front surface. Give the unobserved
    # thickness a conservative minimum so its projected box remains legible.
    minimum = float(np.clip(np.max(dimensions) * 0.12, 0.012, 0.05))
    dimensions = np.maximum(dimensions, minimum)
    local_center = (lower + upper) * 0.5
    center = reference + rotation @ local_center
    return center, dimensions


class ObjectPoseEstimator:
    def __init__(self, depth_config: DepthConfig, pose_config: PoseConfig) -> None:
        self.depth_config = depth_config
        self.pose_config = pose_config
        self.smoother = PoseSmoother(
            pose_config.smoothing_alpha, pose_config.track_max_center_distance_px
        )

    def estimate(
        self, detection: Detection2D, frame: FrameBundle, sequence: int
    ) -> ObjectObservation:
        if frame.depth_m is None:
            return ObjectObservation(detection, None, False, "no aligned depth")
        points, depth, sampled = bbox_point_cloud(detection, frame, self.depth_config)
        if depth is None:
            return ObjectObservation(detection, None, False, "no valid depth")
        u, v = detection.center
        center = deproject_pixel(u, v, depth, frame.intrinsics)
        if len(points) < self.depth_config.min_pose_points:
            width_m = detection.size[0] * depth / frame.intrinsics.fx
            height_m = detection.size[1] * depth / frame.intrinsics.fy
            thickness_m = max(0.012, min(width_m, height_m) * 0.25)
            pose = _make_pose(
                center,
                np.eye(3),
                frame.camera_frame,
                "center_depth_orientation_unavailable",
                min(0.25, len(points) / max(1, self.depth_config.min_pose_points) * 0.25),
                (width_m, height_m, thickness_m),
            )
            pose = self.smoother.update(detection, pose, sequence)
            return ObjectObservation(
                detection,
                pose,
                False,
                f"insufficient pose points: {len(points)}/{self.depth_config.min_pose_points}",
                depth,
            )
        rotation, shape_quality = pca_object_rotation(points)
        centroid, dimensions = oriented_box_from_points(points, rotation)
        density = min(1.0, len(points) / max(1.0, sampled * 0.35))
        pose = _make_pose(
            centroid,
            rotation,
            frame.camera_frame,
            "depth_pca",
            0.55 * shape_quality + 0.45 * density,
            dimensions,
        )
        pose = self.smoother.update(detection, pose, sequence)
        return ObjectObservation(detection, pose, True, "", depth)

    def estimate_all(
        self, detections: Sequence[Detection2D], frame: FrameBundle, sequence: int
    ) -> List[ObjectObservation]:
        return [self.estimate(item, frame, sequence) for item in detections]


def transform_pose(pose: Pose6D, target_T_source: np.ndarray, target_frame: str) -> Pose6D:
    transform = validate_transform(target_T_source)
    source_T_object = make_transform(pose.rotation_matrix, pose.translation_m)
    target_T_object = transform @ source_T_object
    return _make_pose(
        target_T_object[:3, 3],
        target_T_object[:3, :3],
        target_frame,
        pose.method + "+transform",
        pose.quality,
        pose.dimensions_m,
    )

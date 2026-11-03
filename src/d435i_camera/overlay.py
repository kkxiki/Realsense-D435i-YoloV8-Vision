"""Real-time bounding boxes, axes and Chinese six-dimensional overlays."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .geometry import make_transform, project_point
from .labels import chinese_class_name
from .models import FrameBundle, ObjectObservation


COLORS = [
    (80, 220, 120),
    (70, 180, 255),
    (230, 120, 80),
    (210, 100, 220),
    (220, 210, 80),
]

FONT_CANDIDATES = (
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Medium.ttc",
    "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    "C:/Windows/Fonts/msyh.ttc",
    "C:/Windows/Fonts/simhei.ttf",
)


@lru_cache(maxsize=8)
def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for candidate in FONT_CANDIDATES:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def _six_dimensional_lines(observation: ObjectObservation) -> List[str]:
    """Per-object label: Chinese name, confidence and X/Y/Z/Rx/Ry/Rz."""

    name = chinese_class_name(observation.detection.class_name)
    if observation.pose is None:
        return [
            f"物品中文名称：{name}   置信度：{observation.detection.confidence * 100.0:.1f}%",
            "X: -- mm   Y: -- mm   Z: -- mm",
            "Rx: --°   Ry: --°   Rz: --°",
        ]
    x, y, z = [value * 1000.0 for value in observation.pose.translation_m]
    rx, ry, rz = observation.pose.rpy_deg
    return [
        f"物品中文名称：{name}   置信度：{observation.detection.confidence * 100.0:.1f}%",
        f"X: {x:+.1f} mm   Y: {y:+.1f} mm   Z: {z:+.1f} mm",
        f"Rx: {rx:+.1f}°   Ry: {ry:+.1f}°   Rz: {rz:+.1f}°",
    ]


def _draw_text_blocks(
    image_bgr: np.ndarray,
    blocks: Sequence[Tuple[int, int, List[str], Tuple[int, int, int], int]],
) -> np.ndarray:
    """Render all Unicode text in one Pillow conversion for real-time use."""

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    canvas = Image.fromarray(image_rgb)
    draw = ImageDraw.Draw(canvas, "RGBA")
    width, height = canvas.size
    for x, y, lines, bgr, font_size in blocks:
        font = _font(font_size)
        spacing = max(3, font_size // 4)
        text = "\n".join(lines)
        bbox = draw.multiline_textbbox((0, 0), text, font=font, spacing=spacing, stroke_width=1)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        x = max(4, min(int(x), max(4, width - text_width - 16)))
        y = max(4, min(int(y), max(4, height - text_height - 16)))
        rgb = (int(bgr[2]), int(bgr[1]), int(bgr[0]))
        draw.rounded_rectangle(
            (x, y, x + text_width + 14, y + text_height + 12),
            radius=5,
            fill=(10, 14, 19, 205),
            outline=(*rgb, 245),
            width=2,
        )
        draw.multiline_text(
            (x + 7, y + 5),
            text,
            font=font,
            fill=(245, 248, 252, 255),
            spacing=spacing,
            stroke_width=1,
            stroke_fill=(0, 0, 0, 220),
        )
    return cv2.cvtColor(np.asarray(canvas), cv2.COLOR_RGB2BGR)


def draw_pose_axes(
    image: np.ndarray,
    observation: ObjectObservation,
    frame: FrameBundle,
    axis_length_m: float,
    camera_T_pose_frame: np.ndarray | None = None,
) -> None:
    pose = observation.pose
    if pose is None:
        return
    pose_matrix = make_transform(pose.rotation_matrix, pose.translation_m)
    if pose.frame_id != frame.camera_frame:
        if camera_T_pose_frame is None:
            return
        pose_matrix = camera_T_pose_frame @ pose_matrix
    origin_xyz = pose_matrix[:3, 3]
    origin = project_point(origin_xyz, frame.intrinsics)
    if origin is None:
        return
    # OpenCV BGR: X red, Y green, Z blue.
    colors = [(50, 50, 245), (70, 225, 80), (245, 100, 40)]
    for index, color in enumerate(colors):
        endpoint = project_point(
            origin_xyz + pose_matrix[:3, :3][:, index] * float(axis_length_m),
            frame.intrinsics,
        )
        if endpoint is not None:
            cv2.arrowedLine(image, origin, endpoint, color, 2, cv2.LINE_AA, tipLength=0.2)
    cv2.circle(image, origin, 4, (255, 255, 255), -1, cv2.LINE_AA)


def draw_3d_box(
    image: np.ndarray,
    observation: ObjectObservation,
    frame: FrameBundle,
    color: Tuple[int, int, int],
    camera_T_pose_frame: np.ndarray | None = None,
) -> List[Tuple[int, int]]:
    """Project the metric oriented box into the color image and draw all 12 edges."""

    pose = observation.pose
    if pose is None or min(pose.dimensions_m) <= 0.0:
        return []
    pose_matrix = make_transform(pose.rotation_matrix, pose.translation_m)
    if pose.frame_id != frame.camera_frame:
        if camera_T_pose_frame is None:
            return []
        pose_matrix = camera_T_pose_frame @ pose_matrix
    half = np.asarray(pose.dimensions_m, dtype=np.float64) * 0.5
    signs = np.asarray(
        [
            [-1, -1, -1],
            [1, -1, -1],
            [1, 1, -1],
            [-1, 1, -1],
            [-1, -1, 1],
            [1, -1, 1],
            [1, 1, 1],
            [-1, 1, 1],
        ],
        dtype=np.float64,
    )
    local_corners = signs * half
    camera_corners = (
        pose_matrix[:3, :3] @ local_corners.T
    ).T + pose_matrix[:3, 3]
    projected = [project_point(point, frame.intrinsics) for point in camera_corners]
    if any(point is None for point in projected):
        return []
    pixels = [point for point in projected if point is not None]
    rear_edges = ((0, 1), (1, 2), (2, 3), (3, 0))
    front_edges = ((4, 5), (5, 6), (6, 7), (7, 4))
    side_edges = ((0, 4), (1, 5), (2, 6), (3, 7))
    muted = tuple(int(channel * 0.62) for channel in color)
    for first, second in rear_edges:
        cv2.line(image, pixels[first], pixels[second], muted, 2, cv2.LINE_AA)
    for first, second in side_edges:
        cv2.line(image, pixels[first], pixels[second], color, 2, cv2.LINE_AA)
    for first, second in front_edges:
        cv2.line(image, pixels[first], pixels[second], color, 3, cv2.LINE_AA)
    for point in pixels:
        cv2.circle(image, point, 3, color, -1, cv2.LINE_AA)
    return pixels


def draw_overlay(
    frame: FrameBundle,
    observations: Iterable[ObjectObservation],
    processing_ms: float = 0.0,
    show_panel: bool = True,
    show_axes: bool = True,
    axis_length_m: float = 0.06,
    camera_T_pose_frame: np.ndarray | None = None,
) -> np.ndarray:
    del processing_ms  # Timing belongs in the GUI status bar, not in the 6D panel.
    image = frame.color_bgr.copy()
    observations = list(observations)
    height, width = image.shape[:2]
    text_blocks: List[Tuple[int, int, List[str], Tuple[int, int, int], int]] = []
    for index, observation in enumerate(observations):
        detection = observation.detection.clipped(width, height)
        color = COLORS[index % len(COLORS)] if observation.valid else (70, 80, 240)
        x1, y1, x2, y2 = [int(round(value)) for value in detection.bbox_xyxy]
        box_points = draw_3d_box(
            image,
            observation,
            frame,
            color,
            camera_T_pose_frame=camera_T_pose_frame,
        )
        if not box_points:
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
        cx, cy = [int(round(value)) for value in detection.center]
        cv2.drawMarker(image, (cx, cy), color, cv2.MARKER_CROSS, 14, 2)
        box_x = x1
        box_y = y1 - 90 if y1 >= 98 else y1 + 6
        # Keep per-object 6D labels out of the top-left summary panel.
        if show_panel and x1 < min(width - 20, 500) and y1 < min(height - 20, 250):
            if y2 + 82 < height:
                box_y = y2 + 6
            else:
                box_x = x2 + 6
                box_y = y1
        text_blocks.append((box_x, box_y, _six_dimensional_lines(observation), color, 15))
        if show_axes:
            draw_pose_axes(
                image,
                observation,
                frame,
                axis_length_m,
                camera_T_pose_frame=camera_T_pose_frame,
            )

    if show_panel:
        if observations:
            panel_lines: List[str] = []
            for index, item in enumerate(observations[:5]):
                if index:
                    panel_lines.append("")
                panel_lines.extend(_six_dimensional_lines(item))
        else:
            panel_lines = ["未识别到物体", "X: -- mm   Y: -- mm   Z: -- mm", "Rx: --°   Ry: --°   Rz: --°"]
        text_blocks.append((10, 10, panel_lines, (110, 170, 255), 18))
    return _draw_text_blocks(image, text_blocks) if text_blocks else image


def depth_colormap(depth_m: np.ndarray, min_m: float, max_m: float) -> np.ndarray:
    clipped = np.clip(depth_m, min_m, max_m)
    normalized = ((clipped - min_m) / (max_m - min_m) * 255.0).astype(np.uint8)
    colored = cv2.applyColorMap(255 - normalized, cv2.COLORMAP_TURBO)
    colored[~np.isfinite(depth_m) | (depth_m <= 0)] = 0
    return colored

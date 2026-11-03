"""Multi-view OpenCV intrinsic calibration with per-view quality reporting."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from ..config import BoardConfig
from ..models import Intrinsics
from .common import board_object_points, find_chessboard


@dataclass(frozen=True)
class IntrinsicCalibrationResult:
    intrinsics: Intrinsics
    rms_px: float
    per_view_rms_px: Tuple[float, ...]
    sample_count: int
    board: BoardConfig
    created_at: str

    def as_dict(self) -> Dict[str, Any]:
        return {
            "schema": "xuk.camera_intrinsics/v1",
            "created_at": self.created_at,
            "sample_count": self.sample_count,
            "rms_px": self.rms_px,
            "per_view_rms_px": list(self.per_view_rms_px),
            "camera": self.intrinsics.as_dict(),
            "board": {
                "type": self.board.type,
                "inner_corners_cols": self.board.inner_corners_cols,
                "inner_corners_rows": self.board.inner_corners_rows,
                "square_size_mm": self.board.square_size_mm,
                "confirmed": self.board.confirmed,
            },
        }

    def save(self, path: str | Path) -> Path:
        destination = Path(path).expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(self.as_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return destination


class IntrinsicCalibrationSession:
    def __init__(self, board: BoardConfig, min_samples: int = 12) -> None:
        board.require_confirmed()
        self.board = board
        self.min_samples = int(min_samples)
        self.image_points: List[np.ndarray] = []
        self.image_size: Optional[Tuple[int, int]] = None
        self.last_preview: Optional[np.ndarray] = None

    @property
    def sample_count(self) -> int:
        return len(self.image_points)

    def reset(self) -> None:
        self.image_points.clear()
        self.image_size = None
        self.last_preview = None

    def add_frame(self, color_bgr: np.ndarray) -> Tuple[bool, str, np.ndarray]:
        found, corners, preview = find_chessboard(color_bgr, self.board, accurate=True)
        self.last_preview = preview
        if not found or corners is None:
            return False, "未检测到完整棋盘", preview
        height, width = color_bgr.shape[:2]
        current_size = (width, height)
        if self.image_size is not None and self.image_size != current_size:
            return False, f"图像尺寸从 {self.image_size} 变为 {current_size}", preview

        # Reject nearly identical captures; diverse image coverage is more useful than count.
        flat = corners.reshape(-1, 2)
        signature = np.array(
            [
                flat[:, 0].mean() / width,
                flat[:, 1].mean() / height,
                np.ptp(flat[:, 0]) / width,
                np.ptp(flat[:, 1]) / height,
            ]
        )
        for previous in self.image_points:
            old = previous.reshape(-1, 2)
            old_signature = np.array(
                [
                    old[:, 0].mean() / width,
                    old[:, 1].mean() / height,
                    np.ptp(old[:, 0]) / width,
                    np.ptp(old[:, 1]) / height,
                ]
            )
            if float(np.linalg.norm(signature - old_signature)) < 0.025:
                return False, "该视角与已有样本过于相似，请移动/倾斜标定板", preview
        self.image_size = current_size
        self.image_points.append(corners.copy())
        return True, f"已采集内参样本 {self.sample_count}", preview

    def solve(self) -> IntrinsicCalibrationResult:
        if self.sample_count < self.min_samples:
            raise ValueError(f"内参样本不足：{self.sample_count}/{self.min_samples}")
        assert self.image_size is not None
        object_template = board_object_points(self.board)
        object_points = [object_template.copy() for _ in self.image_points]
        rms, camera_matrix, dist, rvecs, tvecs = cv2.calibrateCamera(
            object_points,
            self.image_points,
            self.image_size,
            None,
            None,
        )
        per_view: List[float] = []
        for object_view, image_view, rvec, tvec in zip(
            object_points, self.image_points, rvecs, tvecs
        ):
            projected, _ = cv2.projectPoints(
                object_view, rvec, tvec, camera_matrix, dist
            )
            residual = projected.reshape(-1, 2) - image_view.reshape(-1, 2)
            per_view.append(float(np.sqrt(np.mean(np.sum(residual**2, axis=1)))))
        width, height = self.image_size
        intrinsics = Intrinsics(
            width=width,
            height=height,
            fx=float(camera_matrix[0, 0]),
            fy=float(camera_matrix[1, 1]),
            cx=float(camera_matrix[0, 2]),
            cy=float(camera_matrix[1, 2]),
            distortion_model="opencv_plumb_bob",
            coeffs=tuple(float(value) for value in dist.reshape(-1)),
        )
        return IntrinsicCalibrationResult(
            intrinsics=intrinsics,
            rms_px=float(rms),
            per_view_rms_px=tuple(per_view),
            sample_count=self.sample_count,
            board=self.board,
            created_at=datetime.now(timezone.utc).isoformat(),
        )


def load_intrinsics(path: str | Path) -> Intrinsics:
    data = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    camera = data.get("camera", data)
    return Intrinsics(
        width=int(camera["width"]),
        height=int(camera["height"]),
        fx=float(camera["fx"]),
        fy=float(camera["fy"]),
        cx=float(camera["cx"]),
        cy=float(camera["cy"]),
        distortion_model=str(camera.get("distortion_model", "none")),
        coeffs=tuple(float(value) for value in camera.get("coeffs", [])),
    )


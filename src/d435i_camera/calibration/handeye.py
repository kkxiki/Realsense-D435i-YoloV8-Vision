"""Eye-in-hand calibration: camera on tool, chessboard fixed in the robot base."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import cv2
import numpy as np

from ..config import BoardConfig
from ..geometry import (
    make_transform,
    nearest_rotation,
    rotation_angle_deg,
    validate_transform,
)
from ..models import Intrinsics
from .common import find_chessboard, solve_board_pose


HAND_EYE_METHODS = {
    "TSAI": cv2.CALIB_HAND_EYE_TSAI,
    "PARK": cv2.CALIB_HAND_EYE_PARK,
    "HORAUD": cv2.CALIB_HAND_EYE_HORAUD,
    "ANDREFF": cv2.CALIB_HAND_EYE_ANDREFF,
    "DANIILIDIS": cv2.CALIB_HAND_EYE_DANIILIDIS,
}


@dataclass(frozen=True)
class HandEyeSample:
    base_T_tool: np.ndarray
    camera_T_board: np.ndarray
    reprojection_rms_px: float
    timestamp_s: float


@dataclass(frozen=True)
class HandEyeCalibrationResult:
    tool_T_camera: np.ndarray
    method: str
    sample_count: int
    translation_rms_mm: float
    translation_max_mm: float
    rotation_rms_deg: float
    rotation_max_deg: float
    mean_reprojection_rms_px: float
    per_sample_translation_mm: Tuple[float, ...]
    per_sample_rotation_deg: Tuple[float, ...]
    board: BoardConfig
    created_at: str

    def as_dict(self) -> Dict[str, Any]:
        return {
            "schema": "xuk.eye_in_hand/v1",
            "created_at": self.created_at,
            "mount": "eye_in_hand",
            "transform_convention": "tool_T_camera (camera coordinates -> tool coordinates)",
            "method": self.method,
            "sample_count": self.sample_count,
            "tool_T_camera": self.tool_T_camera.tolist(),
            "translation_m": self.tool_T_camera[:3, 3].tolist(),
            "rotation": self.tool_T_camera[:3, :3].tolist(),
            "quality": {
                "translation_rms_mm": self.translation_rms_mm,
                "translation_max_mm": self.translation_max_mm,
                "rotation_rms_deg": self.rotation_rms_deg,
                "rotation_max_deg": self.rotation_max_deg,
                "mean_reprojection_rms_px": self.mean_reprojection_rms_px,
                "per_sample_translation_mm": list(self.per_sample_translation_mm),
                "per_sample_rotation_deg": list(self.per_sample_rotation_deg),
            },
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


class HandEyeCalibrationSession:
    """Collect paired base_T_tool and camera_T_board samples, then solve tool_T_camera."""

    def __init__(
        self,
        board: BoardConfig,
        intrinsics: Intrinsics,
        min_samples: int = 12,
        method: str = "PARK",
    ) -> None:
        board.require_confirmed()
        intrinsics.validate()
        method = method.upper()
        if method not in HAND_EYE_METHODS:
            raise ValueError(f"unknown hand-eye method {method}")
        self.board = board
        self.intrinsics = intrinsics
        self.min_samples = int(min_samples)
        self.method = method
        self.samples: List[HandEyeSample] = []
        self.last_preview: np.ndarray | None = None

    @property
    def sample_count(self) -> int:
        return len(self.samples)

    def reset(self) -> None:
        self.samples.clear()
        self.last_preview = None

    def add_frame(
        self, color_bgr: np.ndarray, base_T_tool: np.ndarray, timestamp_s: float
    ) -> Tuple[bool, str, np.ndarray]:
        base_T_tool = validate_transform(base_T_tool)
        found, corners, preview = find_chessboard(color_bgr, self.board, accurate=True)
        self.last_preview = preview
        if not found or corners is None:
            return False, "未检测到完整棋盘", preview
        camera_T_board, reprojection_rms = solve_board_pose(
            corners, self.board, self.intrinsics
        )

        if self.samples:
            last = self.samples[-1].base_T_tool
            translation_delta_mm = float(
                np.linalg.norm(base_T_tool[:3, 3] - last[:3, 3]) * 1000.0
            )
            rotation_delta_deg = rotation_angle_deg(last[:3, :3].T @ base_T_tool[:3, :3])
            if translation_delta_mm < 10.0 and rotation_delta_deg < 4.0:
                return False, "机械臂姿态变化不足（需平移≥10 mm或旋转≥4°）", preview

        self.samples.append(
            HandEyeSample(
                base_T_tool=base_T_tool.copy(),
                camera_T_board=camera_T_board,
                reprojection_rms_px=reprojection_rms,
                timestamp_s=float(timestamp_s),
            )
        )
        return (
            True,
            f"已采集手眼样本 {self.sample_count}，重投影 RMS={reprojection_rms:.3f}px",
            preview,
        )

    def solve(self) -> HandEyeCalibrationResult:
        if self.sample_count < self.min_samples:
            raise ValueError(f"手眼样本不足：{self.sample_count}/{self.min_samples}")
        gripper_rotations = [sample.base_T_tool[:3, :3] for sample in self.samples]
        gripper_translations = [sample.base_T_tool[:3, 3] for sample in self.samples]
        target_rotations = [sample.camera_T_board[:3, :3] for sample in self.samples]
        target_translations = [sample.camera_T_board[:3, 3] for sample in self.samples]
        camera_to_tool_r, camera_to_tool_t = cv2.calibrateHandEye(
            gripper_rotations,
            gripper_translations,
            target_rotations,
            target_translations,
            method=HAND_EYE_METHODS[self.method],
        )
        tool_T_camera = make_transform(camera_to_tool_r, camera_to_tool_t.reshape(3))

        # The board is fixed in base. A correct solution makes base_T_board constant.
        base_T_boards = [
            sample.base_T_tool @ tool_T_camera @ sample.camera_T_board
            for sample in self.samples
        ]
        mean_translation = np.mean([item[:3, 3] for item in base_T_boards], axis=0)
        mean_rotation = nearest_rotation(sum(item[:3, :3] for item in base_T_boards))
        translation_errors = np.asarray(
            [np.linalg.norm(item[:3, 3] - mean_translation) * 1000.0 for item in base_T_boards]
        )
        rotation_errors = np.asarray(
            [rotation_angle_deg(mean_rotation.T @ item[:3, :3]) for item in base_T_boards]
        )
        return HandEyeCalibrationResult(
            tool_T_camera=tool_T_camera,
            method=self.method,
            sample_count=self.sample_count,
            translation_rms_mm=float(np.sqrt(np.mean(translation_errors**2))),
            translation_max_mm=float(np.max(translation_errors)),
            rotation_rms_deg=float(np.sqrt(np.mean(rotation_errors**2))),
            rotation_max_deg=float(np.max(rotation_errors)),
            mean_reprojection_rms_px=float(
                np.mean([sample.reprojection_rms_px for sample in self.samples])
            ),
            per_sample_translation_mm=tuple(float(value) for value in translation_errors),
            per_sample_rotation_deg=tuple(float(value) for value in rotation_errors),
            board=self.board,
            created_at=datetime.now(timezone.utc).isoformat(),
        )


def load_handeye(path: str | Path) -> np.ndarray:
    data = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    matrix = np.asarray(data["tool_T_camera"], dtype=np.float64)
    return validate_transform(matrix)

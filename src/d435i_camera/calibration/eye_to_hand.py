"""Eye-to-hand calibration for a fixed camera observing a board on the robot tool."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np

from ..config import BoardConfig
from ..geometry import make_transform, rotation_angle_deg, validate_transform
from ..models import Intrinsics
from .common import find_chessboard, solve_board_pose


EYE_TO_HAND_METHODS = {
    "SHAH": cv2.CALIB_ROBOT_WORLD_HAND_EYE_SHAH,
    "LI": cv2.CALIB_ROBOT_WORLD_HAND_EYE_LI,
}


@dataclass(frozen=True)
class EyeToHandSample:
    """One synchronized robot/camera observation.

    ``base_T_tool`` transforms tool-frame points into robot-base coordinates.
    ``camera_T_board`` transforms board-frame points into camera coordinates.
    """

    base_T_tool: np.ndarray
    camera_T_board: np.ndarray
    reprojection_rms_px: float
    timestamp_s: float


@dataclass(frozen=True)
class EyeToHandCalibrationResult:
    """Fixed-camera calibration result.

    The primary output is ``base_T_camera``. ``tool_T_board`` is solved jointly
    and is useful for checking that the calibration board did not move on the tool.
    """

    base_T_camera: np.ndarray
    tool_T_board: np.ndarray
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
            "schema": "xuk.eye_to_hand/v1",
            "created_at": self.created_at,
            "mount": "fixed_head_camera",
            "transform_convention": "base_T_camera (camera coordinates -> robot base coordinates)",
            "equation": "base_T_tool * tool_T_board = base_T_camera * camera_T_board",
            "method": self.method,
            "sample_count": self.sample_count,
            "base_T_camera": self.base_T_camera.tolist(),
            "tool_T_board": self.tool_T_board.tolist(),
            "translation_m": self.base_T_camera[:3, 3].tolist(),
            "rotation": self.base_T_camera[:3, :3].tolist(),
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


class EyeToHandCalibrationSession:
    """Solve a fixed head camera from paired robot poses and board observations.

    The camera must remain fixed relative to the selected robot base/reference frame,
    while the calibration board must remain rigidly attached to the tool.
    """

    def __init__(
        self,
        board: BoardConfig,
        intrinsics: Intrinsics,
        min_samples: int = 12,
        method: str = "SHAH",
    ) -> None:
        board.require_confirmed()
        intrinsics.validate()
        method = method.upper()
        if method not in EYE_TO_HAND_METHODS:
            raise ValueError(f"unknown eye-to-hand method {method}")
        self.board = board
        self.intrinsics = intrinsics
        self.min_samples = max(3, int(min_samples))
        self.method = method
        self.samples: List[EyeToHandSample] = []
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
                return False, "末端姿态变化不足（需平移≥10 mm 或旋转≥4°）", preview

        self.samples.append(
            EyeToHandSample(
                base_T_tool=base_T_tool.copy(),
                camera_T_board=camera_T_board,
                reprojection_rms_px=reprojection_rms,
                timestamp_s=float(timestamp_s),
            )
        )
        return (
            True,
            f"已采集眼在手外样本 {self.sample_count}，重投影 RMS={reprojection_rms:.3f}px",
            preview,
        )

    def solve(self) -> EyeToHandCalibrationResult:
        if self.sample_count < self.min_samples:
            raise ValueError(f"眼在手外样本不足：{self.sample_count}/{self.min_samples}")

        # OpenCV solves A X = Z B.  Map our equation as follows:
        # A=base_T_tool, X=tool_T_board, Z=base_T_camera, B=camera_T_board.
        tool_to_board_r, tool_to_board_t, base_to_camera_r, base_to_camera_t = (
            cv2.calibrateRobotWorldHandEye(
                [sample.base_T_tool[:3, :3] for sample in self.samples],
                [sample.base_T_tool[:3, 3] for sample in self.samples],
                [sample.camera_T_board[:3, :3] for sample in self.samples],
                [sample.camera_T_board[:3, 3] for sample in self.samples],
                method=EYE_TO_HAND_METHODS[self.method],
            )
        )
        tool_T_board = make_transform(tool_to_board_r, tool_to_board_t.reshape(3))
        base_T_camera = make_transform(base_to_camera_r, base_to_camera_t.reshape(3))

        translation_errors = []
        rotation_errors = []
        for sample in self.samples:
            left = sample.base_T_tool @ tool_T_board
            right = base_T_camera @ sample.camera_T_board
            translation_errors.append(np.linalg.norm(left[:3, 3] - right[:3, 3]) * 1000.0)
            rotation_errors.append(rotation_angle_deg(left[:3, :3].T @ right[:3, :3]))
        translation_values = np.asarray(translation_errors, dtype=np.float64)
        rotation_values = np.asarray(rotation_errors, dtype=np.float64)

        return EyeToHandCalibrationResult(
            base_T_camera=base_T_camera,
            tool_T_board=tool_T_board,
            method=self.method,
            sample_count=self.sample_count,
            translation_rms_mm=float(np.sqrt(np.mean(translation_values**2))),
            translation_max_mm=float(np.max(translation_values)),
            rotation_rms_deg=float(np.sqrt(np.mean(rotation_values**2))),
            rotation_max_deg=float(np.max(rotation_values)),
            mean_reprojection_rms_px=float(
                np.mean([sample.reprojection_rms_px for sample in self.samples])
            ),
            per_sample_translation_mm=tuple(float(value) for value in translation_values),
            per_sample_rotation_deg=tuple(float(value) for value in rotation_values),
            board=self.board,
            created_at=datetime.now(timezone.utc).isoformat(),
        )


def load_eye_to_hand(path: str | Path) -> np.ndarray:
    data = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    matrix = np.asarray(data["base_T_camera"], dtype=np.float64)
    return validate_transform(matrix)

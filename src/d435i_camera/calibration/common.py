"""Chessboard detection and board pose utilities shared by both calibrators."""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np

from ..config import BoardConfig
from ..models import Intrinsics


def board_object_points(board: BoardConfig) -> np.ndarray:
    board.require_confirmed()
    points = np.zeros(
        (board.inner_corners_cols * board.inner_corners_rows, 3), dtype=np.float32
    )
    points[:, :2] = np.mgrid[
        0 : board.inner_corners_cols, 0 : board.inner_corners_rows
    ].T.reshape(-1, 2)
    points *= float(board.square_size_mm) / 1000.0
    return points


def find_chessboard(
    color_bgr: np.ndarray, board: BoardConfig, accurate: bool = True
) -> Tuple[bool, Optional[np.ndarray], np.ndarray]:
    board.require_confirmed()
    gray = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2GRAY)
    pattern = (board.inner_corners_cols, board.inner_corners_rows)
    corners: Optional[np.ndarray] = None
    found = False
    if accurate and hasattr(cv2, "findChessboardCornersSB"):
        flags = cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY | cv2.CALIB_CB_NORMALIZE_IMAGE
        found, detected = cv2.findChessboardCornersSB(gray, pattern, flags=flags)
        if found:
            corners = np.asarray(detected, dtype=np.float32)
    if not found:
        flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
        found, detected = cv2.findChessboardCorners(gray, pattern, flags=flags)
        if found:
            criteria = (
                cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                60,
                1e-4,
            )
            corners = cv2.cornerSubPix(gray, detected, (7, 7), (-1, -1), criteria)
    preview = color_bgr.copy()
    if found and corners is not None:
        cv2.drawChessboardCorners(preview, pattern, corners, True)
    return bool(found), corners, preview


def solve_board_pose(
    corners: np.ndarray, board: BoardConfig, intrinsics: Intrinsics
) -> Tuple[np.ndarray, float]:
    object_points = board_object_points(board).astype(np.float64)
    image_points = np.asarray(corners, dtype=np.float64).reshape(-1, 2)
    success, rvec, tvec = cv2.solvePnP(
        object_points,
        image_points,
        intrinsics.matrix,
        intrinsics.distortion,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not success:
        raise RuntimeError("OpenCV solvePnP failed for the calibration board")
    projected, _ = cv2.projectPoints(
        object_points,
        rvec,
        tvec,
        intrinsics.matrix,
        intrinsics.distortion,
    )
    residual = projected.reshape(-1, 2) - image_points
    reprojection_rms = float(np.sqrt(np.mean(np.sum(residual**2, axis=1))))
    rotation, _ = cv2.Rodrigues(rvec)
    camera_T_board = np.eye(4, dtype=np.float64)
    camera_T_board[:3, :3] = rotation
    camera_T_board[:3, 3] = tvec.reshape(3)
    return camera_T_board, reprojection_rms


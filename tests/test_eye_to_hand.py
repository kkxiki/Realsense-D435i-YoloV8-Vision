import numpy as np

from d435i_camera.calibration.eye_to_hand import (
    EyeToHandCalibrationSession,
    EyeToHandSample,
)
from d435i_camera.config import BoardConfig
from d435i_camera.geometry import invert_transform, make_transform, rotation_angle_deg, rpy_deg_to_matrix
from d435i_camera.models import Intrinsics


def test_synthetic_fixed_head_camera_solver_recovers_base_camera_transform():
    rng = np.random.default_rng(27)
    board = BoardConfig(
        confirmed=True,
        inner_corners_cols=8,
        inner_corners_rows=11,
        square_size_mm=15,
    )
    intrinsics = Intrinsics(848, 480, 615.0, 615.0, 424.0, 240.0)
    expected_base_T_camera = make_transform(
        rpy_deg_to_matrix(-8, 12, 178), [0.72, -0.16, 1.12]
    )
    expected_tool_T_board = make_transform(
        rpy_deg_to_matrix(0, 90, 4), [0.04, 0.0, 0.12]
    )
    session = EyeToHandCalibrationSession(board, intrinsics, min_samples=12, method="SHAH")
    for index in range(30):
        rpy = rng.uniform([-55, -45, -110], [55, 45, 110])
        xyz = rng.uniform([0.25, -0.32, 0.18], [0.72, 0.32, 0.72])
        base_T_tool = make_transform(rpy_deg_to_matrix(*rpy), xyz)
        camera_T_board = (
            invert_transform(expected_base_T_camera) @ base_T_tool @ expected_tool_T_board
        )
        session.samples.append(EyeToHandSample(base_T_tool, camera_T_board, 0.08, float(index)))

    result = session.solve()
    translation_error_mm = np.linalg.norm(
        result.base_T_camera[:3, 3] - expected_base_T_camera[:3, 3]
    ) * 1000.0
    rotation_error_deg = rotation_angle_deg(
        expected_base_T_camera[:3, :3].T @ result.base_T_camera[:3, :3]
    )
    assert translation_error_mm < 0.1
    assert rotation_error_deg < 0.01
    assert result.translation_rms_mm < 0.1
    assert result.rotation_rms_deg < 0.01


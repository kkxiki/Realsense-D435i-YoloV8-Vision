import numpy as np

from d435i_camera.calibration.handeye import HandEyeCalibrationSession, HandEyeSample
from d435i_camera.config import BoardConfig
from d435i_camera.geometry import invert_transform, make_transform, rotation_angle_deg, rpy_deg_to_matrix
from d435i_camera.models import Intrinsics


def test_synthetic_eye_in_hand_solver_recovers_tool_camera_transform():
    rng = np.random.default_rng(11)
    board = BoardConfig(confirmed=True, inner_corners_cols=8, inner_corners_rows=11, square_size_mm=15)
    intrinsics = Intrinsics(848, 480, 615.0, 615.0, 424.0, 240.0)
    expected_tool_T_camera = make_transform(rpy_deg_to_matrix(8, -15, 92), [0.035, -0.012, 0.082])
    base_T_board = make_transform(rpy_deg_to_matrix(0, 0, 20), [0.55, 0.05, 0.18])
    session = HandEyeCalibrationSession(board, intrinsics, min_samples=12, method="PARK")
    for index in range(24):
        rpy = rng.uniform([-55, -45, -100], [55, 45, 100])
        xyz = rng.uniform([0.25, -0.25, 0.15], [0.65, 0.25, 0.55])
        base_T_tool = make_transform(rpy_deg_to_matrix(*rpy), xyz)
        camera_T_board = invert_transform(expected_tool_T_camera) @ invert_transform(base_T_tool) @ base_T_board
        session.samples.append(HandEyeSample(base_T_tool, camera_T_board, 0.1, float(index)))
    result = session.solve()
    translation_error_mm = np.linalg.norm(
        result.tool_T_camera[:3, 3] - expected_tool_T_camera[:3, 3]
    ) * 1000.0
    rotation_error_deg = rotation_angle_deg(
        expected_tool_T_camera[:3, :3].T @ result.tool_T_camera[:3, :3]
    )
    assert translation_error_mm < 0.1
    assert rotation_error_deg < 0.01
    assert result.translation_rms_mm < 0.1
    assert result.rotation_rms_deg < 0.01


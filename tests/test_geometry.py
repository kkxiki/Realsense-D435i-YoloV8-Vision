import numpy as np

from d435i_camera.geometry import (
    deproject_pixel,
    invert_transform,
    make_transform,
    matrix_to_rpy_deg,
    robust_depth,
    rpy_deg_to_matrix,
    transform_point,
)
from d435i_camera.models import Intrinsics


def test_depth_deprojection_and_transform_roundtrip():
    depth = np.zeros((20, 30), dtype=np.float32)
    depth[8:13, 13:18] = 0.75
    value = robust_depth(depth, 15, 10, radius=3, min_m=0.2, max_m=1.5)
    assert value == 0.75
    intrinsics = Intrinsics(30, 20, 100.0, 100.0, 15.0, 10.0)
    point = deproject_pixel(15, 10, value, intrinsics)
    np.testing.assert_allclose(point, [0.0, 0.0, 0.75])
    transform = make_transform(rpy_deg_to_matrix(10, -20, 30), [0.1, -0.2, 0.3])
    recovered = transform_point(invert_transform(transform), transform_point(transform, point))
    np.testing.assert_allclose(recovered, point, atol=1e-10)


def test_rpy_roundtrip_non_singular():
    expected = np.array([12.0, -18.0, 35.0])
    actual = np.array(matrix_to_rpy_deg(rpy_deg_to_matrix(*expected)))
    np.testing.assert_allclose(actual, expected, atol=1e-8)


import numpy as np

from d435i_camera.pose import oriented_box_from_points, pca_object_rotation


def test_pca_orientation_is_right_handed_and_quality_is_finite():
    rng = np.random.default_rng(4)
    points = np.column_stack(
        (
            rng.normal(0.0, 0.008, 1200),
            rng.normal(0.0, 0.055, 1200),
            0.62 + rng.normal(0.0, 0.004, 1200),
        )
    )
    rotation, quality = pca_object_rotation(points)
    np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-8)
    assert np.linalg.det(rotation) > 0.999
    assert 0.0 <= quality <= 1.0
    # Object Z is defined as the longest geometric axis and should be nearly vertical.
    assert abs(rotation[1, 2]) > 0.98


def test_oriented_box_returns_metric_center_and_dimensions():
    rng = np.random.default_rng(8)
    points = rng.uniform((-0.04, -0.08, 0.59), (0.04, 0.08, 0.63), size=(5000, 3))
    center, dimensions = oriented_box_from_points(points, np.eye(3))
    np.testing.assert_allclose(center, (0.0, 0.0, 0.61), atol=0.004)
    np.testing.assert_allclose(dimensions, (0.077, 0.154, 0.038), atol=0.008)

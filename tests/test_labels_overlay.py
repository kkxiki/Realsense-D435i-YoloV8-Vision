import numpy as np

from d435i_camera.labels import chinese_class_name
from d435i_camera.models import Detection2D, FrameBundle, Intrinsics, ObjectObservation, Pose6D
from d435i_camera.overlay import _six_dimensional_lines, draw_3d_box, draw_overlay


def test_bottle_has_required_chinese_name_and_exact_six_dimensional_fields():
    detection = Detection2D(39, "bottle", 0.91, (20, 20, 160, 220))
    pose = Pose6D(
        translation_m=(0.0123, -0.0456, 0.789),
        rotation_matrix=np.eye(3),
        quaternion_xyzw=(0.0, 0.0, 0.0, 1.0),
        rpy_deg=(1.2, -3.4, 5.6),
        frame_id="camera_color_optical_frame",
        method="test",
        quality=0.9,
    )
    observation = ObjectObservation(detection, pose, True)
    lines = _six_dimensional_lines(observation)
    text = " ".join(lines)
    assert chinese_class_name("bottle") == "水瓶"
    assert "物品中文名称：水瓶" in text
    assert "置信度：91.0%" in text
    for key in ("X:", "Y:", "Z:", "Rx:", "Ry:", "Rz:"):
        assert key in text
    assert "quality" not in text and "frame" not in text


def test_unicode_overlay_renders_without_changing_image_shape():
    image = np.zeros((240, 320, 3), dtype=np.uint8)
    frame = FrameBundle(
        color_bgr=image,
        depth_m=np.ones((240, 320), dtype=np.float32),
        intrinsics=Intrinsics(320, 240, 250.0, 250.0, 160.0, 120.0),
        timestamp_s=0.0,
        frame_number=1,
    )
    detection = Detection2D(39, "bottle", 0.91, (60, 40, 180, 210))
    pose = Pose6D(
        translation_m=(0.0, 0.0, 0.8),
        rotation_matrix=np.eye(3),
        quaternion_xyzw=(0.0, 0.0, 0.0, 1.0),
        rpy_deg=(0.0, 0.0, 0.0),
        frame_id=frame.camera_frame,
        method="test",
        quality=1.0,
        dimensions_m=(0.12, 0.20, 0.08),
    )
    observation = ObjectObservation(detection, pose, True)
    box_image = image.copy()
    corners = draw_3d_box(box_image, observation, frame, (80, 220, 120))
    assert len(corners) == 8
    assert np.any(box_image != image)
    output = draw_overlay(frame, [observation], show_panel=False)
    assert output.shape == image.shape
    assert np.any(output != image)
    assert not np.any(output[:8, :8])


def test_common_coco_classes_have_chinese_names():
    assert chinese_class_name("person") == "人"
    assert chinese_class_name("cup") == "杯子"
    assert chinese_class_name("cell phone") == "手机"

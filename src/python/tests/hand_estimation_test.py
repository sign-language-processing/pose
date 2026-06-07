from pathlib import Path

import numpy as np
import numpy.ma as ma
import pytest

from pose_format.numpy import NumPyPoseBody
from pose_format.pose import Pose
from pose_format.pose_header import PoseHeader, PoseHeaderComponent, PoseHeaderDimensions
from pose_format.utils.generic import fake_openpose_pose
from pose_format.utils.hand import estimate_active_hand

DATA_DIR = Path(__file__).parent / "data" / "hand_estimation"

REAL_POSE_CASES = [
    ("84322c38ff23406f222641da30d0a49b.pose", "LEFT"),
    ("7731febd6afbbe90f806a4434c282016.pose", "RIGHT"),
    ("6fb01565da31c5500d1ef2cd2906b06b.pose", "RIGHT"),
    ("7bd6dd48722def3cf84258b96371cb1e.pose", "LEFT"),
    ("0ff215ef4f1f111e217e2cefee0adc77.pose", "RIGHT"),
]

BODY_POINTS = [
    "LEFT_SHOULDER",
    "RIGHT_SHOULDER",
    "LEFT_HIP",
    "RIGHT_HIP",
    "LEFT_ELBOW",
    "RIGHT_ELBOW",
    "LEFT_WRIST",
    "RIGHT_WRIST",
]

HAND_POINTS = [
    "WRIST",
    "INDEX_FINGER_MCP",
    "MIDDLE_FINGER_MCP",
    "RING_FINGER_MCP",
    "PINKY_MCP",
    "INDEX_FINGER_TIP",
    "MIDDLE_FINGER_TIP",
    "PINKY_TIP",
]


def _component(name, points):
    return PoseHeaderComponent(name, points, limbs=[], colors=[], point_format="XYC")


def _pose_template(frames):
    components = [
        _component("POSE_LANDMARKS", BODY_POINTS),
        _component("LEFT_HAND_LANDMARKS", HAND_POINTS),
        _component("RIGHT_HAND_LANDMARKS", HAND_POINTS),
    ]
    header = PoseHeader(version=0.2, dimensions=PoseHeaderDimensions(100, 100), components=components)
    data = np.zeros((frames, 1, header.total_points(), 2), dtype=np.float32)
    confidence = np.zeros((frames, 1, header.total_points()), dtype=np.float32)
    pose = Pose(header, NumPyPoseBody(fps=25.0, data=ma.masked_array(data), confidence=confidence))

    for frame in range(frames):
        _set_body_point(pose, frame, "LEFT_SHOULDER", 40, 30)
        _set_body_point(pose, frame, "RIGHT_SHOULDER", 60, 30)
        _set_body_point(pose, frame, "LEFT_HIP", 42, 70)
        _set_body_point(pose, frame, "RIGHT_HIP", 58, 70)
        _set_body_point(pose, frame, "LEFT_ELBOW", 40, 45)
        _set_body_point(pose, frame, "RIGHT_ELBOW", 60, 45)
        _set_body_point(pose, frame, "LEFT_WRIST", 40, 55)
        _set_body_point(pose, frame, "RIGHT_WRIST", 60, 55)

    return pose


def _set_body_point(pose, frame, point, x, y, conf=1.0):
    index = pose.header.get_point_index("POSE_LANDMARKS", point)
    pose.body.data[frame, 0, index, :2] = (x, y)
    pose.body.confidence[frame, 0, index] = conf


def _set_hand_frame(pose, frame, hand, x, y, conf=1.0):
    for point_no, point in enumerate(HAND_POINTS):
        index = pose.header.get_point_index(f"{hand}_HAND_LANDMARKS", point)
        pose.body.data[frame, 0, index, :2] = (x + point_no * 0.2, y)
        pose.body.confidence[frame, 0, index] = conf


def _make_long_pose(active_hand):
    pose = _pose_template(frames=60)
    other_hand = "LEFT" if active_hand == "RIGHT" else "RIGHT"

    for frame in range(60):
        _set_body_point(pose, frame, f"{active_hand}_WRIST", 60 if active_hand == "RIGHT" else 40, 35)
        _set_body_point(pose, frame, f"{other_hand}_WRIST", 40 if active_hand == "RIGHT" else 60, 58)
        _set_hand_frame(pose, frame, active_hand, 70 if active_hand == "RIGHT" else 30, 35, conf=1.0)
        _set_hand_frame(pose, frame, other_hand, 50, 55, conf=0.1)

    return pose


def test_estimate_active_hand_uses_hand_landmarks_for_long_clips():
    assert estimate_active_hand(_make_long_pose("RIGHT")) == "RIGHT"


def test_estimate_active_hand_is_left_right_symmetric():
    assert estimate_active_hand(_make_long_pose("LEFT")) == "LEFT"


def test_estimate_active_hand_uses_motion_for_short_clips_with_weak_hand_tracking():
    pose = _pose_template(frames=12)

    for frame in range(12):
        _set_body_point(pose, frame, "LEFT_WRIST", 32 + frame * 2, 35)
        _set_body_point(pose, frame, "RIGHT_WRIST", 60, 35)
        _set_hand_frame(pose, frame, "LEFT", 40, 35, conf=0.0)
        _set_hand_frame(pose, frame, "RIGHT", 60, 35, conf=0.0)

    assert estimate_active_hand(pose) == "LEFT"


def test_estimate_active_hand_rejects_non_holistic_poses():
    with pytest.raises(NotImplementedError, match="Unsupported pose header schema openpose"):
        estimate_active_hand(fake_openpose_pose(num_frames=2))


@pytest.mark.parametrize("filename, expected_hand", REAL_POSE_CASES)
def test_estimate_active_hand_on_real_holistic_fixtures(filename, expected_hand):
    pose = Pose.read((DATA_DIR / filename).read_bytes())

    assert estimate_active_hand(pose) == expected_hand


@pytest.mark.parametrize("filename, expected_hand", REAL_POSE_CASES)
def test_estimate_active_hand_flips_after_horizontal_mirror(filename, expected_hand):
    from pose_format.utils.holistic import mirror_horizontal

    opposite_hand = "LEFT" if expected_hand == "RIGHT" else "RIGHT"
    pose = Pose.read((DATA_DIR / filename).read_bytes())

    assert estimate_active_hand(mirror_horizontal(pose)) == opposite_hand

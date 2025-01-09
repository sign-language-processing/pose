from typing import List, get_args
import numpy as np
import pytest
from pose_format import Pose
from pose_format.pose_header import PoseNormalizationInfo
from pose_format.utils.generic import (
    detect_known_pose_format,
    get_component_names,
    get_standard_components_for_known_format,
    SupportedPoseFormat,
    pose_hide_legs,
    pose_shoulders,
    hands_indexes,
    normalize_pose_size,
    pose_normalization_info,
    get_hand_wrist_index,
    get_body_hand_wrist_index,
    correct_wrists,
    hands_components,
)



@pytest.mark.parametrize(
    "fake_poses, expected_type", [(fmt, fmt) for fmt in get_args(SupportedPoseFormat)], indirect=["fake_poses"]
)
def test_detect_format(fake_poses, expected_type):
    for pose in fake_poses:
        detected_format = detect_known_pose_format(get_component_names(pose))
        assert detected_format == expected_type

    with pytest.raises(
        ValueError, match="Could not detect pose format, unknown pose header schema with component names:"
    ):
        detect_known_pose_format(["POSE_WROLD_LANDMARKS"])


@pytest.mark.parametrize(
    "fake_poses, format", [(fmt, fmt) for fmt in get_args(SupportedPoseFormat)], indirect=["fake_poses"]
)
def test_get_component_names(fake_poses: List[Pose], known_pose_format: SupportedPoseFormat):

    standard_components_for_format = get_standard_components_for_known_format(known_pose_format)
    names_for_standard_components_for_format = sorted([c.name for c in standard_components_for_format])
    for pose in fake_poses:

        names_from_poses = sorted(get_component_names(pose))
        names_from_headers = sorted(get_component_names(pose.header))
        names_from_components = sorted(get_component_names(pose.header.components))
        names_from_list = sorted(get_component_names([c.name for c in pose.header.components]))
        assert names_from_poses == names_from_headers
        assert names_from_headers == names_from_components
        assert names_from_list == names_from_components
        assert names_from_components == names_for_standard_components_for_format
        with pytest.raises(ValueError, match="Could not get component_names"):
            get_component_names(pose.body)  # type: ignore


@pytest.mark.parametrize("fake_poses", list(get_args(SupportedPoseFormat)), indirect=["fake_poses"])
def test_pose_hide_legs(fake_poses: List[Pose]):
    for pose in fake_poses:
        orig_nonzeros_count = np.count_nonzero(pose.body.data)

        pose_hide_legs(pose)
        new_nonzeros_count = np.count_nonzero(pose.body.data)

        assert orig_nonzeros_count > new_nonzeros_count


@pytest.mark.parametrize("fake_poses", list(get_args(SupportedPoseFormat)), indirect=["fake_poses"])
def test_pose_shoulders(fake_poses: List[Pose]):
    for pose in fake_poses:
        shoulders = pose_shoulders(pose.header)
        assert len(shoulders) == 2
        assert "RIGHT" in shoulders[0][1] or shoulders[0][1][0] == "R"
        assert "LEFT" in shoulders[1][1] or shoulders[1][1][0] == "L"


@pytest.mark.parametrize("fake_poses", list(get_args(SupportedPoseFormat)), indirect=["fake_poses"])
def test_hands_indexes(fake_poses: List[Pose]):
    for pose in fake_poses:
        indices = hands_indexes(pose.header)
        assert len(indices) > 0


@pytest.mark.parametrize("fake_poses", list(get_args(SupportedPoseFormat)), indirect=["fake_poses"])
def test_normalize_pose_size(fake_poses: List[Pose]):
    for pose in fake_poses:
        normalize_pose_size(pose)
    # TODO: more tests, compare with test data


@pytest.mark.parametrize("fake_poses", list(get_args(SupportedPoseFormat)), indirect=["fake_poses"])
def test_pose_normalization_info(fake_poses: List[Pose]):
    for pose in fake_poses:
        info = pose_normalization_info(pose.header)
        assert isinstance(info, PoseNormalizationInfo)
        assert info.p1 is not None
        assert info.p2 is not None
        assert info.p3 is None
        # TODO: more tests, compare with test data


@pytest.mark.parametrize("fake_poses", list(get_args(SupportedPoseFormat)), indirect=["fake_poses"])
def test_get_hand_wrist_index(fake_poses: List[Pose]):
    for pose in fake_poses:
        for hand in ["LEFT", "RIGHT"]:
            index = get_hand_wrist_index(pose, hand)

            # TODO: what are the expected values?


@pytest.mark.parametrize("fake_poses", list(get_args(SupportedPoseFormat)), indirect=["fake_poses"])
def test_get_body_hand_wrist_index(fake_poses: List[Pose]):
    for pose in fake_poses:
        for hand in ["LEFT", "RIGHT"]:
            index = get_body_hand_wrist_index(pose, hand)
            # TODO: what are the expected values?


@pytest.mark.parametrize("fake_poses", list(get_args(SupportedPoseFormat)), indirect=["fake_poses"])
def test_correct_wrists(fake_poses: List[Pose]):
    for pose in fake_poses:
        corrected_pose = correct_wrists(pose)
        assert np.array_equal(corrected_pose.body.data, pose.body.data) is False
        assert corrected_pose != pose


@pytest.mark.parametrize("fake_poses", list(get_args(SupportedPoseFormat)), indirect=["fake_poses"])
def test_hands_components(fake_poses: List[Pose]):
    for pose in fake_poses:
        hands_components_returned = hands_components(pose.header)
        assert "LEFT" in hands_components_returned[0][0].upper()
        assert "RIGHT" in hands_components_returned[0][1].upper()

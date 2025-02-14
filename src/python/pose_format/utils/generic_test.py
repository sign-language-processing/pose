from collections import defaultdict
from typing import List, get_args
import numpy as np
import pytest
from pose_format.pose import Pose
from pose_format.pose_header import PoseNormalizationInfo
from pose_format.utils.generic import (
    detect_known_pose_format,
    get_component_names,
    get_standard_components_for_known_format,
    KnownPoseFormat,
    pose_hide_legs,
    pose_shoulders,
    hands_indexes,
    normalize_pose_size,
    pose_normalization_info,
    get_hand_wrist_index,
    get_body_hand_wrist_index,
    correct_wrists,
    hands_components,
    fake_pose,
)

TEST_POSE_FORMATS = list(get_args(KnownPoseFormat))
TEST_POSE_FORMATS_WITH_UNKNOWN = list(get_args(KnownPoseFormat)) +["unknown"]

@pytest.mark.parametrize(
    "fake_poses, expected_type", [(fmt, fmt) for fmt in TEST_POSE_FORMATS_WITH_UNKNOWN], indirect=["fake_poses"]
)
def test_detect_format(fake_poses, expected_type):
    known_formats = get_args(KnownPoseFormat)
    for pose in fake_poses:
        if expected_type in known_formats:
            detected_format = detect_known_pose_format(pose)
            assert detected_format == expected_type
        else:
            with pytest.raises(
                ValueError, match="Could not detect pose format, unknown pose header schema with component names:"
            ):
                detect_known_pose_format(pose)


@pytest.mark.parametrize(
    "fake_poses, known_pose_format", [(fmt, fmt) for fmt in TEST_POSE_FORMATS], indirect=["fake_poses"]
)
def test_get_component_names(fake_poses: List[Pose], known_pose_format: KnownPoseFormat):

    standard_components_for_format = get_standard_components_for_known_format(known_pose_format)
    names_for_standard_components_for_format = sorted([c.name for c in standard_components_for_format])
    for pose in fake_poses:

        names_from_poses = sorted(get_component_names(pose))
        names_from_headers = sorted(get_component_names(pose.header))
        assert names_from_poses == names_from_headers
        assert names_for_standard_components_for_format == names_from_headers
        with pytest.raises(ValueError, match="Could not get component_names"):
            get_component_names(pose.body)  # type: ignore


@pytest.mark.parametrize("fake_poses", list(get_args(KnownPoseFormat)), indirect=["fake_poses"])
def test_pose_hide_legs(fake_poses: List[Pose]):
    for pose in fake_poses:
        pose_copy = pose.copy()
        orig_nonzeros_count = np.count_nonzero(pose.body.data)

        detected_format = detect_known_pose_format(pose)
        if detected_format == "openpose_135":
            with pytest.raises(NotImplementedError, match="Unsupported pose header schema"):
                pose = pose_hide_legs(pose)
        else:
            pose = pose_hide_legs(pose)
            new_nonzeros_count = np.count_nonzero(pose.body.data)

            assert orig_nonzeros_count > new_nonzeros_count
            assert len(pose_copy.header.components) == len(pose.header.components)
            for c_orig, c_copy in zip(pose.header.components, pose_copy.header.components):
                assert len(c_orig.points) == len(c_copy.points)            
            # what if we remove the legs before hiding them first? It should not crash.
            pose = pose_hide_legs(pose, remove=True)
            pose = pose_hide_legs(pose, remove=False)


@pytest.mark.parametrize("fake_poses", TEST_POSE_FORMATS, indirect=["fake_poses"])
def test_pose_shoulders(fake_poses: List[Pose]):
    for pose in fake_poses:
        shoulders = pose_shoulders(pose.header)
        assert len(shoulders) == 2
        assert "RIGHT" in shoulders[0][1] or shoulders[0][1][0] == "R"
        assert "LEFT" in shoulders[1][1] or shoulders[1][1][0] == "L"


@pytest.mark.parametrize("fake_poses", TEST_POSE_FORMATS, indirect=["fake_poses"])
def test_hands_indexes(fake_poses: List[Pose]):
    for pose in fake_poses:
        detected_format = detect_known_pose_format(pose)
        if detected_format == "openpose_135":
            with pytest.raises(NotImplementedError, match="Unsupported pose header schema"):
                indices = hands_indexes(pose.header)
        else:
            indices = hands_indexes(pose.header)
            assert len(indices) > 0


@pytest.mark.parametrize("fake_poses", TEST_POSE_FORMATS, indirect=["fake_poses"])
def test_normalize_pose_size(fake_poses: List[Pose]):
    for pose in fake_poses:
        normalize_pose_size(pose)
    # TODO: more tests, compare with test data


@pytest.mark.parametrize("fake_poses", TEST_POSE_FORMATS, indirect=["fake_poses"])
def test_pose_normalization_info(fake_poses: List[Pose]):
    for pose in fake_poses:
        info = pose_normalization_info(pose.header)
        assert isinstance(info, PoseNormalizationInfo)
        assert info.p1 is not None
        assert info.p2 is not None
        assert info.p3 is None
        # TODO: more tests, compare with test data


@pytest.mark.parametrize("fake_poses", TEST_POSE_FORMATS, indirect=["fake_poses"])
def test_get_hand_wrist_index(fake_poses: List[Pose]):
    for pose in fake_poses:
        detected_format = detect_known_pose_format(pose)
        for hand in ["LEFT", "RIGHT"]:
            if detected_format == "openpose_135":
                with pytest.raises(NotImplementedError, match="Unsupported pose header schema"):
                    _ = get_hand_wrist_index(pose, hand)
            else:
                _ = get_hand_wrist_index(pose, hand)

                # TODO: what are the expected values?


@pytest.mark.parametrize("fake_poses", TEST_POSE_FORMATS, indirect=["fake_poses"])
def test_get_body_hand_wrist_index(fake_poses: List[Pose]):
    for pose in fake_poses:
        for hand in ["LEFT", "RIGHT"]:
            detected_format = detect_known_pose_format(pose)
            if detected_format == "openpose_135":
                with pytest.raises(NotImplementedError, match="Unsupported pose header schema"):
                    _ = get_body_hand_wrist_index(pose, hand)
                    # TODO: what are the expected values?
            else:
                _ = get_body_hand_wrist_index(pose, hand)



@pytest.mark.parametrize("fake_poses", TEST_POSE_FORMATS, indirect=["fake_poses"])
def test_correct_wrists(fake_poses: List[Pose]):
    for pose in fake_poses:
        detected_format = detect_known_pose_format(pose)
        if detected_format == "openpose_135":
            with pytest.raises(NotImplementedError, match="Unsupported pose header schema"):
                corrected_pose = correct_wrists(pose)

        else:
            corrected_pose = correct_wrists(pose)
            assert corrected_pose != pose
            assert np.array_equal(corrected_pose.body.data, pose.body.data) is False

@pytest.mark.parametrize("fake_poses", ["holistic"], indirect=["fake_poses"])
def test_remove_one_point_and_one_component(fake_poses: List[Pose]):
    component_to_drop = "POSE_WORLD_LANDMARKS"
    point_to_drop = "LEFT_KNEE"
    for pose in fake_poses:
        original_component_names = []
        original_points_dict = defaultdict(list)
        for component in pose.header.components:
            original_component_names.append(component.name)

            for point in component.points:
                original_points_dict[component.name].append(point)

        assert component_to_drop in original_component_names
        assert point_to_drop in original_points_dict["POSE_LANDMARKS"]
        reduced_pose = pose.remove_components(component_to_drop, {"POSE_LANDMARKS": [point_to_drop]})
        new_component_names, new_points_dict = [], defaultdict(list)
        new_component_names = []
        new_points_dict = defaultdict(list)
        for component in reduced_pose.header.components:
            new_component_names.append(component.name)

            for point in component.points:
                new_points_dict[component.name].append(point)


        assert component_to_drop not in new_component_names
        assert point_to_drop not in new_points_dict["POSE_LANDMARKS"]

@pytest.mark.parametrize("fake_poses", TEST_POSE_FORMATS, indirect=["fake_poses"])
def test_pose_remove_legs(fake_poses: List[Pose]):
    for pose in fake_poses:
        known_pose_format = detect_known_pose_format(pose)
        if known_pose_format == "holistic":
            points_that_should_be_removed = ["LEFT_KNEE", "LEFT_HEEL", "LEFT_FOOT", "LEFT_TOE", "LEFT_FOOT_INDEX",
                                            "RIGHT_KNEE", "RIGHT_HEEL", "RIGHT_FOOT", "RIGHT_TOE", "RIGHT_FOOT_INDEX",]
            c_names = [c.name for c in pose.header.components]
            assert "POSE_LANDMARKS" in c_names
            pose_landmarks_index = c_names.index("POSE_LANDMARKS")
            assert "LEFT_KNEE" in pose.header.components[pose_landmarks_index].points


            pose_with_legs_removed = pose_hide_legs(pose, remove=True)
            assert pose_with_legs_removed != pose
            new_c_names = [c.name for c in pose_with_legs_removed.header.components]
            assert "POSE_LANDMARKS" in new_c_names

            for component in pose_with_legs_removed.header.components:
                point_names = [point.upper() for point in component.points]
                for point_name in point_names:
                    for point_that_should_be_hidden in points_that_should_be_removed:
                        assert point_that_should_be_hidden not in point_name, f"{component.name}: {point_names}"

        elif known_pose_format == "openpose":
            c_names = [c.name for c in pose.header.components]
            points_that_should_be_removed = ['LHip', 'RHip', 'MidHip',
                                             'LKnee', 'RKnee', 
                                             'LAnkle', 'RAnkle',  
                                             'LBigToe', 'RBigToe', 
                                             'LSmallToe', 'RSmallToe',
                                             'LHeel', 'RHeel']
            component_index = c_names.index("pose_keypoints_2d")
            pose_with_legs_removed = pose_hide_legs(pose, remove=True)

            for point_name in points_that_should_be_removed:
                assert point_name not in pose_with_legs_removed.header.components[component_index].points, f"{pose_with_legs_removed.header.components[component_index].name},{pose_with_legs_removed.header.components[component_index].points}"
                assert point_name in pose.header.components[component_index].points
        else:
            with pytest.raises(NotImplementedError, match="Unsupported pose header schema"):
                pose = pose_hide_legs(pose, remove=True)



@pytest.mark.parametrize("fake_poses", TEST_POSE_FORMATS, indirect=["fake_poses"])
def test_hands_components(fake_poses: List[Pose]):
    for pose in fake_poses:
        detected_format = detect_known_pose_format(pose)
        if detected_format == "openpose_135":
            with pytest.raises(NotImplementedError, match="Unsupported pose header schema"):
                hands_components_returned = hands_components(pose.header)
        else:
            hands_components_returned = hands_components(pose.header)
            assert "LEFT" in hands_components_returned[0][0].upper()
            assert "RIGHT" in hands_components_returned[0][1].upper()


@pytest.mark.parametrize("known_pose_format", TEST_POSE_FORMATS)
def test_fake_pose(known_pose_format: KnownPoseFormat):

    for frame_count in [1, 10, 100]:
        for fps in [1, 15, 25, 100]:
            standard_components = get_standard_components_for_known_format(known_pose_format)

            pose = fake_pose(frame_count, fps=fps, components=standard_components)
            point_formats = [c.format for c in pose.header.components]
            data_dimension_expected = 0

            # they should all be consistent
            for point_format in point_formats:
                # something like "XYC" or "XYZC"
                assert point_format == point_formats[0]

            data_dimension_expected = len(point_formats[0]) - 1

            detected_format = detect_known_pose_format(pose)

            if detected_format == 'holistic':
                assert point_formats[0] == "XYZC"
            elif detected_format == 'openpose':
                assert point_formats[0] == "XYC"
            elif detected_format == 'openpose_135':
                assert point_formats[0] == "XYC"

            assert detected_format == known_pose_format
            assert pose.body.fps == fps
            assert pose.body.data.shape == (frame_count, 1, pose.header.total_points(), data_dimension_expected)
            assert pose.body.data.shape[0] == frame_count
            assert pose.header.num_dims() == pose.body.data.shape[-1]

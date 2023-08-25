import json
import os
from contextlib import contextmanager
from tempfile import TemporaryDirectory
from typing import List, Optional
from unittest import TestCase

import numpy as np

from pose_format.pose import Pose
from pose_format.utils.openpose import (OPENPOSE_FRAME_PATTERN,
                                        OpenPose_Components, OpenPoseFrames,
                                        get_frame_id, load_openpose,
                                        load_openpose_directory)

OPENPOSE_TOTAL_KEYPOINTS = 137
OPENPOSE_COMPONENTS_USED = [
    "pose_keypoints_2d", "face_keypoints_2d", "hand_left_keypoints_2d", "hand_right_keypoints_2d"
]
OPENPOSE_COMPONENTS_UNUSED = [
    "pose_keypoints_3d", "face_keypoints_3d", "hand_left_keypoints_3d", "hand_right_keypoints_3d"
]
OPENPOSE_NUM_POINTS_PER_COMPONENT = {
    "pose_keypoints_2d": 25,
    "face_keypoints_2d": 70,
    "hand_left_keypoints_2d": 21,
    "hand_right_keypoints_2d": 21
}


def _generate_random_frames_dict(num_frames: int = 10,
                                 num_people: int = 1,
                                 num_dimensions: int = 2,
                                 num_missing_frames: int = 0) -> OpenPoseFrames:
    """
    Generates dictionary of random frames for testing 
    
    Parameters
    ----------
    num_frames : int, optional
        Number of frames to generate. Default is 10
    num_people : int, optional
        Number of people per frame, by default 1.
    num_dimensions : int, optional
        Number of dimensions for the keypoints; default 2.
    num_missing_frames : int, optional
        Number of frames to randomly remove, default 0.

    Returns
    -------
    OpenPoseFrames
        dictionary of generated frames.

    Notes
    -----
    function used internally for testing purposes.
    """
    frames = {}  # type: OpenPoseFrames

    for frame_id in range(num_frames):
        frame_dict = {"version": 1.3}
        people = []
        for people_id in range(num_people):
            people_dict = {"person_id": people_id}
            for component in OpenPose_Components:
                num_keypoints = len(component.points)

                # for each keypoint: one value for each dimension, plus confidence
                num_pose_values = (num_dimensions + 1) * num_keypoints
                pose_data = np.random.random_sample((num_pose_values,)).tolist()
                people_dict[component.name] = pose_data

            people.append(people_dict)
        frame_dict["people"] = people
        frames[frame_id] = frame_dict

    if num_missing_frames > 0:
        # the highest frame id is never selected, this is intentional
        random_ids = np.random.choice(range(num_frames - 1), num_missing_frames, replace=False)
        for random_id in random_ids:
            del frames[random_id]

    return frames


def _build_openpose_filepath(frame_id: int,
                             directory: str,
                             file_prefix: Optional[str] = None,
                             max_num_id_digits: int = 12) -> str:
    if file_prefix is None:
        file_prefix = ""

    frame_id_as_string = '%0*d' % (max_num_id_digits, frame_id)
    filename = "_".join([file_prefix, frame_id_as_string, "keypoints.json"])

    return os.path.join(directory, filename)


def _generate_openpose_frame_file(filepath: str,
                                  num_people: int = 1,
                                  num_dimensions: int = 2,
                                  add_empty_keys: Optional[List[str]] = None) -> None:

    with open(filepath, "w") as handle:
        frames = _generate_random_frames_dict(1, num_people, num_dimensions, 0)
        frame = frames[0]

        if add_empty_keys is not None:
            for key in add_empty_keys:
                frame[key] = []

        json.dump(frame, handle)


@contextmanager
def _create_tmp_openpose_directory(num_files: int = 10,
                                   num_people: int = 1,
                                   num_dimensions: int = 2,
                                   num_missing_frames: int = 0,
                                   file_prefix: Optional[str] = None,
                                   max_num_id_digits: int = 12,
                                   add_empty_keys: Optional[List[str]] = None) -> str:
    with TemporaryDirectory(prefix="test_openpose") as work_dir:

        frame_ids = [frame_id for frame_id in range(num_files)]

        if num_missing_frames > 0:
            # the highest frame id is never selected, this is intentional
            random_ids = np.random.choice(range(num_files - 1), num_missing_frames, replace=False)
            for random_id in random_ids:
                frame_ids.remove(random_id)

        for frame_id in frame_ids:
            filepath = _build_openpose_filepath(frame_id, work_dir, file_prefix, max_num_id_digits)
            _generate_openpose_frame_file(filepath, num_people, num_dimensions, add_empty_keys)

        yield work_dir


class TestOpenposeComponents(TestCase):
    """ Test cases for OpenPose components"""

    def test_openpose_components_total_points(self):
        """Tests if total points in OpenPose components match expected value"""

        actual_total_points = 0

        for component in OpenPose_Components:
            num_keypoints = len(component.points)
            actual_total_points += num_keypoints

        self.assertEqual(actual_total_points, OPENPOSE_TOTAL_KEYPOINTS)

    def test_openpose_components_names(self):
        """Tests the names in OpenPose components"""

        expected_names = OPENPOSE_COMPONENTS_USED
        actual_names = [c.name for c in OpenPose_Components]

        self.assertEqual(actual_names, expected_names)

    def test_openpose_num_points_per_component_pose(self):
        """Tests number of points for the 'pose_keypoints_2d' component match expected"""

        expected_value = OPENPOSE_NUM_POINTS_PER_COMPONENT["pose_keypoints_2d"]
        actual_value = len(OpenPose_Components[0].points)

        self.assertEqual(actual_value, expected_value)

    def test_openpose_num_points_per_component_face(self):
        """Tests number of points for the 'face_keypoints_2d' component matches expected"""

        expected_value = OPENPOSE_NUM_POINTS_PER_COMPONENT["face_keypoints_2d"]
        actual_value = len(OpenPose_Components[1].points)

        self.assertEqual(actual_value, expected_value)

    def test_openpose_num_points_per_component_hand_left(self):
        """Tests if number of points for the 'hand_left_keypoints_2d' component matches expected"""
        expected_value = OPENPOSE_NUM_POINTS_PER_COMPONENT["hand_left_keypoints_2d"]
        actual_value = len(OpenPose_Components[2].points)

        self.assertEqual(actual_value, expected_value)

    def test_openpose_num_points_per_component_hand_right(self):
        """Tests if the number of points for the 'hand_right_keypoints_2d' component matches expected value"""
        expected_value = OPENPOSE_NUM_POINTS_PER_COMPONENT["hand_right_keypoints_2d"]
        actual_value = len(OpenPose_Components[3].points)

        self.assertEqual(actual_value, expected_value)


class TestLoadOpenpose(TestCase):
    """Test cases for loading OpenPose data"""

    def test_load_openpose_returns_pose_object(self):
        """Test if loaded data is type Pose"""
        frames = _generate_random_frames_dict()
        pose = load_openpose(frames)

        self.assertTrue(isinstance(pose, Pose))

    def test_load_openpose_data_shape(self):
        """Tests if loaded data has expected shape"""
        num_frames = 10
        num_people = 1
        num_dimensions = 2
        num_missing_frames = 0

        expected_shape = (num_frames, num_people, OPENPOSE_TOTAL_KEYPOINTS, num_dimensions)

        frames = _generate_random_frames_dict(num_frames, num_people, num_dimensions, num_missing_frames)
        pose = load_openpose(frames)

        actual_shape = pose.body.data.shape

        self.assertEqual(actual_shape, expected_shape)

    def test_load_openpose_data_shape_missing_frames(self):
        """Tests if data shape from ``load_openpose`` with missing frames."""
        num_frames = 10
        num_people = 1
        num_dimensions = 2
        num_missing_frames = 3

        expected_shape = (num_frames, num_people, OPENPOSE_TOTAL_KEYPOINTS, num_dimensions)

        frames = _generate_random_frames_dict(num_frames, num_people, num_dimensions, num_missing_frames)
        pose = load_openpose(frames)

        actual_shape = pose.body.data.shape

        self.assertEqual(actual_shape, expected_shape)

    def test_load_openpose_extracts_correct_number_of_people(self):
        """Tests if correct number of people are extracted"""
        num_frames = 10
        num_people = 2
        num_dimensions = 2
        num_missing_frames = 0

        expected_shape = (num_frames, num_people, OPENPOSE_TOTAL_KEYPOINTS, num_dimensions)

        frames = _generate_random_frames_dict(num_frames, num_people, num_dimensions, num_missing_frames)
        pose = load_openpose(frames)

        actual_shape = pose.body.data.shape

        self.assertEqual(actual_shape, expected_shape)

    def test_load_openpose_fixed_num_frames(self):
        """Tests ``load_openpose``'s output shape with a fixed number of frames"""
        num_frames_openpose = 8
        num_frames_video = 10
        num_people = 1
        num_dimensions = 2
        num_missing_frames = 0

        expected_shape = (num_frames_video, 1, OPENPOSE_TOTAL_KEYPOINTS, num_dimensions)

        frames = _generate_random_frames_dict(num_frames_openpose, num_people, num_dimensions, num_missing_frames)
        pose = load_openpose(frames, num_frames=num_frames_video)

        actual_shape = pose.body.data.shape

        self.assertEqual(actual_shape, expected_shape)

    def test_load_openpose_fixed_num_frames_and_missing_frames(self):
        """Tests ``load_openpose``'s output shape with fixed frames & missing frames"""
        num_frames_openpose = 14
        num_frames_video = 17
        num_people = 1
        num_dimensions = 2
        num_missing_frames = 3

        expected_shape = (num_frames_video, 1, OPENPOSE_TOTAL_KEYPOINTS, num_dimensions)

        frames = _generate_random_frames_dict(num_frames_openpose, num_people, num_dimensions, num_missing_frames)
        pose = load_openpose(frames, num_frames=num_frames_video)

        actual_shape = pose.body.data.shape

        self.assertEqual(actual_shape, expected_shape)


class TestLoadOpenposeDirectory(TestCase):
    """Testcases for loading OpenPose data from a directory"""

    def test_get_frame_id_zero(self):
        """Test if frame ID extraction works correctly for a filename with ID 0."""
        filename = "CAM2_000000000000_keypoints.json"
        expected_output = 0

        actual_output = get_frame_id(filename, pattern=OPENPOSE_FRAME_PATTERN)
        self.assertEqual(actual_output, expected_output)

    def test_get_frame_id_nonzero(self):
        """Tests if frame ID works correctly for a filename with non-zero ID"""
        filename = "CAM2_000000000007_keypoints.json"
        expected_output = 7

        actual_output = get_frame_id(filename, pattern=OPENPOSE_FRAME_PATTERN)
        self.assertEqual(actual_output, expected_output)

    def test_get_frame_id_several_digits(self):
        """Tests frame ID extraction from a filename with several digits in ID"""
        filename = "CAM2_000000000457_keypoints.json"
        expected_output = 457

        actual_output = get_frame_id(filename, pattern=OPENPOSE_FRAME_PATTERN)
        self.assertEqual(actual_output, expected_output)

    def test_get_frame_id_no_prefix(self):
        """Tests frame ID extraction from a filename with no prefix."""
        filename = "000000000013_keypoints.json"
        expected_output = 13

        actual_output = get_frame_id(filename, pattern=OPENPOSE_FRAME_PATTERN)
        self.assertEqual(actual_output, expected_output)

    def test_load_openpose_directory_returns_pose_object(self):
        """Tests if loading from a directory returns an instance of Pose"""
        with _create_tmp_openpose_directory(num_files=10,
                                            num_people=1,
                                            num_dimensions=2,
                                            file_prefix="MOVIE",
                                            max_num_id_digits=12,
                                            add_empty_keys=None) as work_dir:
            pose = load_openpose_directory(work_dir)
            self.assertTrue(isinstance(pose, Pose))

    def test_load_openpose_directory_data_shape(self):
        """Tests shape of data returned by ``load_openpose_directory``"""
        num_frames = 10
        num_people = 1
        num_dimensions = 2
        num_missing_frames = 0

        expected_shape = (num_frames, num_people, OPENPOSE_TOTAL_KEYPOINTS, num_dimensions)

        with _create_tmp_openpose_directory(num_files=num_frames,
                                            num_people=num_people,
                                            num_dimensions=num_dimensions,
                                            num_missing_frames=num_missing_frames,
                                            file_prefix="MOVIE",
                                            max_num_id_digits=12,
                                            add_empty_keys=None) as work_dir:
            pose = load_openpose_directory(work_dir)

            actual_shape = pose.body.data.shape

            self.assertEqual(actual_shape, expected_shape)

    def test_load_openpose_directory_data_shape_missing_frames(self):
        """Test the data shape from load_openpose_directory with missing frames."""
        num_frames = 10
        num_people = 1
        num_dimensions = 2
        num_missing_frames = 3

        expected_shape = (num_frames, num_people, OPENPOSE_TOTAL_KEYPOINTS, num_dimensions)

        with _create_tmp_openpose_directory(num_files=num_frames,
                                            num_people=num_people,
                                            num_dimensions=num_dimensions,
                                            num_missing_frames=num_missing_frames,
                                            file_prefix="MOVIE",
                                            max_num_id_digits=12,
                                            add_empty_keys=None) as work_dir:
            pose = load_openpose_directory(work_dir)

            actual_shape = pose.body.data.shape

            self.assertEqual(actual_shape, expected_shape)

    def test_load_openpose_directory_extracts_correct_number_of_people(self):
        """Test if the correct number of people are extracted by load_openpose_directory."""
        num_frames = 10
        num_people = 2
        num_dimensions = 2
        num_missing_frames = 3

        expected_shape = (num_frames, num_people, OPENPOSE_TOTAL_KEYPOINTS, num_dimensions)

        with _create_tmp_openpose_directory(num_files=num_frames,
                                            num_people=num_people,
                                            num_dimensions=num_dimensions,
                                            num_missing_frames=num_missing_frames,
                                            file_prefix="MOVIE",
                                            max_num_id_digits=12,
                                            add_empty_keys=None) as work_dir:
            pose = load_openpose_directory(work_dir)

            actual_shape = pose.body.data.shape

            self.assertEqual(actual_shape, expected_shape)

    def test_load_openpose_directory_data_shape_add_empty_keys(self):
        """Test the data shape from load_openpose_directory when adding empty keys."""
        num_frames = 10
        num_people = 1
        num_dimensions = 2
        num_missing_frames = 0

        expected_shape = (num_frames, num_people, OPENPOSE_TOTAL_KEYPOINTS, num_dimensions)

        with _create_tmp_openpose_directory(num_files=num_frames,
                                            num_people=num_people,
                                            num_dimensions=num_dimensions,
                                            num_missing_frames=num_missing_frames,
                                            file_prefix="MOVIE",
                                            max_num_id_digits=12,
                                            add_empty_keys=OPENPOSE_COMPONENTS_UNUSED) as work_dir:
            pose = load_openpose_directory(work_dir)

            actual_shape = pose.body.data.shape

            self.assertEqual(actual_shape, expected_shape)

    def test_load_openpose_directory_data_shape_no_file_prefix(self):
        """Test the data shape from load_openpose_directory without file prefix."""
        num_frames = 10
        num_people = 1
        num_dimensions = 2
        num_missing_frames = 0

        expected_shape = (num_frames, num_people, OPENPOSE_TOTAL_KEYPOINTS, num_dimensions)

        with _create_tmp_openpose_directory(num_files=num_frames,
                                            num_people=num_people,
                                            num_dimensions=num_dimensions,
                                            num_missing_frames=num_missing_frames,
                                            file_prefix=None,
                                            max_num_id_digits=12,
                                            add_empty_keys=OPENPOSE_COMPONENTS_UNUSED) as work_dir:
            pose = load_openpose_directory(work_dir)

            actual_shape = pose.body.data.shape

            self.assertEqual(actual_shape, expected_shape)

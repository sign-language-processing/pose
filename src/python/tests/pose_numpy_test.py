from unittest import TestCase

import numpy as np
import numpy.ma as ma


from .pose_test import _get_random_pose_object_with_numpy_posebody


class TestPoseNumpyPoseBody(TestCase):
    """
    Testcases for Pose objects containing NumPy PoseBody data.
    """

    def test_pose_numpy_generated_with_correct_shape(self):
        pose = _get_random_pose_object_with_numpy_posebody(num_keypoints=5, frames_min=3)

        # does the header match the body?
        expected_keypoints_count_from_header = sum(len(c.points) for c in pose.header.components)
        self.assertEqual(expected_keypoints_count_from_header, pose.body.data.shape[-2])

    def test_pose_numpy_posebody_normalize_preserves_shape(self):
        """
        Tests if the normalization of Pose object with NumPy PoseBody preserves array shape.
        """
        pose = _get_random_pose_object_with_numpy_posebody(num_keypoints=5, frames_min=3)

        shape_before = pose.body.data.shape

        # in the mock data header components are named 0, 1, and so on
        # individual points are named 0_a, 0_b, and so on
        pose.normalize(pose.header.normalization_info(p1=("0", "0_a"), p2=("0", "0_b")))

        shape_after = pose.body.data.shape

        self.assertEqual(shape_before, shape_after, "Normalize did not preserve tensor shape.")

    def test_pose_numpy_posebody_frame_dropout_normal_eager_mode_num_frames_not_zero(self):
        """
        Tests if frame dropout using normal distribution in Pose object with NumPy PoseBody preserves frame count.
        """
        pose = _get_random_pose_object_with_numpy_posebody(num_keypoints=5, frames_min=3)

        pose_after_dropout, _ = pose.frame_dropout_normal()

        num_frames = pose_after_dropout.body.data.shape[0]

        self.assertNotEqual(num_frames, 0, "Number of frames after dropout can never be 0.")

    def test_pose_numpy_posebody_frame_dropout_uniform_eager_mode_num_frames_not_zero(self):
        """
        Tests if frame dropout using uniform distribution in Pose object with NumPy PoseBody preserves frame count.
        """
        pose = _get_random_pose_object_with_numpy_posebody(num_keypoints=5, frames_min=3)

        pose_after_dropout, _ = pose.frame_dropout_uniform()

        num_frames = pose_after_dropout.body.data.shape[0]

        self.assertNotEqual(num_frames, 0, "Number of frames after dropout can never be 0.")

    def test_pose_numpy_posebody_copy_creates_deepcopy(self):
        pose = _get_random_pose_object_with_numpy_posebody(num_keypoints=5, frames_min=3)

        pose_copy = pose.copy()
        self.assertNotEqual(pose, pose_copy, "Copy of pose should not be 'equal' to original")

        self.assertTrue(np.array_equal(pose.body.data, pose_copy.body.data), "Copy's data should match original")

        pose.body.data = ma.zeros(pose.body.data.shape)

        self.assertFalse(np.array_equal(pose.body.data, pose_copy.body.data),
                         "Copy's data should not match original after original is replaced")

        pose = pose_copy.copy()

        self.assertTrue(np.array_equal(pose.body.data, pose_copy.body.data), "Copy's data should match original again")

        pose_copy.body.data[:] = 3.14

        self.assertFalse(np.array_equal(pose.body.data, pose_copy.body.data),
                         "Copy's data should not match original after copy is modified")

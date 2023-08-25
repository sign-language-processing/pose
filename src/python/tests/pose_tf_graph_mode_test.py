from unittest import TestCase

import tensorflow as tf

from tests.pose_test import _get_random_pose_object_with_tf_posebody


class TestPose(TestCase):
    """
    Testcases for the Pose object with TensorFlow operations in graph mode.
    """

    def test_pose_tf_posebody_normalize_graph_mode_does_not_fail(self):
        """
        Tests if the normalization of Pose object with TensorFlow PoseBody in graph mode does not fail.
        """
        with tf.Graph().as_default():

            assert tf.executing_eagerly() is False

            pose = _get_random_pose_object_with_tf_posebody(num_keypoints=5)

            # in the mock data header components are named 0, 1, and so on
            # individual points are named 0_a, 0_b, and so on
            pose.normalize(pose.header.normalization_info(p1=("0", "0_a"), p2=("0", "0_b")))

    def test_pose_tf_posebody_frame_dropout_uniform_graph_mode_does_not_fail(self):
        """
        Tests if frame dropout using uniform distribution in Pose object with TensorFlow PoseBody in graph mode does not fail.
        """
        with tf.Graph().as_default():

            assert tf.executing_eagerly() is False

            pose = _get_random_pose_object_with_tf_posebody(num_keypoints=5)
            pose.frame_dropout_uniform()

    def test_pose_tf_posebody_frame_dropout_normal_graph_mode_does_not_fail(self):
        """
        tests if frame dropout using normal distribution in Pose object with TensorFlow PoseBody in graph mode does not fail.
        """
        with tf.Graph().as_default():

            assert tf.executing_eagerly() is False

            pose = _get_random_pose_object_with_tf_posebody(num_keypoints=5)
            pose.frame_dropout_normal()

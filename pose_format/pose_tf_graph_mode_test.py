import tensorflow as tf

from unittest import TestCase

from pose_format.pose_test import _get_random_pose_object_with_tf_posebody


class TestPose(TestCase):

    def test_pose_tf_posebody_normalize_graph_mode_does_not_fail(self):

        with tf.Graph().as_default():

            assert tf.executing_eagerly() is False

            pose = _get_random_pose_object_with_tf_posebody(num_keypoints=5)

            # in the mock data header components are named 0, 1, and so on
            # individual points are named 0_a, 0_b, and so on
            pose.normalize(pose.header.normalization_info(
                p1=("0", "0_a"),
                p2=("0", "0_b")
            ))

    def test_pose_tf_posebody_frame_dropout_uniform_graph_mode_does_not_fail(self):

        with tf.Graph().as_default():

            assert tf.executing_eagerly() is False

            pose = _get_random_pose_object_with_tf_posebody(num_keypoints=5)
            pose.frame_dropout_uniform()

    def test_pose_tf_posebody_frame_dropout_normal_graph_mode_does_not_fail(self):

        with tf.Graph().as_default():

            assert tf.executing_eagerly() is False

            pose = _get_random_pose_object_with_tf_posebody(num_keypoints=5)
            pose.frame_dropout_normal()

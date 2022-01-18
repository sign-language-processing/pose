import tensorflow as tf

from unittest import TestCase

from pose_format.tensorflow.pose_body import TensorflowPoseBody
from pose_format.tensorflow.masked.tensor import MaskedTensor


class TestTensorflowPoseBody(TestCase):

    def test_tf_pose_body_zero_filled_fills_in_zeros(self):

        tensor = tf.ones(7)
        confidence = tf.ones(7)
        mask = tf.constant([1, 1, 1, 0, 0, 1, 1])

        masked_tensor = MaskedTensor(tensor=tensor, mask=mask)
        body = TensorflowPoseBody(fps=10, data=masked_tensor, confidence=confidence)

        zero_filled_body = body.zero_filled()

        expected_sum = 5

        actual_sum = sum(zero_filled_body.data.numpy())

        self.assertEqual(actual_sum, expected_sum)

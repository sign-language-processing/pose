from unittest import TestCase

import tensorflow as tf

from pose_format.tensorflow.masked.tensor import MaskedTensor
from pose_format.tensorflow.pose_body import TensorflowPoseBody


class TestTensorflowPoseBody(TestCase):
    """TestsCases for the `TensorflowPoseBody` class."""

    def test_tf_pose_body_zero_filled_fills_in_zeros(self):
        """
        Test the `zero_filled` method of `TensorflowPoseBody` class.

        Test constructs a `TensorflowPoseBody` instance with specified fps, data (as a MaskedTensor),
        and confidence. It then calls `zero_filled` method and checks if data is filled with zeros.

        Raises
        ------
        AssertionError
            If sum of zero-filled body data is not equal to the expected sum.
        """
        tensor = tf.ones(7)
        confidence = tf.ones(7)
        mask = tf.constant([1, 1, 1, 0, 0, 1, 1])

        masked_tensor = MaskedTensor(tensor=tensor, mask=mask)
        body = TensorflowPoseBody(fps=10, data=masked_tensor, confidence=confidence)

        zero_filled_body = body.zero_filled()

        expected_sum = 5

        actual_sum = sum(zero_filled_body.data.numpy())

        self.assertEqual(actual_sum, expected_sum)

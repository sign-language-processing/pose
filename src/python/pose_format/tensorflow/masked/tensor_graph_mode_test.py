from unittest import TestCase

import tensorflow as tf

from pose_format.tensorflow.masked.tensor import MaskedTensor
from pose_format.tensorflow.masked.tensor_test import \
    create_random_numpy_tensor_and_mask


class TestMaskedTensor(TestCase):
    """
    Unit tests for `MaskedTensor` class in TensorFlow.
    """

    def test_float_graph_execution_fails(self):
        """
        Test if attempting to convert a MaskedTensor to float raises a TypeError during graph execution.

        Note
        ----
        This test ensures that if a user attempts to convert a MaskedTensor instance 
        to a floating point number during TensorFlow graph execution, 
        a TypeError should be raised.
        """
        with tf.Graph().as_default():

            assert tf.executing_eagerly() is False

            input_shape = (1,)

            tensor, mask = create_random_numpy_tensor_and_mask(shape=input_shape, probability_for_masked=0.1)

            masked_tf = MaskedTensor(tensor=tf.constant(tensor), mask=tf.constant(mask))

            with self.assertRaises(TypeError):
                float(masked_tf)

    def test_eq_graph_execution(self):
        """
        Test the equality operation on a MaskedTensor during graph execution.

        Note
        ----
        This test evaluates if the MaskedTensor can be compared with a 
        float number during TensorFlow graph execution without any errors.
        """

        with tf.Graph().as_default():

            assert tf.executing_eagerly() is False

            input_shape = (7, 6)

            tensor, mask = create_random_numpy_tensor_and_mask(shape=input_shape, probability_for_masked=0.1)

            masked_tf = MaskedTensor(tensor=tf.constant(tensor), mask=tf.constant(mask))

            _ = masked_tf == 6.0

import tensorflow as tf

from unittest import TestCase

from pose_format.tensorflow.masked.tensor import MaskedTensor

from pose_format.tensorflow.masked.tensor_test import create_random_numpy_tensor_and_mask


class TestMaskedTensor(TestCase):

    def test_float_graph_execution_fails(self):

        with tf.Graph().as_default():

            input_shape = (1,)

            tensor, mask = create_random_numpy_tensor_and_mask(shape=input_shape, probability_for_masked=0.1)

            masked_tf = MaskedTensor(tensor=tf.constant(tensor), mask=tf.constant(mask))

            with self.assertRaises(TypeError):
                float(masked_tf)

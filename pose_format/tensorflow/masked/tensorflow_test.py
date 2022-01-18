from unittest import TestCase

import tensorflow as tf

from pose_format.tensorflow.masked.tensor import MaskedTensor
from pose_format.tensorflow.masked.tensorflow import MaskedTensorflow


class TestMaskedTensorflow(TestCase):
    # cat
    def test_cat(self):
        tensor1 = MaskedTensor(tf.constant([1, 2]))
        tensor2 = MaskedTensor(tf.constant([3, 4]))
        stack = MaskedTensorflow.concat([tensor1, tensor2], axis=0)
        res = MaskedTensor(tf.constant([[1, 2, 3, 4]]))
        self.assertTrue(tf.reduce_all(stack == res), msg="Cat is not equal to expected")

    # stack
    def test_stack(self):
        tensor1 = MaskedTensor(tf.constant([1, 2]))
        tensor2 = MaskedTensor(tf.constant([3, 4]))
        stack = MaskedTensorflow.stack([tensor1, tensor2], axis=0)
        res = MaskedTensor(tf.constant([[1, 2], [3, 4]]))
        self.assertTrue(tf.reduce_all(stack == res), msg="Stack is not equal to expected")

    # zeros
    def test_zeros_tensor_shape(self):
        zeros = MaskedTensorflow.zeros((1, 2, 3))
        self.assertEqual(zeros.shape, (1, 2, 3))

    def test_zeros_tensor_value(self):
        zeros = MaskedTensorflow.zeros((1, 2, 3))
        self.assertTrue(tf.reduce_all(zeros == 0), msg="Zeros are not all zeros")

    def test_zeros_tensor_type_float(self):
        dtype = tf.float32
        zeros = MaskedTensorflow.zeros((1, 2, 3), dtype=dtype)
        self.assertEqual(zeros.tensor.dtype, dtype)

    def test_zeros_tensor_type_bool(self):
        dtype = tf.bool
        zeros = MaskedTensorflow.zeros((1, 2, 3), dtype=dtype)
        self.assertEqual(zeros.tensor.dtype, dtype)

    def test_zeros_mask_value(self):
        zeros = MaskedTensorflow.zeros((1, 2, 3))
        self.assertTrue(tf.reduce_all(zeros.mask == tf.zeros((1, 2, 3), dtype=tf.dtypes.bool)),
                        msg="Zeros mask are not all zeros")

    # Fallback
    def test_not_implemented_method(self):
        tensor = MaskedTensor(tensor=tf.constant([1, 2, 3]))
        tensor_square = MaskedTensorflow.square(tensor)
        self.assertTrue(tf.reduce_all(tensor_square == tf.constant([1, 4, 9])), msg="Square is not equal to expected")

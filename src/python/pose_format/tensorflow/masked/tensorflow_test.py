from unittest import TestCase

import tensorflow as tf

from pose_format.tensorflow.masked.tensor import MaskedTensor
from pose_format.tensorflow.masked.tensorflow import MaskedTensorflow


class TestMaskedTensorflow(TestCase):
    """
    A test class for checking the operations in the MaskedTensorflow class.
    """

    # cat
    def test_cat(self):
        """Test the `concat` concatination method for two tensors along axis=0."""
        tensor1 = MaskedTensor(tf.constant([1, 2]))
        tensor2 = MaskedTensor(tf.constant([3, 4]))
        stack = MaskedTensorflow.concat([tensor1, tensor2], axis=0)
        res = MaskedTensor(tf.constant([[1, 2, 3, 4]]))
        self.assertTrue(tf.reduce_all(stack == res), msg="Cat is not equal to expected")

    # stack
    def test_stack(self):
        """
        Test the `stack` method for two tensors along axis=0.

        """
        tensor1 = MaskedTensor(tf.constant([1, 2]))
        tensor2 = MaskedTensor(tf.constant([3, 4]))
        stack = MaskedTensorflow.stack([tensor1, tensor2], axis=0)
        res = MaskedTensor(tf.constant([[1, 2], [3, 4]]))
        self.assertTrue(tf.reduce_all(stack == res), msg="Stack is not equal to expected")

    # zeros
    def test_zeros_tensor_shape(self):
        """
        Test the shape of the tensor obtained from `zeros` method.

        """
        zeros = MaskedTensorflow.zeros((1, 2, 3))
        self.assertEqual(zeros.shape, (1, 2, 3))

    def test_zeros_tensor_value(self):
        """
        Test the values of the tensor obtained from `zeros` method.

        """
        zeros = MaskedTensorflow.zeros((1, 2, 3))
        self.assertTrue(tf.reduce_all(zeros == 0), msg="Zeros are not all zeros")

    def test_zeros_tensor_type_float(self):
        """
        Test  dtype of  tensor obtained from `zeros` method with dtype=tf.float32.
        """
        dtype = tf.float32
        zeros = MaskedTensorflow.zeros((1, 2, 3), dtype=dtype)
        self.assertEqual(zeros.tensor.dtype, dtype)

    def test_zeros_tensor_type_bool(self):
        """Test the dtype of the tensor obtained from `zeros` method with dtype=tf.bool."""
        dtype = tf.bool
        zeros = MaskedTensorflow.zeros((1, 2, 3), dtype=dtype)
        self.assertEqual(zeros.tensor.dtype, dtype)

    def test_zeros_mask_value(self):
        """
        Test the mask values of the tensor obtained from `zeros` method.
        """
        zeros = MaskedTensorflow.zeros((1, 2, 3))
        self.assertTrue(tf.reduce_all(zeros.mask == tf.zeros((1, 2, 3), dtype=tf.dtypes.bool)),
                        msg="Zeros mask are not all zeros")

    # Fallback
    def test_not_implemented_method(self):
        """
        Test a method that is not explicitly defined for MaskedTensor but is implemented via fallback.
        """
        tensor = MaskedTensor(tensor=tf.constant([1, 2, 3]))
        tensor_square = MaskedTensorflow.square(tensor)
        self.assertTrue(tf.reduce_all(tensor_square == tf.constant([1, 4, 9])), msg="Square is not equal to expected")

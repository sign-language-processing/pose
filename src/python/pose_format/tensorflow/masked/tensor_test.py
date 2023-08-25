from typing import Tuple
from unittest import TestCase

import numpy as np
import numpy.ma as ma
import tensorflow as tf

from pose_format.tensorflow.masked.tensor import MaskedTensor


def create_random_numpy_tensor_and_mask(shape: Tuple,
                                        probability_for_masked: float,
                                        num_nans: int = 0) -> Tuple[np.array, np.array]:
    """
    Creates a random numpy tensor and a corresponding mask.

    Parameters
    ----------
    shape : Tuple
        The desired shape of the tensor.
    probability_for_masked : float
        The probability that an element is masked.
    num_nans : int, optional
        Number of NaNs to be inserted into the tensor, default is 0.

    Returns
    -------
    Tuple[np.array, np.array]
        A tuple containing the generated tensor and its corresponding mask.
    """
    tensor = np.random.random_sample(size=shape)

    if num_nans > 0:
        index = np.random.choice(tensor.size, num_nans, replace=False)
        tensor.ravel()[index] = np.nan

    mask = np.random.choice(a=[False, True], size=shape, p=[probability_for_masked, 1 - probability_for_masked])

    return tensor, mask


class TestMaskedTensor(TestCase):
    """Unit tests for the MaskedTensor class."""

    def test_fix_nan(self):
        """
        Test if NaN values in a MaskedTensor are fixed (removed).
        """

        tensor, mask = create_random_numpy_tensor_and_mask(shape=(5, 7, 11), probability_for_masked=0.2, num_nans=20)

        masked_tf = MaskedTensor(tensor=tf.constant(tensor), mask=tf.constant(mask))

        nan_fixed = masked_tf.fix_nan()
        result_as_numpy = nan_fixed.zero_filled().numpy()

        self.assertFalse(np.isnan(result_as_numpy).any(), msg="Function fix_nan did not remove all nan values.")

    def test_mean(self):
        """
        Test if the computed mean of a MaskedTensor matches the numpy MaskedArray.
        """

        tensor, mask = create_random_numpy_tensor_and_mask(shape=(5, 7), probability_for_masked=0.2)

        masked_np = ma.array(tensor, mask=np.logical_not(mask))
        mean_ma = masked_np.mean(axis=0)

        masked_tf = MaskedTensor(tensor=tf.constant(tensor), mask=tf.constant(mask))
        mean_tf = masked_tf.mean(axis=0)

        np_tf = mean_tf.zero_filled().numpy()
        np_ma = mean_ma.filled(0)

        self.assertTrue(np.allclose(np_tf, np_ma), msg="Mean is not equal to expected")

    def test_std(self):
        """
        Test if the computed standard deviation of a MaskedTensor matches the numpy MaskedArray.
        """
        tensor, mask = create_random_numpy_tensor_and_mask(shape=(7, 3), probability_for_masked=0.1)

        masked_np = ma.array(tensor, mask=np.logical_not(mask))
        std_ma = masked_np.std(axis=0)

        masked_tf = MaskedTensor(tensor=tf.constant(tensor), mask=tf.constant(mask))
        std_tf = masked_tf.std(axis=0)

        np_tf = std_tf.zero_filled().numpy()
        np_ma = std_ma.filled(0)

        self.assertTrue(np.allclose(np_tf, np_ma), msg="STD is not equal to expected")

    def test_reshape_identical_to_numpy_reshape(self):
        """
        Test if the reshape method of a MaskedTensor produces results identical to numpy's reshape.
        """

        input_shape = (7, 3, 4)
        target_shape = (21, 4)

        tensor, mask = create_random_numpy_tensor_and_mask(shape=input_shape, probability_for_masked=0.1)
        masked_np = ma.array(tensor, mask=np.logical_not(mask))
        reshaped_np = masked_np.reshape(target_shape)
        reshaped_expected = reshaped_np.filled(0)

        masked_tf = MaskedTensor(tensor=tf.constant(tensor), mask=tf.constant(mask))
        reshaped_tf = masked_tf.reshape(shape=target_shape)
        reshaped_actual = reshaped_tf.zero_filled().numpy()

        self.assertTrue(np.allclose(reshaped_actual, reshaped_expected),
                        msg="Reshape operations do not produce the same result")

    def test_reshape_return_type_is_correct(self):
        """
        Test if the return type of the reshape method of a MaskedTensor is itself a MaskedTensor.
        """
        input_shape = (12,)
        target_shape = (3, 4)

        tensor, mask = create_random_numpy_tensor_and_mask(shape=input_shape, probability_for_masked=0.1)

        masked_tf = MaskedTensor(tensor=tf.constant(tensor), mask=tf.constant(mask))
        reshaped_tf = masked_tf.reshape(shape=target_shape)

        self.assertIsInstance(reshaped_tf, MaskedTensor, "Return value of reshape is not an instance of MaskedTensor")

    def test_float_eager_execution_return_type_is_correct(self):
        """
        Test if a MaskedTensor can be correctly cast to a float during eager execution.
        """
        input_shape = (1,)

        tensor, mask = create_random_numpy_tensor_and_mask(shape=input_shape, probability_for_masked=0.1)

        masked_tf = MaskedTensor(tensor=tf.constant(tensor), mask=tf.constant(mask))
        as_float = float(masked_tf)

        self.assertIsInstance(as_float, float, "Dtype after casting masked tensor to float not correct")

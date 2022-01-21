from unittest import TestCase

import numpy.ma as ma
import numpy as np

import tensorflow as tf

from typing import Tuple

from pose_format.tensorflow.masked.tensor import MaskedTensor


def create_random_numpy_tensor_and_mask(shape: Tuple,
                                        probability_for_masked: float,
                                        num_nans: int = 0) -> Tuple[np.array, np.array]:

    tensor = np.random.random_sample(size=shape)

    if num_nans > 0:
        index = np.random.choice(tensor.size, num_nans, replace=False)
        tensor.ravel()[index] = np.nan

    mask = np.random.choice(a=[False, True], size=shape, p=[probability_for_masked, 1 - probability_for_masked])

    return tensor, mask


class TestMaskedTensor(TestCase):

    def test_fix_nan(self):

        tensor, mask = create_random_numpy_tensor_and_mask(shape=(5, 7, 11),
                                                           probability_for_masked=0.2,
                                                           num_nans=20)

        masked_tf = MaskedTensor(tensor=tf.constant(tensor), mask=tf.constant(mask))

        nan_fixed = masked_tf.fix_nan()
        result_as_numpy = nan_fixed.zero_filled().numpy()

        self.assertFalse(np.isnan(result_as_numpy).any(), msg="Function fix_nan did not remove all nan values.")

    def test_mean(self):

        tensor, mask = create_random_numpy_tensor_and_mask(shape=(5, 7), probability_for_masked=0.2)

        masked_np = ma.array(tensor, mask=np.logical_not(mask))
        mean_ma = masked_np.mean(axis=0)

        masked_tf = MaskedTensor(tensor=tf.constant(tensor), mask=tf.constant(mask))
        mean_tf = masked_tf.mean(axis=0)

        np_tf = mean_tf.zero_filled().numpy()
        np_ma = mean_ma.filled(0)

        self.assertTrue(np.allclose(np_tf, np_ma), msg="Mean is not equal to expected")

    def test_std(self):
        tensor, mask = create_random_numpy_tensor_and_mask(shape=(7, 3), probability_for_masked=0.1)

        masked_np = ma.array(tensor, mask=np.logical_not(mask))
        std_ma = masked_np.std(axis=0)

        masked_tf = MaskedTensor(tensor=tf.constant(tensor), mask=tf.constant(mask))
        std_tf = masked_tf.std(axis=0)

        np_tf = std_tf.zero_filled().numpy()
        np_ma = std_ma.filled(0)

        self.assertTrue(np.allclose(np_tf, np_ma), msg="STD is not equal to expected")

    def test_reshape_identical_to_numpy_reshape(self):

        input_shape = (7, 3, 4)
        target_shape = (21, 4)

        tensor, mask = create_random_numpy_tensor_and_mask(shape=input_shape, probability_for_masked=0.1)
        masked_np = ma.array(tensor, mask=np.logical_not(mask))
        reshaped_np = masked_np.reshape(target_shape)
        reshaped_expected = reshaped_np.filled(0)

        masked_tf = MaskedTensor(tensor=tf.constant(tensor), mask=tf.constant(mask))
        reshaped_tf = masked_tf.reshape(shape=target_shape)
        reshaped_actual = reshaped_tf.zero_filled().numpy()

        self.assertTrue(np.allclose(reshaped_actual, reshaped_expected), msg="Reshape operations do not produce the same result")

    def test_reshape_return_type_is_correct(self):

        input_shape = (12,)
        target_shape = (3, 4)

        tensor, mask = create_random_numpy_tensor_and_mask(shape=input_shape, probability_for_masked=0.1)

        masked_tf = MaskedTensor(tensor=tf.constant(tensor), mask=tf.constant(mask))
        reshaped_tf = masked_tf.reshape(shape=target_shape)

        self.assertIsInstance(reshaped_tf, MaskedTensor, "Return value of reshape is not an instance of MaskedTensor")

    def test_float_eager_execution_return_type_is_correct(self):

        input_shape = (1,)

        tensor, mask = create_random_numpy_tensor_and_mask(shape=input_shape, probability_for_masked=0.1)

        masked_tf = MaskedTensor(tensor=tf.constant(tensor), mask=tf.constant(mask))
        as_float = float(masked_tf)

        self.assertIsInstance(as_float, float, "Dtype after casting masked tensor to float not correct")

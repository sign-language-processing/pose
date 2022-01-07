from unittest import TestCase

import tensorflow as tf

from pose_format.tensorflow.masked.tensor import MaskedTensor
import numpy.ma as ma
import numpy as np


class TestMaskedTensor(TestCase):
    def test_mean(self):
        tensor = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float)
        mask = np.array([[1, 1], [0, 1], [1, 1]], dtype=np.bool)

        masked_np = ma.array(tensor, mask=np.logical_not(mask))
        mean_ma = masked_np.mean(axis=(0))

        masked_tf = MaskedTensor(tensor=tf.constant(tensor), mask=tf.constant(mask))
        mean_tf = masked_tf.mean(axis=(0))

        np_tf = mean_tf.zero_filled().numpy()
        np_ma = mean_ma.filled(0)

        self.assertTrue(np.allclose(np_tf, np_ma), msg="Mean is not equal to expected")

    def test_std(self):
        tensor = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float)
        mask = np.array([[1, 1], [0, 1], [1, 1]], dtype=np.bool)

        masked_np = ma.array(tensor, mask=np.logical_not(mask))
        std_ma = masked_np.std(axis=(0))

        masked_tf = MaskedTensor(tensor=tf.constant(tensor), mask=tf.constant(mask))
        std_tf = masked_tf.std(axis=(0))

        np_tf = std_tf.zero_filled().numpy()
        np_ma = std_ma.filled(0)

        self.assertTrue(np.allclose(np_tf, np_ma), msg="STD is not equal to expected")

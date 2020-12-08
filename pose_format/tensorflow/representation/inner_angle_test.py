import os
from unittest import TestCase

import math

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import tensorflow as tf

from pose_format.tensorflow.representation.inner_angle import InnerAngleRepresentation

representation = InnerAngleRepresentation()


class TestInnerAngleRepresentation(TestCase):
    def test_call_value_should_be_inner_angle(self):
        p1s = tf.constant([[[[2, 3, 4]]]], dtype=tf.float32)
        p2s = tf.constant([[[[1, 1, 1]]]], dtype=tf.float32)
        p3s = tf.constant([[[[3, 4, 2]]]], dtype=tf.float32)
        angles = representation(p1s, p2s, p3s)
        self.assertAlmostEqual(float(angles[0][0][0]), math.acos(11 / 14))

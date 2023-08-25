import math
import os
from unittest import TestCase

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import tensorflow as tf

from pose_format.tensorflow.representation.angle import AngleRepresentation

representation = AngleRepresentation()


class TestAngleRepresentation(TestCase):
    """Test case for AngleRepresentation class"""

    def test_call_value_should_be_angle(self):
        """Test if the calculated angles are correct."""
        p1s = tf.constant([[[[1, 2, 3]]]], dtype=tf.float32)
        p2s = tf.constant([[[[4, 5, 6]]]], dtype=tf.float32)
        angles = representation(p1s, p2s)
        self.assertAlmostEqual(float(angles[0][0][0]), 1 / 4 * math.pi)

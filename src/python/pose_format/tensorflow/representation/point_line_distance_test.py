import math
import os
from unittest import TestCase

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import tensorflow as tf

from pose_format.tensorflow.representation.point_line_distance import \
    PointLineDistanceRepresentation

representation = PointLineDistanceRepresentation()


class TestPointLineDistanceRepresentation(TestCase):
    """
    Test case for PointLineDistanceRepresentation class
"""

    def test_call_value_should_be_distance(self):
        """Test if the calculated distances are correct"""
        p1s = tf.constant([[[[2, 3, 4]]]], dtype=tf.float32)
        p2s = tf.constant([[[[1, 1, 1]]]], dtype=tf.float32)
        p3s = tf.constant([[[[3, 4, 2]]]], dtype=tf.float32)
        distances = representation(p1s, p2s, p3s)
        self.assertAlmostEqual(float(distances[0][0][0]), math.sqrt(75 / 14), places=6)

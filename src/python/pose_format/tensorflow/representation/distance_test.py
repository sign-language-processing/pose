import math
import os
from unittest import TestCase

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import tensorflow as tf

from pose_format.tensorflow.representation.distance import \
    DistanceRepresentation

representation = DistanceRepresentation()


class TestDistanceRepresentation(TestCase):
    """Test case for DistanceRepresentation class."""

    def test_call_value_should_be_distance(self):
        """Test if the calculated distances are correct."""
        p1s = tf.constant([[[[1, 2, 3]]]], dtype=tf.float32)
        p2s = tf.constant([[[[4, 5, 6]]]], dtype=tf.float32)
        distances = representation(p1s, p2s)
        self.assertAlmostEqual(float(distances[0][0][0]), math.sqrt(27), places=6)

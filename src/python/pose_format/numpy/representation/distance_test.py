from unittest import TestCase

import numpy as np
import numpy.ma as ma
import math

from pose_format.numpy.representation.distance import DistanceRepresentation

representation = DistanceRepresentation()


class TestDistanceRepresentation(TestCase):
    def test_call_value_should_be_distance(self):
        p1s = ma.array([[[[1, 2, 3]]]], dtype=np.float32)
        p2s = ma.array([[[[4, 5, 6]]]], dtype=np.float32)
        distances = representation(p1s, p2s)
        self.assertAlmostEqual(float(distances[0][0][0]), math.sqrt(27), places=6)

    def test_call_masked_value_should_be_zero(self):
        mask = [[[[True, True, True]]]]
        p1s = ma.array([[[[1, 2, 3]]]], mask=mask, dtype=np.float32)
        p2s = ma.array([[[[4, 5, 6]]]], dtype=np.float32)
        distances = representation(p1s, p2s)
        self.assertEqual(float(distances[0][0][0]), 0)

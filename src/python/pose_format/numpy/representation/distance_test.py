import math
from unittest import TestCase

import numpy as np
import numpy.ma as ma

from pose_format.numpy.representation.distance import DistanceRepresentation

representation = DistanceRepresentation()


class TestDistanceRepresentation(TestCase):
    """
    Unit tests for DistanceRepresentation class to computes Euclidean distance between sets of 3D points.
    """

    def test_call_value_should_be_distance(self):
        """
        Test if computed distance between two points is correct.

        Note
        ----
        This test initializes two sets of 3D points:
        p1 = (1, 2, 3) and p2 = (4, 5, 6). It then checks if the 
        computed Euclidean distance between p1 and p2 is sqrt(27).
        """

        p1s = ma.array([[[[1, 2, 3]]]], dtype=np.float32)
        p2s = ma.array([[[[4, 5, 6]]]], dtype=np.float32)
        distances = representation(p1s, p2s)
        self.assertAlmostEqual(float(distances[0][0][0]), math.sqrt(27), places=6)

    def test_call_masked_value_should_be_zero(self):
        """
        Test if the distance for masked values is zero.

        Note
        ----
        This test checks scenario where one set of points 
        is masked.The computed distance in such in such a case be 0.
        """

        mask = [[[[True, True, True]]]]
        p1s = ma.array([[[[1, 2, 3]]]], mask=mask, dtype=np.float32)
        p2s = ma.array([[[[4, 5, 6]]]], dtype=np.float32)
        distances = representation(p1s, p2s)
        self.assertEqual(float(distances[0][0][0]), 0)

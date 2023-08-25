import math
from unittest import TestCase

import torch

from pose_format.torch.masked.tensor import MaskedTensor
from pose_format.torch.representation.distance import DistanceRepresentation

representation = DistanceRepresentation()


class TestDistanceRepresentation(TestCase):
    """
    Test cases of distance representation between points.
    """

    def test_call_value_should_be_distance(self):
        """
        Tests if the returned distance is as expected for given non-masked/unmasked points.
        """
        p1s = MaskedTensor(torch.tensor([[[[1, 2, 3]]]], dtype=torch.float))
        p2s = MaskedTensor(torch.tensor([[[[4, 5, 6]]]], dtype=torch.float))
        distances = representation(p1s, p2s)
        self.assertAlmostEqual(float(distances[0][0][0]), math.sqrt(27), places=6)

    def test_call_masked_value_should_be_zero(self):
        """
        Test if masked values return a distance of zero.
        """
        mask = torch.tensor([[[[0, 1, 1]]]], dtype=torch.bool)
        p1s = MaskedTensor(torch.tensor([[[[1, 2, 3]]]], dtype=torch.float), mask)
        p2s = MaskedTensor(torch.tensor([[[[4, 5, 6]]]], dtype=torch.float))
        distances = representation(p1s, p2s)
        self.assertEqual(float(distances[0][0][0]), 0)

from unittest import TestCase

import math
import torch

from pose_format.torch.masked.tensor import MaskedTensor
from pose_format.torch.representation.point_line_distance import PointLineDistanceRepresentation

representation = PointLineDistanceRepresentation()


class TestPointLineDistanceRepresentation(TestCase):
    def test_call_value_should_be_distance(self):
        p1s = MaskedTensor(torch.tensor([[[[2, 3, 4]]]], dtype=torch.float))
        p2s = MaskedTensor(torch.tensor([[[[1, 1, 1]]]], dtype=torch.float))
        p3s = MaskedTensor(torch.tensor([[[[3, 4, 2]]]], dtype=torch.float))
        distances = representation(p1s, p2s, p3s)
        self.assertAlmostEqual(float(distances[0][0][0]), math.sqrt(75 / 14), places=6)

    def test_call_masked_value_should_be_zero(self):
        mask = torch.tensor([[[[0, 1, 1]]]], dtype=torch.bool)
        p1s = MaskedTensor(torch.tensor([[[[2, 3, 4]]]], dtype=torch.float), mask)
        p2s = MaskedTensor(torch.tensor([[[[1, 1, 1]]]], dtype=torch.float))
        p3s = MaskedTensor(torch.tensor([[[[3, 4, 2]]]], dtype=torch.float))
        distances = representation(p1s, p2s, p3s)
        self.assertEqual(float(distances[0][0][0]), 0)

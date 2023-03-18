from unittest import TestCase

import math
import torch

from pose_format.torch.masked.tensor import MaskedTensor
from pose_format.torch.representation.angle import AngleRepresentation

representation = AngleRepresentation()


class TestAngleRepresentation(TestCase):
    def test_call_value_should_be_angle(self):
        p1s = MaskedTensor(torch.tensor([[[[1, 2, 3]]]], dtype=torch.float))
        p2s = MaskedTensor(torch.tensor([[[[4, 5, 6]]]], dtype=torch.float))
        angles = representation(p1s, p2s)
        self.assertAlmostEqual(float(angles[0][0][0]), 1 / 4 * math.pi)

    def test_call_masked_value_should_be_zero(self):
        mask = torch.tensor([[[[0, 1, 1]]]], dtype=torch.bool)
        p1s = MaskedTensor(torch.tensor([[[[1, 2, 3]]]], dtype=torch.float), mask)
        p2s = MaskedTensor(torch.tensor([[[[4, 5, 6]]]], dtype=torch.float))
        angles = representation(p1s, p2s)
        self.assertEqual(float(angles[0][0][0]), 0)

import math
from unittest import TestCase

import torch

from pose_format.torch.masked.tensor import MaskedTensor
from pose_format.torch.representation.inner_angle import \
    InnerAngleRepresentation

representation = InnerAngleRepresentation()


class TestInnerAngleRepresentation(TestCase):
    """
    Unit test for the `InnerAngleRepresentation` class.
    
    This test class verifies that the angle calculations for the InnerAngleRepresentation 
    class are correct and handle edge cases appropriately.
    
    Methods
    -------
    test_call_value_should_be_inner_angle():
        Tests if the calculated angle matches the expected angle value.
    
    test_call_masked_value_should_be_zero():
        Tests if a masked value in the input results in a zero output angle.
    """

    def test_call_value_should_be_inner_angle(self):
        """
        Tests if the computed angle from the `InnerAngleRepresentation` matches the expected value.
        
        This test sets up three points and expects the computed angle at the middle point to be 
        approximately equal to the angle calculated via arccosine of a predefined value.
        """
        p1s = MaskedTensor(torch.tensor([[[[2, 3, 4]]]], dtype=torch.float))
        p2s = MaskedTensor(torch.tensor([[[[1, 1, 1]]]], dtype=torch.float))
        p3s = MaskedTensor(torch.tensor([[[[3, 4, 2]]]], dtype=torch.float))
        angles = representation(p1s, p2s, p3s)
        self.assertAlmostEqual(float(angles[0][0][0]), math.acos(11 / 14))

    def test_call_masked_value_should_be_zero(self):
        """
        Tests if masking an input value results in an output angle of zero.
        
        This test masks one of the input points and expects the computed angle at the middle 
        point to be zero.
        """
        mask = torch.tensor([[[[0, 1, 1]]]], dtype=torch.bool)
        p1s = MaskedTensor(torch.tensor([[[[2, 3, 4]]]], dtype=torch.float), mask)
        p2s = MaskedTensor(torch.tensor([[[[1, 1, 1]]]], dtype=torch.float))
        p3s = MaskedTensor(torch.tensor([[[[3, 4, 2]]]], dtype=torch.float))
        angles = representation(p1s, p2s, p3s)
        self.assertEqual(float(angles[0][0][0]), 0)

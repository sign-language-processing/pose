"""
Module for testing optical flow computations. Provides test cases for optical flow computation from pose landmarks data.
"""
from unittest import TestCase

import numpy as np

from pose_format.numpy.representation.distance import DistanceRepresentation
from pose_format.pose import Pose
from pose_format.utils.optical_flow import OpticalFlowCalculator


class TestOpticalFlow(TestCase):
    """
    Test cases for optical flow computation.
    """

    def test_optical_flow(self):
        """
        Tests optical flow from pose landmarks data against a numeric reference.

        Compares the computed flow with `data/optical_flow.npy`. The reference is
        the raw numeric output (not a rendered image), so the test is independent
        of matplotlib/freetype rendering differences across environments.
        """
        calculator = OpticalFlowCalculator(fps=30, distance=DistanceRepresentation())

        with open('tests/data/mediapipe.pose', 'rb') as f:
            pose = Pose.read(f.read())
            pose = pose.get_components(["POSE_LANDMARKS", "RIGHT_HAND_LANDMARKS", "LEFT_HAND_LANDMARKS"])

        flow = calculator(pose.body.data).squeeze(axis=1)

        expected = np.load('tests/data/optical_flow.npy')
        self.assertEqual(flow.shape, expected.shape)
        self.assertTrue(np.allclose(flow, expected),
                        "Computed optical flow does not match the reference.")

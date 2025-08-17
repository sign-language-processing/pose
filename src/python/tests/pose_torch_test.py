from typing import Optional, Tuple
from unittest import TestCase

import torch
import numpy as np

from pose_format import Pose

from pose_format.torch.masked import MaskedTensor as TorchMaskedTensor
from pose_format.torch.pose_body import TorchPoseBody

from .pose_test import _create_pose_header


def _create_random_torch_data(frames_min: Optional[int] = None,
                              frames_max: Optional[int] = None,
                              num_frames: Optional[int] = None,
                              num_keypoints: int = 137,
                              num_dimensions: int = 2) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Creates random PyTorch data for testing.

    Parameters
    ----------
    frames_min : Optional[int], default=None
        Minimum number of frames for random generation if `num_frames` is not specified.
    frames_max : Optional[int], default=None
        Maximum number of frames for random generation if `num_frames` is not specified.
    num_frames : Optional[int], default=None
        Specific number of frames.
    num_keypoints : int, default=137
        Number of keypoints in the pose data.
    num_dimensions : int, default=2
        Number of dimensions in the pose data.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        Random tensor data, mask, and confidence values.
    """
    if num_frames is None:
        assert None not in [frames_min, frames_max]
        num_frames = np.random.randint(frames_min, frames_max + 1)
    else:
        assert frames_min is None and frames_max is None

    # Avoid a mean of zero to test certain pose methods
    tensor = torch.randn((num_frames, 1, num_keypoints, num_dimensions)) + 1.0  # Shape: (Frames, People, Points, Dims)

    confidence = torch.rand((num_frames, 1, num_keypoints), dtype=torch.float32)  # Shape: (Frames, People, Points)

    mask = torch.randint(0, 2, (num_frames, 1, num_keypoints, num_dimensions), dtype=torch.bool)  # Bool mask

    return tensor, mask, confidence



def _get_random_pose_object_with_torch_posebody(num_keypoints: int, frames_min: int = 1, frames_max: int = 10) -> Pose:
    """
    Generates a random Pose object with PyTorch pose body for testing.

    Parameters
    ----------
    num_keypoints : int
        Number of keypoints in the pose data.
    frames_min : int, default=1
        Minimum number of frames for random generation.
    frames_max : int, default=10
        Maximum number of frames for random generation.

    Returns
    -------
    Pose
        Randomly generated Pose object.
    """

    tensor, mask, confidence = _create_random_torch_data(frames_min=frames_min,
                                                         frames_max=frames_max,
                                                         num_keypoints=num_keypoints)

    masked_tensor = TorchMaskedTensor(tensor=tensor, mask=mask)
    body = TorchPoseBody(fps=10, data=masked_tensor, confidence=confidence)

    header = _create_pose_header(width=10, height=7, depth=0, num_components=3, num_keypoints=num_keypoints)

    return Pose(header=header, body=body)



class TestPoseTorchPoseBody(TestCase):

    def test_pose_torch_posebody_copy_tensors_detached(self):
        pose = _get_random_pose_object_with_torch_posebody(num_keypoints=5)
        pose_copy = pose.copy()

        self.assertFalse(pose.body.data.data.requires_grad, "Copied data should be detached from computation graph")
        self.assertFalse(pose_copy.body.data.mask.requires_grad,
                         "Copied mask should be detached from computation graph")

    def test_pose_torch_posebody_copy_creates_deepcopy(self):
        pose = _get_random_pose_object_with_torch_posebody(num_keypoints=5)
        self.assertIsInstance(pose.body, TorchPoseBody)
        self.assertIsInstance(pose.body.data, TorchMaskedTensor)

        pose_copy = pose.copy()
        self.assertIsInstance(pose_copy.body, TorchPoseBody)
        self.assertIsInstance(pose_copy.body.data, TorchMaskedTensor)

        self.assertNotEqual(pose, pose_copy, "Copy of pose should not be 'equal' to original")
        self.assertTrue(pose.body.data.tensor.equal(pose_copy.body.data.tensor), "Copy's data should match original")
        self.assertTrue(pose.body.data.mask.equal(pose_copy.body.data.mask), "Copy's mask should match original")

        pose.body.data = TorchMaskedTensor(tensor=torch.zeros_like(pose.body.data.tensor),
                                           mask=torch.ones_like(pose.body.data.mask))

        self.assertFalse(pose.body.data.tensor.equal(pose_copy.body.data.tensor),
                         "Copy's data should not match original after original is replaced")
        self.assertFalse(pose.body.data.mask.equal(pose_copy.body.data.mask),
                         "Copy's mask should not match original after original is replaced")

        pose = pose_copy.copy()

        self.assertTrue(pose.body.data.tensor.equal(pose_copy.body.data.tensor),
                        "Copy's data should match original again")
        self.assertTrue(pose.body.data.mask.equal(pose_copy.body.data.mask), "Copy's mask should match original again")

        pose_copy.body.data.tensor.fill_(3.14)

        self.assertFalse(pose.body.data.tensor.equal(pose_copy.body.data.tensor),
                         "Copy's data should not match original after copy is modified")

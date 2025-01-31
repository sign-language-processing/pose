from typing import List, Union

import numpy as np
import torch

from ..pose_body import POINTS_DIMS, PoseBody
from .masked.tensor import MaskedTensor


class TorchPoseBody(PoseBody):
    """
    TorchPoseBody class of pose information with PyTorch tensors.

    This class extends the PoseBody class and provides methods for manipulating pose data using PyTorch tensors.
    """

    """str: Reader format for unpacking Torch tensors."""
    tensor_reader = 'unpack_torch'

    def __init__(self, fps: float, data: Union[MaskedTensor, torch.Tensor], confidence: torch.Tensor):
        if isinstance(data, torch.Tensor):  # If array is not masked
            mask = confidence > 0
            stacked_mask = torch.stack([mask] * data.shape[-1], dim=3)
            data = MaskedTensor(data, stacked_mask)

        super().__init__(fps, data, confidence)

    def cuda(self):
        """Move data and confidence of tensors to GPU"""
        self.data = self.data.cuda()
        self.confidence = self.confidence.cuda()

    def copy(self) -> 'TorchPoseBody':
        data_copy = MaskedTensor(tensor=self.data.tensor.detach().clone().to(self.data.tensor.device),
                                 mask=self.data.mask.detach().clone().to(self.data.mask.device),
                                 )
        confidence_copy = self.confidence.detach().clone().to(self.confidence.device)

        return self.__class__(fps=self.fps,
                             data=data_copy,
                             confidence=confidence_copy)


    def zero_filled(self) -> 'TorchPoseBody':
        """
        Fill invalid values with zeros.

        Returns
        -------
        TorchPoseBody
            TorchPoseBody instance with masked data filled with zeros.

        """
        copy = self.copy()
        copy.data = copy.data.zero_filled()
        return copy

    def matmul(self, matrix: np.ndarray) -> 'TorchPoseBody':
        """
        Matrix multiplication on pose data.

        Parameters
        ----------
        matrix : np.ndarray
            matrix to perform multiplication with

        Returns
        -------
        TorchPoseBody
            A new TorchPoseBody instance with results of matrix multiplication.

        """
        data = self.data.matmul(torch.from_numpy(matrix))
        return self.__class__(fps=self.fps, data=data, confidence=self.confidence)

    def points_perspective(self):
        """
        Get pose data with dimensions permuted according to POINTS_DIMS.

        Returns
        -------
        :class:`~pose_format.torch.masked.tensor.MaskedTensor`
            A :class:`~pose_format.torch.masked.tensor.MaskedTensor` instance with dimensions permuted for points perspective.

        """
        return self.data.permute(POINTS_DIMS)

    def get_points(self, indexes: List[int]):
        """
        Get specific points from pose data.

        Parameters
        ----------
        indexes : List[int]
            List of indexes specifying the points that you need.

        Returns
        -------
        TorchPoseBody
            New TorchPoseBody instance containing specified points and associated confidence values.

        """
        data = self.points_perspective()
        new_data = data[indexes].permute(POINTS_DIMS)

        confidence_reshape = (2, 1, 0)
        confidence = self.confidence.permute(confidence_reshape)
        new_confidence = confidence[indexes].permute(confidence_reshape)

        return self.__class__(self.fps, new_data, new_confidence)

    def flatten(self):
        """
        Flatten pose data along the associated confidence values.

        Returns
        -------
        torch.Tensor
            Flattened tensor containing indexes, confidence values, and data.

        """
        shape = self.data.shape
        data = self.data.tensor.reshape(-1, shape[-1])  # Not masked data
        confidence = self.confidence.flatten()
        indexes = torch.tensor(list(np.ndindex(shape[:-1])), dtype=torch.float32, device=data.device)
        flat = torch.cat([indexes, torch.unsqueeze(confidence, dim=1), data], dim=1)
        # Filter data from flat
        flat = flat[confidence != 0.]
        # Scale the first axis by fps
        scalar = torch.ones(len(shape) + shape[-1], device=data.device)
        scalar[0] = 1 / self.fps
        return flat * scalar




from typing import List, Union

import numpy as np
import torch

from .masked.tensor import MaskedTensor
from ..pose_body import PoseBody, POINTS_DIMS
from ..pose_header import PoseHeader
from ..utils.reader import BufferReader


class TorchPoseBody(PoseBody):
    tensor_reader = 'unpack_torch'

    def __init__(self, fps: float, data: Union[MaskedTensor, torch.Tensor], confidence: torch.Tensor):
        if isinstance(data, torch.Tensor):  # If array is not masked
            mask = confidence > 0
            stacked_mask = torch.stack([mask] * data.shape[-1], dim=3)
            data = MaskedTensor(data, stacked_mask)

        super().__init__(fps, data, confidence)

    def cuda(self):
        self.data = self.data.cuda()
        self.confidence = self.confidence.cuda()

    def zero_filled(self):
        self.data.zero_filled()
        return self

    def matmul(self, matrix: np.ndarray):
        data = self.data.matmul(torch.from_numpy(matrix))
        return self.__class__(fps=self.fps, data=data, confidence=self.confidence)

    def points_perspective(self):
        return self.data.permute(POINTS_DIMS)

    def get_points(self, indexes: List[int]):
        data = self.points_perspective()
        new_data = data[indexes].permute(POINTS_DIMS)

        confidence_reshape = (2, 1, 0)
        confidence = self.confidence.permute(confidence_reshape)
        new_confidence = confidence[indexes].permute(confidence_reshape)

        return self.__class__(self.fps, new_data, new_confidence)

    def flatten(self):
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

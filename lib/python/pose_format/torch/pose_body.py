from typing import List, Union

import torch
import numpy as np

from ..pose_header import PoseHeader
from ..pose_body import PoseBody, POINTS_DIMS
from ..utils.reader import BufferReader
from .masked.tensor import MaskedTensor


class TorchPoseBody(PoseBody):
    tensor_reader = 'unpack_torch'

    def __init__(self, fps: int, data: Union[MaskedTensor, torch.Tensor], confidence: torch.Tensor):
        if isinstance(data, torch.Tensor):  # If array is not masked
            mask = confidence > 0
            data = MaskedTensor(data, torch.stack([mask] * 2, dim=3))

        super().__init__(fps, data, confidence)

    @classmethod
    def read_v0_0(cls, header: PoseHeader, reader: BufferReader):
        raise NotImplementedError("Reading v0 files with torch is not supported")

    def cuda(self):
        self.data = self.data.cuda()
        self.confidence = self.confidence.cuda()

    def zero_filled(self):
        self.data.zero_filled()

    def matmul(self, matrix: np.ndarray):
        data = self.data.matmul(torch.from_numpy(matrix))
        return TorchPoseBody(fps=self.fps, data=data, confidence=self.confidence)

    def points_perspective(self):
        return self.data.permute(POINTS_DIMS)

    def get_points(self, indexes: List[int]):
        data = self.data.permute(POINTS_DIMS)
        new_data = data[indexes].permute(POINTS_DIMS)

        confidence_reshape = (2, 1, 0)
        confidence = self.confidence.permute(confidence_reshape)
        new_confidence = confidence[indexes].permute(confidence_reshape)

        return TorchPoseBody(self.fps, new_data, new_confidence)

    def flatten(self):
        shape = self.data.shape
        data = self.data.tensor.reshape(-1, shape[-1])  # Not masked data
        confidence = torch.tensor(self.confidence.flatten()) # TODO no need to recast ot torch.. but memory access issue
        indexes = torch.tensor(list(np.ndindex(shape[:-1])), dtype=torch.float32, device=data.device)
        flat = torch.cat([indexes, torch.unsqueeze(confidence, dim=1), data], dim=1)
        # Filter data from flat
        flat = flat[confidence != 0.]
        # Scale the first axis by fps
        scalar = torch.ones(len(shape) + shape[-1],device=data.device)
        scalar[0] = 1 / self.fps
        return flat * scalar
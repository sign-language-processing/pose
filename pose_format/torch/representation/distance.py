import torch
from torch import nn

from pose_format.torch.masked.tensor import MaskedTensor
from pose_format.torch.masked.torch import MaskedTorch


class DistanceRepresentation(nn.Module):
    def distance(self, p1s: MaskedTensor, p2s: MaskedTensor) -> MaskedTensor:
        diff = p1s - p2s  # (..., Len, Dims)
        square = diff.pow_(2)
        sum_squares = square.sum(dim=-1)
        return MaskedTorch.sqrt(sum_squares)

    def forward(self, p1s: MaskedTensor, p2s: MaskedTensor) -> torch.Tensor:
        """
        Euclidean distance between two points
        :param p1s: MaskedTensor (Points, Batch, Len, Dims)
        :param p2s: MaskedTensor (Points, Batch, Len, Dims)
        :return: torch.Tensor (Points, Batch, Len)
        """
        return self.distance(p1s, p2s).zero_filled()

import torch
from pose_format.utils.masked_tensor import MaskedTensor
from torch import nn


class DistanceRepresentation(nn.Module):
    def forward(self, p1s: MaskedTensor, p2s: MaskedTensor) -> torch.Tensor:
        """
        Euclidean distance between two points
        :param p1s: MaskedTensor (Points, Batch, Len, Dims)
        :param p2s: MaskedTensor (Points, Batch, Len, Dims)
        :return: torch.Tensor (Points, Batch, Len)
        """
        diff = p1s - p2s  # (Points, Batch, Len, Dims)
        square = diff.pow_(2)
        sum_squares = square.sum(dim=-1).zero_filled()

        return sum_squares.sqrt()

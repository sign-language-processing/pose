import torch
from pose_format.utils.masked_tensor import MaskedTensor
from torch import nn

from ..masked.torch import MaskedTorch


class InnerAngleRepresentation(nn.Module):
    def forward(self, a: MaskedTensor, b: MaskedTensor, c: MaskedTensor) -> torch.Tensor:
        """
        Angle in point b for the triangle <a, b, c>
        :param a: MaskedTensor (Points, Batch, Len, Dims)
        :param b: MaskedTensor (Points, Batch, Len, Dims)
        :param c: MaskedTensor (Points, Batch, Len, Dims)
        :return: torch.Tensor (Points, Batch, Len)
        """
        # Following https://stackoverflow.com/questions/19729831/angle-between-3-points-in-3d-space
        v1 = a - b  # (Points, Batch, Len, Dims)
        v2 = c - b  # (Points, Batch, Len, Dims)

        v1_norm = MaskedTorch.norm(v1, dim=3)
        v2_norm = MaskedTorch.norm(v2, dim=3)

        slopes = MaskedTorch.sum(v1_norm * v2_norm, dim=3).zero_filled()
        slopes[slopes != slopes] = 0  # Fix NaN, TODO think of faster way

        angles = torch.acos(slopes)

        return angles
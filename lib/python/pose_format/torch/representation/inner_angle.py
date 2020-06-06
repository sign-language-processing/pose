import torch
from ..masked.tensor import MaskedTensor
from ..masked.torch import MaskedTorch
from torch import nn


def get_vectors_norm(vectors: MaskedTensor):
    square = MaskedTorch.square(vectors)
    summed = square.sum(dim=3)
    v_mag = MaskedTorch.sqrt(summed)
    v_norm = vectors / v_mag
    return v_norm


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

        v1_norm = get_vectors_norm(v1)
        v2_norm = get_vectors_norm(v2)

        slopes = (v1_norm * v2_norm).sum(dim=3)
        angles = MaskedTorch.acos(slopes)

        angles = angles.zero_filled()
        angles[angles != angles] = 0  # Fix NaN, TODO think of faster way

        return angles
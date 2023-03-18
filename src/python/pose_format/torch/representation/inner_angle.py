import torch
from torch import nn

from ..masked.tensor import MaskedTensor
from ..masked.torch import MaskedTorch


def get_vectors_norm(vectors: MaskedTensor):
    square = MaskedTorch.square(vectors)
    summed = square.sum(dim=-1)
    v_mag = MaskedTorch.sqrt(summed)
    mag_stack = MaskedTorch.stack([v_mag] * vectors.shape[-1], dim=-1)
    return vectors.div(mag_stack)


class InnerAngleRepresentation(nn.Module):
    def forward(self, p1s: MaskedTensor, p2s: MaskedTensor, p3s: MaskedTensor) -> torch.Tensor:
        """
        Angle in point p2s for the triangle <p1s, p2s, c>
        :param p1s: MaskedTensor (Points, Batch, Len, Dims)
        :param p2s: MaskedTensor (Points, Batch, Len, Dims)
        :param p3s: MaskedTensor (Points, Batch, Len, Dims)
        :return: torch.Tensor (Points, Batch, Len)
        """
        # Following https://stackoverflow.com/questions/19729831/angle-between-3-points-in-3d-space
        v1 = p1s - p2s  # (Points, Batch, Len, Dims)
        v2 = p3s - p2s  # (Points, Batch, Len, Dims)

        v1_norm = get_vectors_norm(v1)
        v2_norm = get_vectors_norm(v2)

        slopes = (v1_norm * v2_norm).sum(dim=-1)
        angles = MaskedTorch.acos(slopes)

        angles = angles.zero_filled()
        angles[angles != angles] = 0  # Fix NaN, TODO think of faster way

        return angles

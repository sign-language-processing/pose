import torch
from torch import nn

from ..masked.tensor import MaskedTensor


class PointsRepresentation(nn.Module):
  def forward(self, p1s: MaskedTensor) -> torch.Tensor:
    """
    Angle of the X/Y axis between two points
    :param p1s: MaskedTensor (Points, Batch, Len, Dims)
    :return: torch.Tensor (Points*Dims, Batch, Len)
    """

    p1s = p1s.zero_filled()
    p1s = p1s.transpose(1, 3) # (Points, Dims, Len, Batch)
    p1s = p1s.transpose(2, 3) # (Points, Dims, Batch, Len)
    shape = p1s.shape

    return p1s.view((-1, shape[2], shape[3]))

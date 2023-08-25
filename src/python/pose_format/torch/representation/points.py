import torch
from torch import nn

from ..masked.tensor import MaskedTensor


class PointsRepresentation(nn.Module):
    """
    Class to represent points in a tensor format for processing.
  """

    def forward(self, p1s: MaskedTensor) -> torch.Tensor:
        """
    Transforms input tensor representing points into a desired tensor format.
    
    The transformation process with zero-filling the masked values in input tensor 
    and reshaping tensor by transposing its dimensions to match the desired output format.

    Parameters
    ----------
    p1s : :class:`~pose_format.torch.masked.tensor.MaskedTensor`
        Tensor representing a set of points.
        Shape: (Points, Batch, Len, Dims).

    Returns
    -------
    torch.Tensor
        Transformed tensor representing the points.
        Shape: (Points*Dims, Batch, Len).

    Note
    ----
    This method first fills  masked values in input tensor with zeros.
    Then, it reshapes tensor by transposing dimensions to match its desired output format
    """

        p1s = p1s.zero_filled()
        p1s = p1s.transpose(1, 3)  # (Points, Dims, Len, Batch)
        p1s = p1s.transpose(2, 3)  # (Points, Dims, Batch, Len)
        shape = p1s.shape

        return p1s.view((-1, shape[2], shape[3]))

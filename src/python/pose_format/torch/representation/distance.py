import torch
from torch import nn

from pose_format.torch.masked.tensor import MaskedTensor
from pose_format.torch.masked.torch import MaskedTorch


class DistanceRepresentation(nn.Module):
    """
    Represents the Euclidean distance between two points in space.
    """

    def distance(self, p1s: MaskedTensor, p2s: MaskedTensor) -> MaskedTensor:
        """
        Calculate the Euclidean distance between two sets of points.
        
        Parameters
        ----------
        p1s : :class:`~pose_format.torch.masked.tensor.MaskedTensor`
            Tensor representing the first set of points.
        
        p2s : :class:`~pose_format.torch.masked.tensor.MaskedTensor`
            Tensor representing the second set of points.
        
        Returns
        -------
        :class:`~pose_format.torch.masked.tensor.MaskedTensor`
            Tensor representing the calculated distances.
        """
        diff = p1s - p2s  # (..., Len, Dims)
        square = diff.pow_(2)
        sum_squares = square.sum(dim=-1)
        return MaskedTorch.sqrt(sum_squares)

    def forward(self, p1s: MaskedTensor, p2s: MaskedTensor) -> torch.Tensor:
        """
        Computes Euclidean distance between two sets of points.
        
        Parameters
        ----------
        p1s : :class:`~pose_format.torch.masked.tensor.MaskedTensor`
            Tensor representing the first set of points. Shape: (Points, Batch, Len, Dims).
        
        p2s : :class:`~pose_format.torch.masked.tensor.MaskedTensor`
            Tensor representing the second set of points. Shape: (Points, Batch, Len, Dims).
        
        Returns
        -------
        torch.Tensor
            Tensor representing the Euclidean distances. Shape: (Points, Batch, Len).
        """
        return self.distance(p1s, p2s).zero_filled()

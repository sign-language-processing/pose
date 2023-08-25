import torch
from torch import nn

from ..masked.tensor import MaskedTensor
from ..masked.torch import MaskedTorch
from .distance import DistanceRepresentation


class PointLineDistanceRepresentation(nn.Module):
    """
    Class computing distance between a point and a line segment.
    
    Parameters
    ----------
    distance : :class:`~pose_format.torch.representation.distance.DistanceRepresentation`
        Instance of the `DistanceRepresentation` class to compute the Euclidean distance.

    """

    def __init__(self):
        super(PointLineDistanceRepresentation, self).__init__()
        self.distance = DistanceRepresentation()

    def forward(self, p1s: MaskedTensor, p2s: MaskedTensor, p3s: MaskedTensor) -> torch.Tensor:
        """
        Computes  distance from the point `p1s` to the line formed by points `p2s` and `p3s`.
        
        The method uses Heron's Formula to find the area of the triangle formed by the three points
        and then calculates the height of the triangle to determine the distance from the point 
        `p1s` to the line <p2s, p3s>.
        
        Parameters
        ----------
        p1s : :class:`~pose_format.torch.masked.tensor.MaskedTensor`
            Tensor representing the point for which the distance to the line is calculated.
            Shape: (Points, Batch, Len, Dims).
        
        p2s : :class:`~pose_format.torch.masked.tensor.MaskedTensor`
            Tensor representing one end-point of the line. 
            Shape: (Points, Batch, Len, Dims).
        
        p3s : :class:`~pose_format.torch.masked.tensor.MaskedTensor`
            Tensor representing the other end-point of the line.
            Shape: (Points, Batch, Len, Dims).
        
        Returns
        -------
        torch.Tensor
            Tensor representing the distances from the point `p1s` to the line <p2s, p3s>.
            Shape: (Points, Batch, Len).
        
        Note
        ----
        This is following Heron's Formula: https://en.wikipedia.org/wiki/Heron%27s_formula.
        """
        # Following Heron's Formula https://en.wikipedia.org/wiki/Heron%27s_formula
        a = self.distance.distance(p1s, p2s)
        b = self.distance.distance(p2s, p3s)
        c = self.distance.distance(p1s, p3s)
        s: MaskedTensor = (a + b + c) / 2
        squared = s * (s - a) * (s - b) * (s - c)
        area = MaskedTorch.sqrt(squared)

        # Calc "height" of the triangle
        square_area: MaskedTensor = area * 2
        distance = square_area / b
        distance.fix_nan()

        return distance.zero_filled()

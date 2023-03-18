import torch
from torch import nn

from .distance import DistanceRepresentation
from ..masked.tensor import MaskedTensor
from ..masked.torch import MaskedTorch


class PointLineDistanceRepresentation(nn.Module):
    def __init__(self):
        super(PointLineDistanceRepresentation, self).__init__()
        self.distance = DistanceRepresentation()

    def forward(self, p1s: MaskedTensor, p2s: MaskedTensor, p3s: MaskedTensor) -> torch.Tensor:
        """
        Distance between the point p1s to the line <p2s, p3s>
        :param p1s: MaskedTensor (Points, Batch, Len, Dims)
        :param p2s: MaskedTensor (Points, Batch, Len, Dims)
        :param p3s: MaskedTensor (Points, Batch, Len, Dims)
        :return: torch.Tensor (Points, Batch, Len)
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

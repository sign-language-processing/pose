import torch
from torch import nn

from ..masked.tensor import MaskedTensor


class AngleRepresentation(nn.Module):
    """
    Class to compute the angle between the X/Y axis and the line segments formed by two sets of points.
    """

    def forward(self, p1s: MaskedTensor, p2s: MaskedTensor) -> torch.Tensor:
        """
        Computes angle in radians between X/Y axis and line segments made by two sets of points.
        
        Parameters
        ----------
        p1s : :class:`~pose_format.torch.masked.tensor.MaskedTensor`
            A tensor representing the first set of points with shape (Points, Batch, Len, Dims).
        
        p2s : :class:`~pose_format.torch.masked.tensor.MaskedTensor`
            A tensor representing the second set of points with the same shape as `p1s`.
        
        Returns
        -------
        torch.Tensor
            A tensor of angles (in radians) with shape (Points, Batch, Len).
        
        Note
        ----
        The slope is determined for each pair of points. The arctangent function is then applied to calculate the angle in radians.
        """
        dims = p1s.shape[-1]

        d = p2s - p1s  # (Points, Batch, Len, Dims)
        xs, ys = d.split([1] * dims, dim=3)[:2]  # (Points, Batch, Len, 1)
        slopes = ys.div(xs).fix_nan().zero_filled().squeeze(axis=3)

        return torch.atan(slopes)


if __name__ == "__main__":
    representation = AngleRepresentation()
    p1s = MaskedTensor(torch.tensor([[[[1, 2, 3]]]], dtype=torch.float))
    print(p1s.shape)

    p2s = MaskedTensor(torch.tensor([[[[4, 5, 6]]]], dtype=torch.float))
    angles = representation(p1s, p2s)
    print(angles)

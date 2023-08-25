import torch
from torch import nn

from ..masked.tensor import MaskedTensor
from ..masked.torch import MaskedTorch


def get_vectors_norm(vectors: MaskedTensor):
    """
    Computes the normalized vectors from the given masked vectors.

    Parameters
    ----------

    vectors : :class:`~pose_format.torch.masked.tensor.MaskedTensor`
        The input masked vectors with any shape.

    Returns
    -------
    :class:`~pose_format.torch.masked.tensor.MaskedTensor`
        The normalized masked vectors of the same shape as the input.
    Notes
    -----
    The function squares the input vectors, then sums along the last dimension. 
    Taking the square root of the sum provides the magnitude. The original vectors 
    are then divided by this magnitude to normalize.
    """
    square = MaskedTorch.square(vectors)
    summed = square.sum(dim=-1)
    v_mag = MaskedTorch.sqrt(summed)
    mag_stack = MaskedTorch.stack([v_mag] * vectors.shape[-1], dim=-1)
    return vectors.div(mag_stack)


class InnerAngleRepresentation(nn.Module):
    """
    A neural network module to compute the inner angle at a point for a triangle.
    """

    def forward(self, p1s: MaskedTensor, p2s: MaskedTensor, p3s: MaskedTensor) -> torch.Tensor:
        """
        Computes the angle in point `p2s` for the triangle defined by the points <p1s, p2s, p3s>.
        
        Parameters
        ----------
        p1s : :class:`~pose_format.torch.masked.tensor.MaskedTensor`
            A tensor representing the first set of points, with shape (Points, Batch, Len, Dims).
        p2s : :class:`~pose_format.torch.masked.tensor.MaskedTensor`
            A tensor representing the second set of points (at which the angle is calculated), with shape (Points, Batch, Len, Dims).
        p3s : :class:`~pose_format.torch.masked.tensor.MaskedTensor`
            A tensor representing the third set of points, with shape (Points, Batch, Len, Dims).

        Returns
        -------
        torch.Tensor
            A tensor representing the computed angles at point `p2s`, with shape (Points, Batch, Len).

        Note
        ----
        The method is based on the approach suggested in: 
        https://stackoverflow.com/questions/19729831/angle-between-3-points-in-3d-space
        
        The function first computes the vectors v1 and v2 by subtracting points p1s and p3s 
        from p2s, respectively. The vectors are then normalized. The angle is calculated by 
        finding the arccosine of the dot product of the normalized vectors. NaN values in 
        the resulting tensor are set to zero.
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

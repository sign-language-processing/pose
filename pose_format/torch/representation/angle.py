import torch
from torch import nn

from ..masked.tensor import MaskedTensor


class AngleRepresentation(nn.Module):
    def forward(self, p1s: MaskedTensor, p2s: MaskedTensor) -> torch.Tensor:
        """
        Angle of the X/Y axis between two points
        :param p1s: MaskedTensor (Points, Batch, Len, Dims)
        :param p2s: MaskedTensor (Points, Batch, Len, Dims)
        :return: torch.Tensor (Points, Batch, Len)
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

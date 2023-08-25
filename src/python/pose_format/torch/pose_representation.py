from typing import List

import torch

from ..pose_header import PoseHeader
from ..pose_representation import PoseRepresentation


class TorchPoseRepresentation(PoseRepresentation):
    """
    TorchPoseRepresentation class representing pose information using PyTorch tensors.

    This class extends the PoseRepresentation class and provides methods for manipulating and representing pose data
    using PyTorch tensors.

    Parameters
    ----------
    header : PoseHeader
        Header describing the pose data structure.
    rep_modules1 : List
        List of additional representation modules (level 1) to apply to pose data.
    rep_modules2 : List
        List of additional representation modules (level 2) to apply to pose data.
    rep_modules3 : List
        List of additional representation modules (level 3) to apply to pose data.
    """

    def __init__(self, header: PoseHeader, rep_modules1: List = [], rep_modules2: List = [], rep_modules3: List = []):
        super(TorchPoseRepresentation, self).__init__(header, rep_modules1, rep_modules2, rep_modules3)

        # Change limb points to torch
        self.limb_pt1s = torch.tensor(self.limb_pt1s, dtype=torch.long)
        self.limb_pt2s = torch.tensor(self.limb_pt2s, dtype=torch.long)

        # Change triangle points to torch
        self.triangle_pt1s = torch.tensor(self.triangle_pt1s, dtype=torch.long)
        self.triangle_pt2s = torch.tensor(self.triangle_pt2s, dtype=torch.long)
        self.triangle_pt3s = torch.tensor(self.triangle_pt3s, dtype=torch.long)

    def group_embeds(self, embeds: List[torch.Tensor]):
        """
        Group and reshape embedded tensors for batch processing.

        Parameters
        ----------
        embeds : List[torch.Tensor]
            List of embedded tensors of size (embed_size, Batch, Len).

        Returns
        -------
        torch.Tensor
            A tensor of size (Batch, Len, embed_size) with grouped and reshaped embedded tensors.

        """
        group = torch.cat(embeds, dim=0)  # (embed_size, Batch, Len)
        return group.permute(dims=[1, 2, 0])

    def permute(self, src, shape: tuple):
        """
        Permute dimensions of tensor according to a specified shape (tuple).

        Parameters
        ----------
        src : torch.Tensor
            tensor to  permute
        shape : tuple
            desired shape of the tensor after permutation.

        Returns
        -------
        torch.Tensor
            tensor with permuted dimensions according to specified shape.

        """
        return src.permute(shape)

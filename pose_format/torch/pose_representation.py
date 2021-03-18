from typing import List

import torch

from ..pose_header import PoseHeader
from ..pose_representation import PoseRepresentation


class TorchPoseRepresentation(PoseRepresentation):
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
        :param embeds: torch.Tensor List of tensors size (embed_size, Batch, Len)
        :return: Size (Batch, Len, embed_size)
        """
        group = torch.cat(embeds, dim=0)  # (embed_size, Batch, Len)
        return group.permute(dims=[1, 2, 0])

    def permute(self, src, shape: tuple):
        return src.permute(shape)

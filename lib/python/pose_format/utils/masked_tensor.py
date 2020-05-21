from typing import Tuple, List, Union

import torch


class MaskedTensor:
    def __init__(self, tensor: torch.Tensor, mask: torch.Tensor = None):
        self.tensor = tensor
        self.mask = mask if mask is not None else torch.ones(tensor.shape, dtype=torch.bool).to(tensor.device)

        self.device = tensor.device

    def __len__(self):
        return self.tensor.shape[0]

    def __getitem__(self, key):
        tensor = self.tensor[key]
        mask = self.mask[key]
        return MaskedTensor(tensor=tensor, mask=mask)

    def __sub__(self, other: "MaskedTensor"):
        tensor = self.tensor - other.tensor
        mask = self.mask & other.mask
        return MaskedTensor(tensor=tensor, mask=mask)

    def __mul__(self, other: torch.Tensor):
        tensor = self.tensor * other
        return MaskedTensor(tensor=tensor, mask=self.mask)

    def pow_(self, exponent: float):
        self.tensor.pow_(exponent)
        return self

    def sum(self, dim: int):
        tensor = self.tensor.sum(dim=dim)
        mask = self.mask.prod(dim=dim)
        return MaskedTensor(tensor=tensor, mask=mask)

    @property
    def shape(self):
        return self.tensor.shape

    def cuda(self, device=None, non_blocking: bool = False):
        tensor = self.tensor.cuda(device=device, non_blocking=non_blocking)
        mask = self.mask.cuda(device=device, non_blocking=non_blocking)
        return MaskedTensor(tensor=tensor, mask=mask)

    def zero_filled(self) -> torch.Tensor:
        return self.tensor.mul_(self.mask)  # In place multiplication

    def div(self, other: "MaskedTensor", in_place=False, update_mask=True):
        tensor = torch.div(self.tensor, other.tensor, out=self.tensor if in_place else None)
        mask = self.mask & other.mask if update_mask else self.mask
        return MaskedTensor(tensor, mask)

    def matmul(self, matrix: torch.Tensor):
        tensor = torch.matmul(self.tensor, matrix.to(self.device))
        return MaskedTensor(tensor, self.mask)

    def transpose(self, dim0: int, dim1: int):
        tensor = self.tensor.transpose(dim0, dim1)
        mask = self.mask.transpose(dim0, dim1)
        return MaskedTensor(tensor=tensor, mask=mask)

    def permute(self, dims: tuple):
        tensor = self.tensor.permute(dims)
        mask = self.mask.permute(dims)
        return MaskedTensor(tensor=tensor, mask=mask)

    def squeeze(self, dim: int):
        tensor = self.tensor.squeeze(dim)
        mask = self.mask.squeeze(dim)
        return MaskedTensor(tensor=tensor, mask=mask)

    def split(self, split_size_or_sections, dim=0):
        tensors = torch.split(self.tensor, split_size_or_sections, dim)
        masks = torch.split(self.mask, split_size_or_sections, dim)
        return [MaskedTensor(tensor=tensor, mask=mask) for tensor, mask in zip(tensors, masks)]

    def reshape(self, shape: tuple):
        tensor = self.tensor.reshape(shape=shape)
        mask = self.mask.reshape(shape=shape)
        return MaskedTensor(tensor=tensor, mask=mask)


class MaskedTorch:
    @staticmethod
    def cat(tensors: List[Union[MaskedTensor, torch.Tensor]], dim: int):
        tensors: List[MaskedTensor] = [t if isinstance(t, MaskedTensor) else MaskedTensor(tensor=t) for t in tensors]
        tensor = torch.cat([t.tensor for t in tensors], dim=dim)
        mask = torch.cat([t.mask for t in tensors], dim=dim)
        return MaskedTensor(tensor=tensor, mask=mask)

    @staticmethod
    def stack(tensors: List[MaskedTensor], dim: int):
        tensor = torch.stack([t.tensor for t in tensors], dim=dim)
        mask = torch.stack([t.mask for t in tensors], dim=dim)
        return MaskedTensor(tensor=tensor, mask=mask)

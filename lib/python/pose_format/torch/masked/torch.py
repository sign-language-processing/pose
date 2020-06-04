import torch
from typing import List, Union

from .tensor import MaskedTensor


class TorchFallback(type):
    def __getattr__(cls, attr):
        def func(*args, **kwargs):
            if len(args) > 0 and isinstance(args[0], MaskedTensor):
                args = list(args)
                args[0] = args[0].tensor
            return getattr(torch, attr)(*args, **kwargs)

        return func


class MaskedTorch(metaclass=TorchFallback):
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

    @staticmethod
    def zeros(*size, dtype=None):
        tensor = torch.zeros(*size, dtype=dtype)
        mask = torch.zeros(*size, dtype=torch.bool)
        return MaskedTensor(tensor=tensor, mask=mask)
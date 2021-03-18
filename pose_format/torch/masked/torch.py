from typing import List, Union

import torch

from pose_format.torch.masked.tensor import MaskedTensor


class TorchFallback(type):
    doesnt_change_mask = {
        "sqrt", "square", "unsqueeze",
        "cos", "sin", "tan", "acos", "asin", "atan"
    }

    def __getattr__(cls, attr):
        def func(*args, **kwargs):
            if len(args) > 0 and isinstance(args[0], MaskedTensor):
                args = list(args)
                mask = args[0].mask
                args[0] = args[0].tensor

                res = getattr(torch, attr)(*args, **kwargs)
                if attr in TorchFallback.doesnt_change_mask:
                    return MaskedTensor(res, mask)
                else:
                    return res

            else:  # If this action is done on an unmasked tensor
                return getattr(torch, attr)(*args, **kwargs)

        return func


class MaskedTorch(metaclass=TorchFallback):
    @staticmethod
    def cat(tensors: List[Union[MaskedTensor, torch.Tensor]], dim: int) -> MaskedTensor:
        tensors: List[MaskedTensor] = [t if isinstance(t, MaskedTensor) else MaskedTensor(tensor=t) for t in tensors]
        tensor = torch.cat([t.tensor for t in tensors], dim=dim)
        mask = torch.cat([t.mask for t in tensors], dim=dim)
        return MaskedTensor(tensor=tensor, mask=mask)

    @staticmethod
    def stack(tensors: List[MaskedTensor], dim: int) -> MaskedTensor:
        tensor = torch.stack([t.tensor for t in tensors], dim=dim)
        mask = torch.stack([t.mask for t in tensors], dim=dim)
        return MaskedTensor(tensor=tensor, mask=mask)

    @staticmethod
    def zeros(*size, dtype=None) -> MaskedTensor:
        tensor = torch.zeros(*size, dtype=dtype)
        mask = torch.zeros(*size, dtype=torch.bool)
        return MaskedTensor(tensor=tensor, mask=mask)

    @staticmethod
    def squeeze(masked_tensor: MaskedTensor) -> MaskedTensor:
        tensor = torch.squeeze(masked_tensor.tensor)
        mask = torch.squeeze(masked_tensor.mask)
        return MaskedTensor(tensor=tensor, mask=mask)

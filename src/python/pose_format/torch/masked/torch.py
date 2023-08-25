from typing import List, Union

import torch

from pose_format.torch.masked.tensor import MaskedTensor


class TorchFallback(type):
    """Meta class that gives a fallback mechanism to use torch functions on :class:`~pose_format.torch.masked.tensor.MaskedTensor` objects. :noindex:"""
    doesnt_change_mask = {"sqrt", "square", "unsqueeze", "cos", "sin", "tan", "acos", "asin", "atan"}

    def __getattr__(cls, attr):
        """
        Redirects calls to PyTorch functions to handle :class:`~pose_format.torch.masked.tensor.MaskedTensor` instances.

        If the first argument is a :class:`~pose_format.torch.masked.tensor.MaskedTensor`, its mask is taken into account.
        """

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
    """class mimicing torch functions and giving  support for :class:`~pose_format.torch.masked.tensor.MaskedTensor`."""

    @staticmethod
    def cat(tensors: List[Union[MaskedTensor, torch.Tensor]], dim: int) -> MaskedTensor:
        """
        Concatenate :class:`~pose_format.torch.masked.tensor.MaskedTensor` objects along a specified dimension.

        Parameters
        ----------
        tensors : list
            List of tensors or :class:`~pose_format.torch.masked.tensor.MaskedTensor` objects to be concatenated.
        dim : int
            Dimension along to concatenate.

        Returns
        -------
        :class:`~pose_format.torch.masked.tensor.MaskedTensor`
            Concatenated tensor.
        """
        tensors: List[MaskedTensor] = [t if isinstance(t, MaskedTensor) else MaskedTensor(tensor=t) for t in tensors]
        tensor = torch.cat([t.tensor for t in tensors], dim=dim)
        mask = torch.cat([t.mask for t in tensors], dim=dim)
        return MaskedTensor(tensor=tensor, mask=mask)

    @staticmethod
    def stack(tensors: List[MaskedTensor], dim: int) -> MaskedTensor:
        """
        Stack :class:`~pose_format.torch.masked.tensor.MaskedTensor` objects along a new dimension.

        Parameters
        ----------
        tensors : list
            List of :class:`~pose_format.torch.masked.tensor.MaskedTensor` objects to be stacked.
        dim : int
            New dimension along which to stack.

        Returns
        -------
        :class:`~pose_format.torch.masked.tensor.MaskedTensor`
            Stacked maked tensor.
        """
        tensor = torch.stack([t.tensor for t in tensors], dim=dim)
        mask = torch.stack([t.mask for t in tensors], dim=dim)
        return MaskedTensor(tensor=tensor, mask=mask)

    @staticmethod
    def zeros(*size, dtype=None) -> MaskedTensor:
        """
        Creates a :class:`~pose_format.torch.masked.tensor.MaskedTensor` of zeros with a given shape and data type.

        Parameters
        ----------
        *size : ints
            Dimensions of desired tensor.
        dtype : torch.dtype, optional
            Data type of the tensor. If None, defaults to `torch.float`.

        Returns
        -------
        :class:`~pose_format.torch.masked.tensor.MaskedTensor`
            masked tensor filled with zeros.
        """
        tensor = torch.zeros(*size, dtype=dtype)
        mask = torch.zeros(*size, dtype=torch.bool)
        return MaskedTensor(tensor=tensor, mask=mask)

    @staticmethod
    def squeeze(masked_tensor: MaskedTensor) -> MaskedTensor:
        """
        Remove dimensions of size 1 from :class:`~pose_format.torch.masked.tensor.MaskedTensor`.

        Parameters
        ----------
        masked_tensor : :class:`~pose_format.torch.masked.tensor.MaskedTensor`
            tensor from which dimensions are to be removed.

        Returns
        -------
        :class:`~pose_format.torch.masked.tensor.MaskedTensor`
            Squeezed masked tensor.
        """
        tensor = torch.squeeze(masked_tensor.tensor)
        mask = torch.squeeze(masked_tensor.mask)
        return MaskedTensor(tensor=tensor, mask=mask)

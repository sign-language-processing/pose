import torch


class MaskedTensor:
    """
    Container for a PyTorch tensor, providing utility functions for tensor masking.

    Parameters
    ----------
    tensor : torch.Tensor
        Tensor data.
    mask : torch.Tensor, optional
        A boolean mask tensor of the same shape as `tensor`. If specified, elements 
        of `tensor` corresponding to `True` values in the mask are considered valid. 
        Defaults to a tensor of all `True` values.
    """

    def __init__(self, tensor: torch.Tensor, mask: torch.Tensor = None):
        self.tensor = tensor
        self.mask = mask if mask is not None else torch.ones(tensor.shape, dtype=torch.bool).to(tensor.device)

    def __getattr__(self, item):
        """
        Gets attributes of tensor.

        Raises
        ------
        NotImplementedError
            If called attribute is not implemented.
        """
        val = self.tensor.__getattribute__(item)
        if hasattr(val, '__call__'):  # If is a function
            # return getattr(MaskedTorch, item)(self)
            raise NotImplementedError("callbable '%s' not defined" % item)
        else:
            return val

    def __len__(self):
        """
        Gets size of first dimension of the tensor.

        Returns
        -------
        int
            Size of first dimension of tensor.
        """
        return self.tensor.shape[0]

    def __getitem__(self, key):
        """
        Get a subset of a tensor based on a key or slice.

        Returns
        -------
        :class:`~pose_format.torch.masked.tensor.MaskedTensor`
            Subset of the tensor.
        """
        tensor = self.tensor[key]
        mask = self.mask[key]
        return MaskedTensor(tensor=tensor, mask=mask)

    def arithmetic(self, action: str, other):
        """
        Helper method to perform arithmetic operations on tensors.

        Parameters
        ----------
        action : str
            The arithmetic operation to be performed.
        other : Union[~pose_format.torch.masked.tensor.MaskedTensor`, torch.Tensor, float, int]
            The second operand.

        Returns
        -------
        :class:`~pose_format.torch.masked.tensor.MaskedTensor`
            New `MaskedTensor` after the operation.
        """
        if isinstance(other, MaskedTensor):
            tensor = getattr(self.tensor, action)(other.tensor)
            mask = self.mask & other.mask
        else:
            tensor = getattr(self.tensor, action)(other)
            mask = self.mask
        return MaskedTensor(tensor=tensor, mask=mask)

    def __add__(self, other):
        """
        Performs element-wise addition with another tensor or scalar.

        Parameters
        ----------
        other : Union[:class:`~pose_format.torch.masked.tensor.MaskedTensor`, torch.Tensor, float, int]
            The tensor or scalar to add.

        Returns
        -------
        :class:`~pose_format.torch.masked.tensor.MaskedTensor`
            Resultant tensor after addition.
        """
        return self.arithmetic("__add__", other)

    def __sub__(self, other):
        """
        Performs element-wise subtraction with another tensor or scalar.

        Parameters
        ----------
        other : Union[:class:`~pose_format.torch.masked.tensor.MaskedTensor`, torch.Tensor, float, int]
            The tensor or scalar to subtract.

        Returns
        -------
        :class:`~pose_format.torch.masked.tensor.MaskedTensor`
            Resultant tensor after subtraction.
        """
        return self.arithmetic("__sub__", other)

    def __mul__(self, other):
        """
        Performs element-wise multiplication with another tensor or scalar.

        Parameters
        ----------
        other : Union[:class:`~pose_format.torch.masked.tensor.MaskedTensor`, torch.Tensor, float, int]
            The tensor or scalar to multiply.

        Returns
        -------
        :class:`~pose_format.torch.masked.tensor.MaskedTensor`
            Resultant tensor after multiplication.
        """
        return self.arithmetic("__mul__", other)

    def __truediv__(self, other):
        """
        Performs element-wise division with another tensor or scalar.

        Parameters
        ----------
        other : Union[:class:`~pose_format.torch.masked.tensor.MaskedTensor`, torch.Tensor, float, int]
            The tensor or scalar to divide by.

        Returns
        -------
        :class:`~pose_format.torch.masked.tensor.MaskedTensor`
            Resultant tensor after division.
        """
        return self.arithmetic("__truediv__", other)

    def __eq__(self, other):
        """
        Compares the tensor for element-wise equality with another tensor.

        Parameters
        ----------
        other : torch.Tensor
            The tensor to compare.

        Returns
        -------
        torch.Tensor
            A boolean tensor with `True` where elements are equal and `False` otherwise.
        """
        return self.tensor == other

    def pow_(self, exponent: float):
        """
        Raises tensor to power of a given exponent in-place.

        Parameters
        ----------
        exponent : float
            The exponent value.

        Returns
        -------
        :class:`~pose_format.torch.masked.tensor.MaskedTensor`
            Masked tensor raised to a given exponent.
        """
        self.tensor.pow_(exponent)
        return self

    def sum(self, dim: int):
        """
        Sums along a specified dimension.

        Parameters
        ----------
        dim : int
            dimension to sum over.

        Returns
        -------
        :class:`~pose_format.torch.masked.tensor.MaskedTensor`
            Summed tensor along the specified dimension.
        """
        tensor = self.tensor.sum(dim=dim)
        mask = self.mask.prod(dim=dim).bool()
        return MaskedTensor(tensor=tensor, mask=mask)

    def size(self, *args):
        """
        Get size of tensor for specified dimensions.

        Returns
        -------
        torch.Size
            Size of tensor.
        """
        return self.tensor.size(*args)

    def fix_nan(self):  # TODO think of faster way
        """
        Replaces any NaN values in the tensor with zeros.

        Returns
        -------
        :class:`~pose_format.torch.masked.tensor.MaskedTensor`
            Tensor with NaN values replaced by zeros.
        """
        self.tensor[self.tensor != self.tensor] = 0
        return self

    def to(self, device):
        """
        Moves tensor to a custom device.

        Parameters
        ----------
        device : str or torch.device
            The target device.

        Returns
        -------
        :class:`~pose_format.torch.masked.tensor.MaskedTensor`
            Tensor on the other device.
        """
        tensor = self.tensor.to(device)
        mask = self.mask.to(device)
        return MaskedTensor(tensor=tensor, mask=mask)

    def cuda(self, device=None, non_blocking: bool = False):
        """
        Moves tensor to the GPU.

        Parameters
        ----------
        device : str or torch.device, optional
            The target CUDA device.
        non_blocking : bool, optional
            Whether to perform an operation asynchronously. Default is False.

        Returns
        -------
        :class:`~pose_format.torch.masked.tensor.MaskedTensor`
            Tensor on CUDA device.
        """
        tensor = self.tensor.cuda(device=device, non_blocking=non_blocking)
        mask = self.mask.cuda(device=device, non_blocking=non_blocking)
        return MaskedTensor(tensor=tensor, mask=mask)

    def zero_filled(self) -> torch.Tensor:
        """
        Get tensor with masked values set to zero.

        Returns
        -------
        torch.Tensor
            Tensor with masked values set to zero.
        """
        return self.tensor.mul(self.mask)

    def div(self, other: "MaskedTensor", in_place=False, update_mask=True):
        """
        Performs element-wise division with another tensor.

        Parameters
        ----------
        other : :class:`~pose_format.torch.masked.tensor.MaskedTensor`
            The tensor to divide with.
        in_place : bool, optional
            If True, performs the operation in-place. Default is False.
        update_mask : bool, optional
            If True, updates the mask after division. Default is True.

        Returns
        -------
        :class:`~pose_format.torch.masked.tensor.MaskedTensor`
            Resultant tensor after division.
        """
        tensor = torch.div(self.tensor, other.tensor, out=self.tensor if in_place else None)
        mask = self.mask & other.mask if update_mask else self.mask
        return MaskedTensor(tensor, mask)

    def matmul(self, matrix: torch.Tensor):
        """
        Perform matrix multiplication.

        Parameters
        ----------
        matrix : torch.Tensor
            matrix to multiply with.

        Returns
        -------
        :class:`~pose_format.torch.masked.tensor.MaskedTensor`
            New masked tensor after multiplication.
        """
        tensor = torch.matmul(self.tensor, matrix.to(self.device))
        return MaskedTensor(tensor, self.mask)

    def transpose(self, dim0, dim1):
        """
        Transposes tensor along two dimensions.

        Parameters
        ----------
        dim0, dim1 : int
            Two dimensions to which to transpose.

        Returns
        -------
        :class:`~pose_format.torch.masked.tensor.MaskedTensor`
            Transposed masked tensor.
        """
        tensor = self.tensor.transpose(dim0, dim1)
        mask = self.mask.transpose(dim0, dim1)
        return MaskedTensor(tensor=tensor, mask=mask)

    def permute(self, dims: tuple):
        """
        Permute dimensions of tensor.

        Parameters
        ----------
        dims : tuple
            Desired ordering of dimensions.

        Returns
        -------
        :class:`~pose_format.torch.masked.tensor.MaskedTensor`
            Permuted masked tensor.
        """
        tensor = self.tensor.permute(dims)
        mask = self.mask.permute(dims)
        return MaskedTensor(tensor=tensor, mask=mask)

    def squeeze(self, dim):
        """
        Squeeze tensor along chosen dimension.

        Parameters
        ----------
        dim : int
            Dimension to squeeze.

        Returns
        -------
        :class:`~pose_format.torch.masked.tensor.MaskedTensor`
            Squeezed masked tensor.
        """
        tensor = self.tensor.squeeze(dim)
        mask = self.mask.squeeze(dim)
        return MaskedTensor(tensor=tensor, mask=mask)

    def split(self, split_size_or_sections, dim=0):
        """
        Split tensor into multiple tensors.

        Parameters
        ----------
        split_size_or_sections : int or tuple
            Size or sections to split tensor.
        dim : int, optional
            Dimension along which to split tensor. Default is 0.

        Returns
        -------
        list[:class:`~pose_format.torch.masked.tensor.MaskedTensor`]
            List of split tensors.
        """
        tensors = torch.split(self.tensor, split_size_or_sections, dim)
        masks = torch.split(self.mask, split_size_or_sections, dim)
        return [MaskedTensor(tensor=tensor, mask=mask) for tensor, mask in zip(tensors, masks)]

    def reshape(self, shape: tuple):
        """
        Reshape tensor to given shape.

        Parameters
        ----------
        shape : tuple
            Desired shape.

        Returns
        -------
        :class:`~pose_format.torch.masked.tensor.MaskedTensor`
            Reshaped tensor.
        """
        tensor = self.tensor.reshape(shape=shape)
        mask = self.mask.reshape(shape=shape)
        return MaskedTensor(tensor=tensor, mask=mask)

    def rename(self, *names):
        """
        Rename tensor's dimensions.

        Parameters
        ----------
        names : tuple
            Desired names for each dimension.

        Returns
        -------
        :class:`~pose_format.torch.masked.tensor.MaskedTensor`
            Renamed masked tensor.
        """
        tensor = self.tensor.rename(*names)
        mask = self.mask.rename(*names)
        return MaskedTensor(tensor=tensor, mask=mask)

    def __getstate__(self):
        return vars(self)

    def __setstate__(self, state):
        vars(self).update(state)

from typing import List

import tensorflow as tf


class MaskedTensor:

    def __init__(self, tensor: tf.Tensor, mask: tf.Tensor = None):

        self.tensor = tensor
        self.mask = mask if mask is not None else tf.ones(tensor.shape, dtype=tf.bool)  # .to(tensor.device)

    def __getattr__(self, item):
        """
        Get attributes from the tensor, unless it's a callable in which case an error is raised.

        Parameters
        ----------
        item : str
            Name of the attribute to fetch.

        Raises
        ------
        NotImplementedError
            If the requested attribute is callable.
        """
        val = self.tensor.__getattribute__(item)
        if hasattr(val, '__call__'):  # If is a function
            raise NotImplementedError("callable '%s' not defined" % item)
        else:
            return val

    def __len__(self):
        """
        Return the length of the tensor.

        Returns
        -------
        int
            Length of the tensor along the first dimension.

        """
        shape = self.tensor.shape
        return shape[0] if len(shape) > 0 else 1

    def __getitem__(self, key):
        """
        Get elements from tensor and corresponding mask based on a key.

        Parameters
        ----------
        key : list or int or slice or tf.Tensor
            Indexing key used to get the elements.

        Returns
        -------
        :class:`pose_format.tensorflow.masked.tensor.MaskedTensor`
            A new MaskedTensor containing elements selected by the indexing key.

        """
        if isinstance(key, list):
            tensor = tf.gather(self.tensor, key)
            mask = tf.gather(self.mask, key)
        else:
            tensor = self.tensor[key]
            mask = self.mask[key]
        return MaskedTensor(tensor=tensor, mask=mask)

    def arithmetic(self, action: str, other):
        """
        For element-wise arithmetic operations with another tensor.

        Parameters
        ----------
        action : str
            Name of the arithmetic operation
        other : :class:`pose_format.tensorflow.masked.tensor.MaskedTensor` or tf.Tensor
            Tensor or MaskedTensor to perform the operation with.

        Returns
        -------
        :class:`pose_format.tensorflow.masked.tensor.MaskedTensor`
            A new MaskedTensor containing the result of the arithmetic operation.

        """
        if isinstance(other, MaskedTensor):
            tensor = getattr(self.tensor, action)(other.tensor)
            mask = self.mask & other.mask
        else:
            tensor = getattr(self.tensor, action)(other)
            mask = self.mask
        return MaskedTensor(tensor=tensor, mask=mask)

    def __float__(self):
        return float(self.tensor)

    def __add__(self, other):
        return self.arithmetic("__add__", other)

    def __sub__(self, other):
        return self.arithmetic("__sub__", other)

    def __mul__(self, other):
        return self.arithmetic("__mul__", other)

    def __truediv__(self, other):
        return self.arithmetic("__truediv__", other)

    def __rtruediv__(self, other):
        return self.arithmetic("__rtruediv__", other)

    def __eq__(self, other):
        other_tensor = other.tensor if isinstance(other, MaskedTensor) else other
        return self.tensor == other_tensor

    def __pow__(self, power):
        return self.arithmetic("__pow__", power)

    def __round__(self, ndigits):
        multiplier = tf.constant(10**ndigits, dtype=tf.float32)
        return tf.round(self.tensor * multiplier) / multiplier

    def square(self):
        """
        Element-wise square of the tensor.

        Returns
        -------
        :class:`pose_format.tensorflow.masked.tensor.MaskedTensor`
            A new MaskedTensor containing the squared values of the original tensor.

        """
        tensor = tf.math.square(self.tensor)
        return MaskedTensor(tensor=tensor, mask=self.mask)

    def float(self):
        """
        Convert tensor's data type to float32 while preserving mask.

        Returns
        -------
        :class:`pose_format.tensorflow.masked.tensor.MaskedTensor`
            A new MaskedTensor with the tensor's data type converted to float32.

        """
        tensor = tf.cast(self.tensor, dtype=tf.float32)
        return MaskedTensor(tensor=tensor, mask=self.mask)

    def sqrt(self):
        """
        Element-wise square root of the tensor

        Returns
        -------
        :class:`pose_format.tensorflow.masked.tensor.MaskedTensor`
            A new MaskedTensor containing the square root values of the original tensor.

        """
        tensor = tf.math.sqrt(self.tensor)
        return MaskedTensor(tensor=tensor, mask=self.mask)

    def sum(self, axis):
        """
        Sum of tensor along specified axis while updating mask.

        Parameters
        ----------
        axis : int or None
            Axis along which to compute sum. If None, compute the sum over all elements.

        Returns
        -------
        :class:`pose_format.tensorflow.masked.tensor.MaskedTensor`
            A new MaskedTensor containing the sums of the tensor along the specified axis.

        """
        tensor = tf.math.reduce_sum(self.tensor, axis=axis)
        mask = tf.cast(tf.math.reduce_prod(tf.cast(self.mask, tf.int32), axis=axis), tf.bool)
        return MaskedTensor(tensor=tensor, mask=mask)

    def size(self, *args):
        """
        Get tensor's size along dimensions.

        Parameters
        ----------
        *args : int
            Dimensions for which to get size

        Returns
        -------
        int or tuple of int
            Size of tensor of specified dimensions.

        """
        return self.tensor.size(*args)

    def fix_nan(self):
        """
        Replace NaN values with zeros while keeping mask.

        Returns
        -------
        :class:`pose_format.tensorflow.masked.tensor.MaskedTensor`
            New MaskedTensor with NaN values replaced by zeros.

        """
        self.tensor = tf.where(tf.math.is_finite(self.tensor), self.tensor, tf.zeros_like(self.tensor))
        return self

    def zero_filled(self) -> tf.Tensor:
        """
        Fill invalid values (as indicated by the mask) with zeros.

        Returns
        -------
        tf.Tensor
            Tensor with the same shape as `self.tensor` but with zeros where the mask is False.
        """
        return self.tensor * tf.cast(self.mask, dtype=self.tensor.dtype)

    def div(self, other: "MaskedTensor", in_place=False, update_mask=True) -> "MaskedTensor":
        """
        Divide tensor by another tensor.

        Parameters
        ----------
        other : :class:`pose_format.tensorflow.masked.tensor.MaskedTensor`
            The divisor tensor.
        in_place : bool, optional
            Whether to do division in place. Default is False.
        update_mask : bool, optional
            Whether to update mask after division. Default is True.

        Returns
        -------
        :class:`pose_format.tensorflow.masked.tensor.MaskedTensor`
            Masked tensor after division.
        """
        tensor = tf.div(self.tensor, other.tensor, out=self.tensor if in_place else None)
        mask = self.mask & other.mask if update_mask else self.mask
        return MaskedTensor(tensor, mask)

    def matmul(self, matrix: tf.Tensor) -> "MaskedTensor":
        """
        Matrix multiplication a given matrix.

    Parameters
    ----------
    matrix : tf.Tensor
        Matrix to perform multiplication with.

    Returns
    -------
    :class:`pose_format.tensorflow.masked.tensor.MaskedTensor`
        MaskedTensor` with result of matrix multiplication.

    """
        tensor = tf.matmul(self.tensor, matrix)
        return MaskedTensor(tensor=tensor, mask=self.mask)

    def transpose(self, perm: List[int]) -> "MaskedTensor":
        """
        Transpose tensor according to given permutation.

        Parameters
        ----------
        perm : List[int]
            The new order of dimensions/permutation after transposition.

        Returns
        -------
        :class:`pose_format.tensorflow.masked.tensor.MaskedTensor`
            MaskedTensor with dimensions transposed according to the given permutation.

        """
        tensor = tf.transpose(self.tensor, perm=perm)
        mask = tf.transpose(self.mask, perm=perm)
        return MaskedTensor(tensor=tensor, mask=mask)

    def permute(self, dims: tuple) -> "MaskedTensor":
        """ Permute the dimensions of the tensor according to the provided tuple.

        Parameters
        ----------
        dims : tuple
            The new order of dimensions after permutation.

        Returns
        -------
        :class:`pose_format.tensorflow.masked.tensor.MaskedTensor`
            A new MaskedTensor with dimensions permuted according to the given tuple.

        """
        tensor = self.tensor.permute(dims=dims)
        mask = self.mask.permute(dims=dims)
        return MaskedTensor(tensor=tensor, mask=mask)

    def squeeze(self, axis) -> "MaskedTensor":
        """
        Remove dimensions with size 1 while updating the mask.

        Parameters
        ----------
        axis : int or None
            The axis along which to perform squeezing.

        Returns
        -------
        :class:`pose_format.tensorflow.masked.tensor.MaskedTensor`
            MaskedTensor` with dimensions removed and mask updated.

        """
        tensor = tf.squeeze(self.tensor, axis=axis)
        mask = tf.squeeze(self.mask, axis=axis)
        return MaskedTensor(tensor=tensor, mask=mask)

    def split(self, split_size_or_sections, axis=0):
        """
        Split tensor 

        Parameters
        ----------
        split_size_or_sections : int or tf.Tensor
            Number of splits or sizes of each split/sections.
        axis : int, optional
            Axis along which to do the splitting. Default is 0.

        Returns
        -------
        list of :class:`pose_format.tensorflow.masked.tensor.MaskedTensor`
            List of new MaskedTensor objects containing the splits.

        """
        tensors = tf.split(self.tensor, split_size_or_sections, axis)
        masks = tf.split(self.mask, split_size_or_sections, axis)
        return [MaskedTensor(tensor=tensor, mask=mask) for tensor, mask in zip(tensors, masks)]

    def reshape(self, shape: tuple) -> "MaskedTensor":
        """
        Reshape tensor into custom shape (tuple)

        Parameters
        ----------
        shape : tuple
            New shape of tensor.

        Returns
        -------
        :class:`pose_format.tensorflow.masked.tensor.MaskedTensor`
            new MaskedTensor with specified shape.

        """
        tensor = tf.reshape(self.tensor, shape=shape)
        mask = tf.reshape(self.mask, shape=shape)
        return MaskedTensor(tensor=tensor, mask=mask)

    def gather(self, indexes):
        """
        Gather elements from tensor using indexes.

        Parameters
        ----------
        indexes : tf.Tensor or list or int
            Indexes used to select elements from tensor

        Returns
        -------
        :class:`pose_format.tensorflow.masked.tensor.MaskedTensor`
            A new MaskedTensor containing elements gathered from the tensor using the indexes.

        """
        tensor = tf.gather(self.tensor, indexes)
        mask = tf.gather(self.mask, indexes)
        return MaskedTensor(tensor=tensor, mask=mask)

    def rename(self, *names) -> "MaskedTensor":
        """
        Rename using custom names.

        Parameters
        ----------
        *names : str
            New names of the dimensions.

        Returns
        -------
        :class:`pose_format.tensorflow.masked.tensor.MaskedTensor`
            A new MaskedTensor with dimensions renamed.
        """
        tensor = self.tensor.rename(*names)
        mask = self.mask.rename(*names)
        return MaskedTensor(tensor=tensor, mask=mask)

    def mean(self, axis=None) -> "MaskedTensor":
        """
        Compute mean of tensor along a custom axis.

        Parameters
        ----------
        axis : None or int, optional
            Sxis along which to compute the mean. If None, compute the mean of the entire tensor. Default is None.

        Returns
        -------
        :class:`pose_format.tensorflow.masked.tensor.MaskedTensor`
            The mean of the masked tensor.
        """
        mt_sum = tf.math.reduce_sum(self.zero_filled(), axis=axis)
        mt_count = tf.math.reduce_sum(tf.cast(self.mask, mt_sum.dtype), axis=axis)
        tensor = tf.math.divide(mt_sum, mt_count)
        mask = tf.cast(mt_count, tf.bool)
        mt = MaskedTensor(tensor=tensor, mask=mask)
        return mt.fix_nan()

    def variance(self, axis=None) -> "MaskedTensor":
        """
        Compute variance of tensor along a specified axis

        Parameters
        ----------
        axis : None or int, optional
            Axis along which to compute the variance. If None, compute the variance of the entire tensor. Default is None.

        Returns
        -------
        :class:`pose_format.tensorflow.masked.tensor.MaskedTensor`
            The variance of the masked tensor.

        """
        means = self.mean(axis=axis)
        diff = self - means
        squared_deviations = diff.square()
        return squared_deviations.mean(axis=axis)

    def std(self, axis=None) -> "MaskedTensor":
        """
        Compute the standard deviation of the tensor along the specified axis.

        Parameters
        ----------
        axis : None or int, optional
            The axis along which to compute the standard deviation. If None, compute the standard deviation of the entire tensor. Default is None.

        Returns
        -------
        :class:`pose_format.tensorflow.masked.tensor.MaskedTensor`
            The standard deviation of the tensor.

        """
        variance = self.variance(axis=axis)
        return variance.sqrt()

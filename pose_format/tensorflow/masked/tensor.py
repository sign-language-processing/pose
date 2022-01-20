from typing import List

import tensorflow as tf


class MaskedTensor:
    def __init__(self, tensor: tf.Tensor, mask: tf.Tensor = None):
        self.tensor = tensor
        self.mask = mask if mask is not None else tf.ones(tensor.shape, dtype=tf.bool)  # .to(tensor.device)

    def __getattr__(self, item):
        val = self.tensor.__getattribute__(item)
        if hasattr(val, '__call__'):  # If is a function
            raise NotImplementedError("callable '%s' not defined" % item)
        else:
            return val

    def __len__(self):
        shape = self.tensor.shape
        return shape[0] if len(shape) > 0 else 1

    def __getitem__(self, key):
        if isinstance(key, list):
            tensor = tf.gather(self.tensor, key)
            mask = tf.gather(self.mask, key)
        else:
            tensor = self.tensor[key]
            mask = self.mask[key]
        return MaskedTensor(tensor=tensor, mask=mask)

    def arithmetic(self, action: str, other):
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
        tensor = tf.math.square(self.tensor)
        return MaskedTensor(tensor=tensor, mask=self.mask)

    def float(self):
        tensor = tf.cast(self.tensor, dtype=tf.float32)
        return MaskedTensor(tensor=tensor, mask=self.mask)

    def sqrt(self):
        tensor = tf.math.sqrt(self.tensor)
        return MaskedTensor(tensor=tensor, mask=self.mask)

    def sum(self, axis):
        tensor = tf.math.reduce_sum(self.tensor, axis=axis)
        mask = tf.cast(tf.math.reduce_prod(tf.cast(self.mask, tf.int32), axis=axis), tf.bool)
        return MaskedTensor(tensor=tensor, mask=mask)

    def size(self, *args):
        return self.tensor.size(*args)

    def fix_nan(self):
        self.tensor = tf.where(tf.math.is_finite(self.tensor), self.tensor, tf.zeros_like(self.tensor))
        return self

    def zero_filled(self) -> tf.Tensor:
        return self.tensor * tf.cast(self.mask, dtype=self.tensor.dtype)

    def div(self, other: "MaskedTensor", in_place=False, update_mask=True):
        tensor = tf.div(self.tensor, other.tensor, out=self.tensor if in_place else None)
        mask = self.mask & other.mask if update_mask else self.mask
        return MaskedTensor(tensor, mask)

    def matmul(self, matrix: tf.Tensor) -> "MaskedTensor":
        tensor = tf.matmul(self.tensor, matrix)
        return MaskedTensor(tensor=tensor, mask=self.mask)

    def transpose(self, perm: List[int]) -> "MaskedTensor":
        tensor = tf.transpose(self.tensor, perm=perm)
        mask = tf.transpose(self.mask, perm=perm)
        return MaskedTensor(tensor=tensor, mask=mask)

    def permute(self, dims: tuple) -> "MaskedTensor":
        tensor = self.tensor.permute(dims=dims)
        mask = self.mask.permute(dims=dims)
        return MaskedTensor(tensor=tensor, mask=mask)

    def squeeze(self, axis) -> "MaskedTensor":
        tensor = tf.squeeze(self.tensor, axis=axis)
        mask = tf.squeeze(self.mask, axis=axis)
        return MaskedTensor(tensor=tensor, mask=mask)

    def split(self, split_size_or_sections, axis=0):
        tensors = tf.split(self.tensor, split_size_or_sections, axis)
        masks = tf.split(self.mask, split_size_or_sections, axis)
        return [MaskedTensor(tensor=tensor, mask=mask) for tensor, mask in zip(tensors, masks)]

    def reshape(self, shape: tuple) -> "MaskedTensor":
        tensor = tf.reshape(self.tensor, shape=shape)
        mask = tf.reshape(self.mask, shape=shape)
        return MaskedTensor(tensor=tensor, mask=mask)

    def gather(self, indexes):
        tensor = tf.gather(self.tensor, indexes)
        mask = tf.gather(self.mask, indexes)
        return MaskedTensor(tensor=tensor, mask=mask)

    def rename(self, *names) -> "MaskedTensor":
        tensor = self.tensor.rename(*names)
        mask = self.mask.rename(*names)
        return MaskedTensor(tensor=tensor, mask=mask)

    def mean(self, axis=None) -> "MaskedTensor":
        mt_sum = tf.math.reduce_sum(self.zero_filled(), axis=axis)
        mt_count = tf.math.reduce_sum(tf.cast(self.mask, mt_sum.dtype), axis=axis)
        tensor = tf.math.divide(mt_sum, mt_count)
        mask = tf.cast(mt_count, tf.bool)
        mt = MaskedTensor(tensor=tensor, mask=mask)
        return mt.fix_nan()

    def variance(self, axis=None) -> "MaskedTensor":
        means = self.mean(axis=axis)
        diff = self - means
        squared_deviations = diff.square()
        return squared_deviations.mean(axis=axis)

    def std(self, axis=None) -> "MaskedTensor":
        variance = self.variance(axis=axis)
        return variance.sqrt()

from typing import List, Union

import tensorflow

from pose_format.tensorflow.masked.tensor import MaskedTensor


class TensorflowFallback(type):
    """A metaclass for managing the fallback operations on MaskedTensors with Tensorflow functions."""

    doesnt_change_mask = {"sqrt", "square", "cos", "sin", "tan", "acos", "asin", "atan"}

    def __getattr__(cls, attr):
        """
        to return Tensorflow functions that can work on MaskedTensors.
        
        Parameters
        ----------
        attr : str
            Tensorflow function name
            
        Returns
        -------
        function
            function that can handle both MaskedTensor and regular/unmasked Tensorflow Tensor objects.
        """

        def func(*args, **kwargs):
            if len(args) > 0 and isinstance(args[0], MaskedTensor):
                args = list(args)
                mask = args[0].mask
                args[0] = args[0].tensor

                res = getattr(tensorflow, attr)(*args, **kwargs)
                if attr in TensorflowFallback.doesnt_change_mask:
                    return MaskedTensor(res, mask)
                else:
                    return res

            else:  # If this action is done on an unmasked tensor
                return getattr(tensorflow, attr)(*args, **kwargs)

        return func


class MaskedTensorflow(metaclass=TensorflowFallback):
    """
    Class that performs Tensorflow operations on MaskedTensors. 
    It uses the TensorflowFallback metaclass to handle functions not explicitly defined in this class.
    """

    @staticmethod
    def concat(tensors: List[Union[MaskedTensor, tensorflow.Tensor]], axis: int) -> MaskedTensor:
        """
        Concatenates a list of tensors along a specified axis.
        
        Parameters
        ----------
        tensors : list
            List of MaskedTensor or tensorflow.Tensor objects.
        axis : int
            The axis along which to concatenate the tensors.
            
        Returns
        -------
        :class:`~pose_format.tensorflow.masked.tensor.MaskedTensor`
            concatenated Maskedtensor
        """
        tensors: List[MaskedTensor] = [t if isinstance(t, MaskedTensor) else MaskedTensor(tensor=t) for t in tensors]
        tensor = tensorflow.concat([t.tensor for t in tensors], axis=axis)
        mask = tensorflow.concat([t.mask for t in tensors], axis=axis)
        return MaskedTensor(tensor=tensor, mask=mask)

    @staticmethod
    def stack(tensors: List[MaskedTensor], axis: int) -> MaskedTensor:
        """
        Stacks a list of tensors along a specified axis.
        
        Parameters
        ----------
        tensors : list
            List of MaskedTensor objects.
        axis : int
            The axis along which to stack the tensors.
            
        Returns
        -------
        :class:`~pose_format.tensorflow.masked.tensor.MaskedTensor`
            masekd stacked tensor.
        """
        tensor = tensorflow.stack([t.tensor for t in tensors], axis=axis)
        mask = tensorflow.stack([t.mask for t in tensors], axis=axis)
        return MaskedTensor(tensor=tensor, mask=mask)

    @staticmethod
    def zeros(size, dtype=tensorflow.float32) -> MaskedTensor:
        """
        Returns a MaskedTensor of zeros with the specified size and dtype.
        
        Parameters
        ----------
        size : tuple
            The shape of the output tensor.
        dtype : tensorflow datatype, optional
            The datatype of the output tensor, default is tensorflow.float32.
            
        Returns
        -------
        :class:`~pose_format.tensorflow.masked.tensor.MaskedTensor`
            masked tensor of zeros.
        """
        tensor = tensorflow.zeros(size, dtype=dtype)
        mask = tensorflow.zeros(size, dtype=tensorflow.bool)
        return MaskedTensor(tensor=tensor, mask=mask)

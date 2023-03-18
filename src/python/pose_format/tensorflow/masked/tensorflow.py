from typing import List, Union

import tensorflow

from pose_format.tensorflow.masked.tensor import MaskedTensor


class TensorflowFallback(type):
    doesnt_change_mask = {
        "sqrt", "square",
        "cos", "sin", "tan", "acos", "asin", "atan"
    }

    def __getattr__(cls, attr):
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
    @staticmethod
    def concat(tensors: List[Union[MaskedTensor, tensorflow.Tensor]], axis: int) -> MaskedTensor:
        tensors: List[MaskedTensor] = [t if isinstance(t, MaskedTensor) else MaskedTensor(tensor=t) for t in tensors]
        tensor = tensorflow.concat([t.tensor for t in tensors], axis=axis)
        mask = tensorflow.concat([t.mask for t in tensors], axis=axis)
        return MaskedTensor(tensor=tensor, mask=mask)

    @staticmethod
    def stack(tensors: List[MaskedTensor], axis: int) -> MaskedTensor:
        tensor = tensorflow.stack([t.tensor for t in tensors], axis=axis)
        mask = tensorflow.stack([t.mask for t in tensors], axis=axis)
        return MaskedTensor(tensor=tensor, mask=mask)

    @staticmethod
    def zeros(size, dtype=tensorflow.float32) -> MaskedTensor:
        tensor = tensorflow.zeros(size, dtype=dtype)
        mask = tensorflow.zeros(size, dtype=tensorflow.bool)
        return MaskedTensor(tensor=tensor, mask=mask)

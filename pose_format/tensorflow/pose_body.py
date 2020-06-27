from typing import List, Union

import tensorflow
import numpy as np

from ..pose_header import PoseHeader
from ..pose_body import PoseBody, POINTS_DIMS
from ..utils.reader import BufferReader
from .masked.tensor import MaskedTensor


class TensorflowPoseBody(PoseBody):
    tensor_reader = 'unpack_tensorflow'

    def __init__(self, fps: int, data: Union[MaskedTensor, tensorflow.Tensor], confidence: tensorflow.Tensor):
        if isinstance(data, tensorflow.Tensor):  # If array is not masked
            mask = confidence > 0
            data = MaskedTensor(data, tensorflow.stack([mask] * 2, dim=3))

        super().__init__(fps, data, confidence)

    def zero_filled(self):
        self.data.zero_filled()
        return self

from typing import Union, List

import tensorflow as tf
import numpy as np

from ..pose_body import PoseBody
from .masked.tensor import MaskedTensor

TF_POSE_RECORD_DESCRIPTION = {
    'fps': tf.io.FixedLenFeature([], tf.int64, default_value=0),
    'pose_data': tf.io.FixedLenFeature([], tf.string),
    'pose_confidence': tf.io.FixedLenFeature([], tf.string),
}


class TensorflowPoseBody(PoseBody):
    tensor_reader = 'unpack_tensorflow'

    def __init__(self, fps: int, data: Union[MaskedTensor, tf.Tensor], confidence: tf.Tensor):
        if isinstance(data, tf.Tensor):  # If array is not masked
            mask = confidence > 0
            data = MaskedTensor(data, tf.stack([mask] * 2, axis=3))

        super().__init__(fps, data, confidence)

    def zero_filled(self):
        self.data.zero_filled()
        return self

    def select_frames(self, frame_indexes: List[int]):
        data = self.data[frame_indexes]
        confidence = tf.gather(self.confidence, frame_indexes)
        return self.__class__(fps=self.fps, data=data, confidence=confidence)

    def matmul(self, matrix: np.ndarray) -> __qualname__:
        data = self.data.matmul(tf.convert_to_tensor(matrix))
        return self.__class__(fps=self.fps, data=data, confidence=self.confidence)

    def as_tfrecord(self):
        data = tf.io.serialize_tensor(self.data.tensor).numpy()
        confidence = tf.io.serialize_tensor(self.confidence).numpy()

        return {
            'fps': tf.train.Feature(int64_list=tf.train.Int64List(value=[self.fps])),
            'pose_data': tf.train.Feature(bytes_list=tf.train.BytesList(value=[data])),
            'pose_confidence': tf.train.Feature(bytes_list=tf.train.BytesList(value=[confidence]))
        }

    @classmethod
    def from_tfrecord(cls, tfrecord_dict: dict):
        fps = tfrecord_dict['fps'].numpy()
        data = tf.io.parse_tensor(tfrecord_dict['pose_data'], out_type=tf.float32)
        confidence = tf.io.parse_tensor(tfrecord_dict['pose_confidence'], out_type=tf.float32)
        return cls(fps=fps, data=data, confidence=confidence)


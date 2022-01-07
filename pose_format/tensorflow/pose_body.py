from typing import Union, List

import numpy as np
import tensorflow as tf

from .masked.tensor import MaskedTensor
from ..pose_body import PoseBody, POINTS_DIMS

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
        data = self.data.gather(frame_indexes)
        confidence = tf.gather(self.confidence, frame_indexes)
        return self.__class__(fps=self.fps, data=data, confidence=confidence)

    def frame_dropout(self, dropout_std: float):
        data_len = tf.cast(tf.shape(self.data.tensor)[0], dtype=tf.float32)

        select_percent = tf.random.uniform([1], minval=0.2, maxval=1)[0]
        number_sample = tf.cast(tf.round(data_len * select_percent), dtype=tf.int32)

        idxs = tf.range(data_len - 1, dtype=tf.int32)
        select_indexes = tf.sort(tf.random.shuffle(idxs)[:number_sample])
        select_indexes = tf.cast(select_indexes, dtype=tf.int32)

        return self.select_frames(select_indexes), select_indexes

    def points_perspective(self):
        return self.data.transpose(perm=POINTS_DIMS)

    def get_points(self, indexes: List[int]):
        data = self.data.transpose(perm=POINTS_DIMS)
        new_data = data[indexes].transpose(perm=POINTS_DIMS)

        confidence_reshape = [2, 1, 0]
        confidence = tf.transpose(self.confidence, perm=confidence_reshape)
        new_confidence = tf.transpose(tf.gather(confidence, indexes), perm=confidence_reshape)

        return TensorflowPoseBody(self.fps, new_data, new_confidence)

    def matmul(self, matrix: np.ndarray) -> __qualname__:
        matrix = tf.convert_to_tensor(matrix, dtype=self.data.dtype)
        data = self.data.matmul(matrix)
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
        fps = tf.cast(tfrecord_dict['fps'], dtype=tf.float32)
        data = tf.io.parse_tensor(tfrecord_dict['pose_data'], out_type=tf.float32)
        confidence = tf.io.parse_tensor(tfrecord_dict['pose_confidence'], out_type=tf.float32)
        return cls(fps=fps, data=data, confidence=confidence)

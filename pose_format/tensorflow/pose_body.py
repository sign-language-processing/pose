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

    def __init__(self, fps: float, data: Union[MaskedTensor, tf.Tensor], confidence: tf.Tensor):
        if isinstance(data, tf.Tensor):  # If array is not masked
            mask = confidence > 0
            data = MaskedTensor(data, tf.stack([mask] * 2, axis=3))

        super().__init__(fps, data, confidence)

    def zero_filled(self):
        self.data = self.data.zero_filled()
        return self

    def select_frames(self, frame_indexes: List[int]):
        data = self.data.gather(frame_indexes)
        confidence = tf.gather(self.confidence, frame_indexes)
        return self.__class__(fps=self.fps, data=data, confidence=confidence)

    def frame_dropout_given_percent(self, dropout_percent: float):
        """
        Remove some frames from the data at random.

        :param dropout_percent:
        :return:
        """
        data_len = tf.cast(tf.shape(self.data.tensor)[0], dtype=tf.float32)

        number_sample = tf.squeeze(tf.round(data_len * dropout_percent))

        # always keep at least 1 frame
        number_sample = tf.maximum(1.0, number_sample)

        number_sample = tf.cast(number_sample, dtype=tf.int32)

        idxs = tf.range(data_len - 1, dtype=tf.int32)

        select_indexes = tf.sort(tf.random.shuffle(idxs)[:number_sample])
        select_indexes = tf.cast(select_indexes, dtype=tf.int32)

        return self.select_frames(select_indexes), select_indexes

    def frame_dropout_uniform(self,
                              dropout_min: float = 0.2,
                              dropout_max: float = 1.0):

        dropout_percent = tf.random.uniform([1], minval=dropout_min, maxval=dropout_max)[0]

        return self.frame_dropout_given_percent(dropout_percent)

    def frame_dropout_normal(self,
                             dropout_mean: float = 0.5,
                             dropout_std: float = 0.1):

        dropout_percent = tf.random.normal([1], mean=dropout_mean, stddev=dropout_std)[0]

        # clip negative values to zero
        dropout_percent = tf.maximum(dropout_percent, tf.constant([0.0]))

        return self.frame_dropout_given_percent(dropout_percent)

    def points_perspective(self) -> MaskedTensor:
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

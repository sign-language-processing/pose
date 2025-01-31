from typing import List, Union

import numpy as np
import tensorflow as tf

from ..pose_body import POINTS_DIMS, PoseBody
from .masked.tensor import MaskedTensor

TF_POSE_RECORD_DESCRIPTION = {
    'fps': tf.io.FixedLenFeature([], tf.int64, default_value=0),
    'pose_data': tf.io.FixedLenFeature([], tf.string),
    'pose_confidence': tf.io.FixedLenFeature([], tf.string),
}


class TensorflowPoseBody(PoseBody):
    """
    Representation of pose body data, optimized for TensorFlow operations.

    * Inherits from PoseBody 

    Parameters
    ----------
    fps : float
        The frames per second for the pose data.
    data : Union[:class:`~pose_format.tensorflow.masked.tensor.MaskedTensor`, tf.Tensor]
        The pose data.
    confidence : tf.Tensor
        The confidence scores for the pose data.
    """

    """str: The method used to read the tensor data. (Type: str)"""
    tensor_reader = 'unpack_tensorflow'

    def __init__(self, fps: float, data: Union[MaskedTensor, tf.Tensor], confidence: tf.Tensor):
        """
        Initializes the TensorflowPoseBody with fps, data and confidence.

        """
        if isinstance(data, tf.Tensor):  # If array is not masked
            mask = confidence > 0
            data = MaskedTensor(data, tf.stack([mask] * 2, axis=3))

        super().__init__(fps, data, confidence)

    def zero_filled(self) -> 'TensorflowPoseBody':
        """Return an instance with zero-filled data."""
        copy = self.copy()
        copy.data = self.data.zero_filled()
        return copy

    def select_frames(self, frame_indexes: List[int]):
        """
        Selects and returns a subset of frames based on the frame indexes.

        Parameters
        ----------
        frame_indexes : List[int]
            List of frame indexes

        Returns
        -------
        TensorflowPoseBody
            Instance with the selected frames
        """
        data = self.data.gather(frame_indexes)
        confidence = tf.gather(self.confidence, frame_indexes)
        return self.__class__(fps=self.fps, data=data, confidence=confidence)

    def frame_dropout_given_percent(self, dropout_percent: float):
        """
        Remove some frames from the data at random from pose data.

        Parameters
        ----------
        dropout_percent : float
            The percentage of frames to drop.

        Returns
        -------
        TensorflowPoseBody, tf.Tensor
            A new instance with dropped frames and the selected frame indexes.
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

    def frame_dropout_uniform(self, dropout_min: float = 0.2, dropout_max: float = 1.0):
        """
        Drops randomly frames based on a given uniform distribution
        
        Parameters
        ----------
        dropout_min : float, optional
            minimum percentage for dropout, by default 0.2.
        dropout_max : float, optional
            maximum percentage for dropout, by default 1.0.

        Returns
        -------
        TensorflowPoseBody
            Instance with frames dropped based on a uniform distribution.
        """

        dropout_percent = tf.random.uniform([1], minval=dropout_min, maxval=dropout_max)[0]

        return self.frame_dropout_given_percent(dropout_percent)

    def frame_dropout_normal(self, dropout_mean: float = 0.5, dropout_std: float = 0.1):
        """
        Given mean and standard deviation, randomly drops out based on normal distribution. 

        Parameters
        ----------
        dropout_mean : float, optional
            The mean for the normal distribution, by default 0.5.
        dropout_std : float, optional
            The standard deviation for the normal distribution, by default 0.1.

        Returns
        -------
        TensorflowPoseBody
            instance with frames dropped based on normal distribution.
        """

        dropout_percent = tf.random.normal([1], mean=dropout_mean, stddev=dropout_std)[0]

        # clip negative values to zero
        dropout_percent = tf.maximum(dropout_percent, tf.constant([0.0]))

        return self.frame_dropout_given_percent(dropout_percent)

    def points_perspective(self) -> MaskedTensor:
        """
        Returns perspective transformation of pose points.

        Returns
        -------
        :class:`~pose_format.tensorflow.masked.tensor.MaskedTensor`
            Transformed pose data.
        """
        return self.data.transpose(perm=POINTS_DIMS)

    def copy(self) -> 'TensorflowPoseBody':
        # Ensure copies are fully detached from the TF computation graph by round-trip through numpy
        detached_data = tf.convert_to_tensor(self.data.tensor.numpy())
        detached_mask = tf.convert_to_tensor(self.data.mask.numpy())
        data_copy = MaskedTensor(detached_data, detached_mask)
        confidence_copy = tf.convert_to_tensor(self.confidence.numpy())
        return self.__class__(
            fps=self.fps,
            data=data_copy,
            confidence=confidence_copy)

    def get_points(self, indexes: List[int]):
        """
        Gets and returns points from pose data based on indexes 

        Parameters
        ----------
        indexes : List[int]
            List of point indexes to get.

        Returns
        -------
        TensorflowPoseBody
            Instance containing only the gotten points.
        """
        data = self.data.transpose(perm=POINTS_DIMS)
        new_data = data[indexes].transpose(perm=POINTS_DIMS)

        confidence_reshape = [2, 1, 0]
        confidence = tf.transpose(self.confidence, perm=confidence_reshape)
        new_confidence = tf.transpose(tf.gather(confidence, indexes), perm=confidence_reshape)

        return TensorflowPoseBody(self.fps, new_data, new_confidence)

    def matmul(self, matrix: np.ndarray) -> __qualname__:
        """
        Multiplies pose data with a given matrix.

        Parameters
        ----------
        matrix : np.ndarray
            Matrix to multiply with pose data.

        Returns
        -------
        TensorflowPoseBody
            Instance with the pose data multiplied by the matrix.
        """
        matrix = tf.convert_to_tensor(matrix, dtype=self.data.dtype)
        data = self.data.matmul(matrix)
        return self.__class__(fps=self.fps, data=data, confidence=self.confidence)

    def as_tfrecord(self):
        """
        Converts into TensorFlow (tf) record format

        Returns
        -------
        dict
            dictionary representation of TensorFlow (tf) record for the pose body
        """
        data = tf.io.serialize_tensor(self.data.tensor).numpy()
        confidence = tf.io.serialize_tensor(self.confidence).numpy()

        return {
            'fps': tf.train.Feature(int64_list=tf.train.Int64List(value=[self.fps])),
            'pose_data': tf.train.Feature(bytes_list=tf.train.BytesList(value=[data])),
            'pose_confidence': tf.train.Feature(bytes_list=tf.train.BytesList(value=[confidence]))
        }

    @classmethod
    def from_tfrecord(cls, tfrecord_dict: dict):
        """
        From a TensorFlow record dictionary, it creates a instance of TensorflowPoseBody

        Parameters
        ----------
        tfrecord_dict : dict
            Dictionary representation of TensorFlow (tf) record data.

        Returns
        -------
        TensorflowPoseBody
            An instance constructed from given TensorFlow record data
        """
        fps = tf.cast(tfrecord_dict['fps'], dtype=tf.float32)
        data = tf.io.parse_tensor(tfrecord_dict['pose_data'], out_type=tf.float32)
        confidence = tf.io.parse_tensor(tfrecord_dict['pose_confidence'], out_type=tf.float32)
        return cls(fps=fps, data=data, confidence=confidence)

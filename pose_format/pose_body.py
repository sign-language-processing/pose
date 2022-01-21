from random import sample
from typing import List, Tuple, BinaryIO

import numpy as np

from pose_format.pose_header import PoseHeader
from pose_format.utils.reader import BufferReader, ConstStructs

POINTS_DIMS = (2, 1, 0, 3)


class PoseBody:
    tensor_reader = 'ABSTRACT-DO-NOT-USE'

    def __init__(self, fps: float, data, confidence):
        self.fps = fps
        self.data = data  # Shape (Frames, People, Points, Dims) - eg (93, 1, 137, 2)
        self.confidence = confidence  # Shape (Frames, People, Points) - eg (93, 1, 137)

    @classmethod
    def read(cls, header: PoseHeader, reader: BufferReader):
        if header.version == 0:
            return cls.read_v0_0(header, reader)
        elif round(header.version, 3) == 0.1:
            return cls.read_v0_1(header, reader)

        raise NotImplementedError("Unknown version - %f" % header.version)

    @classmethod
    def read_v0_0(cls, header: PoseHeader, reader: BufferReader):
        raise NotImplementedError("'read_v0_0' not implemented on '%s'" % cls.__class__)

    @classmethod
    def read_v0_1(cls, header: PoseHeader, reader: BufferReader):
        fps, _frames = reader.unpack(ConstStructs.double_ushort)

        _people = reader.unpack(ConstStructs.ushort)
        _points = sum([len(c.points) for c in header.components])
        _dims = max([len(c.format) for c in header.components]) - 1

        # _frames is defined as short, which sometimes is not enough! TODO change to int
        _frames = int(reader.bytes_left() / (_people * _points * (_dims + 1) * 4))

        tensor_reader = reader.__getattribute__(cls.tensor_reader)
        data = tensor_reader(ConstStructs.float, shape=(_frames, _people, _points, _dims))
        confidence = tensor_reader(ConstStructs.float, shape=(_frames, _people, _points))

        return cls(fps, data, confidence)

    def write(self, version: float, buffer: BinaryIO):
        """
        Write the data to a file based on the version of the spec
        :param version: float
        :param buffer: BinaryIO
        """
        raise NotImplementedError("'write' not implemented on '%s'" % self.__class__)

    def numpy(self):
        """
        Convert the current PoseBody representation to NumpyPoseBody
        :return: NumpyPoseBody
        """
        raise NotImplementedError("'numpy' not implemented on '%s'" % self.__class__)

    def torch(self):
        """
        Convert the current PoseBody representation to TorchPoseBody
        :return: TorchPoseBody
        """
        raise NotImplementedError("'torch' not implemented on '%s'" % self.__class__)

    def tensorflow(self):
        """
        Convert the current PoseBody representation to TensorflowPoseBody
        :return: TensorflowPoseBody
        """
        raise NotImplementedError("'tensorflow' not implemented on '%s'" % self.__class__)

    def flatten(self):
        """
        Convert the data from the (Frames, People, Points, Dims) masked representation to array of points.
        Every item in the result array contains the following dimensions:
        0. Time in milliseconds
        1. Person ID
        2. Point ID
        3. X dimension
        4. Y dimension
        5. Z dimension if exists
        6. Pose estimation confidence
        :return:
        """
        raise NotImplementedError("'flatten' not implemented on '%s'" % self.__class__)

    def slice_step(self, by: int) -> "PoseBody":
        """
        Slice the data by skipping rows.
        This slicing affects the FPS.
        :param by: take one row every "by" rows
        :return: PoseBody
        """
        new_data = self.data[::by]
        new_confidence = self.confidence[::by]
        new_fps = self.fps / by

        return self.__class__(fps=new_fps, data=new_data, confidence=new_confidence)

    def augment2d(self, rotation_std=0.2, shear_std=0.2, scale_std=0.2):
        """
        :param rotation_std: Rotation in radians
        :param shear_std: Shear X in percent
        :param scale_std: Scale X in percent
        :return:
        """
        matrix = np.eye(2)

        # Based on https://en.wikipedia.org/wiki/Shear_matrix
        if shear_std > 0:
            shear_matrix = np.eye(2)
            shear_matrix[0][1] = np.random.normal(loc=0, scale=shear_std, size=1)[0]
            matrix = np.dot(matrix, shear_matrix)

        # Based on https://en.wikipedia.org/wiki/Rotation_matrix
        if rotation_std > 0:
            rotation_angle = np.random.normal(loc=0, scale=rotation_std, size=1)[0]
            rotation_cos = np.cos(rotation_angle)
            rotation_sin = np.sin(rotation_angle)
            rotation_matrix = np.array([[rotation_cos, -rotation_sin], [rotation_sin, rotation_cos]])
            matrix = np.dot(matrix, rotation_matrix)

        # Based on https://en.wikipedia.org/wiki/Scaling_(geometry)
        if scale_std > 0:
            scale_matrix = np.eye(2)
            scale_matrix[1][1] += np.random.normal(loc=0, scale=scale_std, size=1)[0]
            matrix = np.dot(matrix, scale_matrix)

        # Cast to matrix the correct size
        dim_matrix = np.eye(self.data.shape[-1])
        dim_matrix[0:2, 0:2] = matrix

        return self.matmul(dim_matrix.astype(dtype=np.float32))

    def zero_filled(self) -> __qualname__:
        raise NotImplementedError("'zero_filled' not implemented on '%s'" % self.__class__)

    def matmul(self, matrix: np.ndarray) -> __qualname__:
        raise NotImplementedError("'matmul' not implemented on '%s'" % self.__class__)

    def get_points(self, indexes: List[int]) -> __qualname__:
        raise NotImplementedError("'get_points' not implemented on '%s'" % self.__class__)

    def bbox(self, header: PoseHeader) -> __qualname__:
        raise NotImplementedError("'bbox' not implemented on '%s'" % self.__class__)

    def points_perspective(self):
        raise NotImplementedError("'points_perspective' not implemented on '%s'" % self.__class__)

    def select_frames(self, frame_indexes: List[int]):
        data = self.data[frame_indexes]
        confidence = self.confidence[frame_indexes]
        return self.__class__(fps=self.fps, data=data, confidence=confidence)

    def frame_dropout_given_percent(self, dropout_percent: float) -> Tuple["PoseBody", List[int]]:
        data_len = len(self.data)
        dropout_number = min(int(data_len * dropout_percent), int(data_len * 0.99))
        dropout_indexes = set(sample(range(0, data_len), dropout_number))
        select_indexes = [i for i in range(0, data_len) if i not in dropout_indexes]

        return self.select_frames(select_indexes), select_indexes

    def frame_dropout_uniform(self,
                              dropout_min: float = 0.2,
                              dropout_max: float = 1.0) -> Tuple["PoseBody", List[int]]:
        dropout_percent = np.random.uniform(low=dropout_min, high=dropout_max, size=1)[0]

        return self.frame_dropout_given_percent(dropout_percent)

    def frame_dropout_normal(self,
                             dropout_mean: float = 0.5,
                             dropout_std: float = 0.1) -> Tuple["PoseBody", List[int]]:
        dropout_percent = np.abs(np.random.normal(loc=dropout_mean, scale=dropout_std, size=1))[0]

        return self.frame_dropout_given_percent(dropout_percent)

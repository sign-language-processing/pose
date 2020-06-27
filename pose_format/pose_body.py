from random import sample
from typing import List, Tuple, BinaryIO

import numpy as np
from .utils.reader import BufferReader, ConstStructs
from .pose_header import PoseHeader

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

        _dims = max([len(c.format) for c in header.components]) - 1
        _points = sum([len(c.points) for c in header.components])

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

        return self.__class__(fps=self.fps / by, data=new_data, confidence=new_confidence)

    def zero_filled(self) -> __qualname__:
        raise NotImplementedError("'zero_filled' not implemented on '%s'" % self.__class__)

    def matmul(self, matrix: np.ndarray) -> __qualname__:
        raise NotImplementedError("'matmul' not implemented on '%s'" % self.__class__)

    def points_perspective(self):
        raise NotImplementedError("'points_perspective' not implemented on '%s'" % self.__class__)

    def select_frames(self, frame_indexes: List[int]):
        data = self.data[frame_indexes]
        confidence = self.confidence[frame_indexes]
        return self.__class__(fps=self.fps, data=data, confidence=confidence)

    def frame_dropout(self, dropout_std: float) -> Tuple["PoseBody", List[int]]:
        dropout_percent = np.abs(np.random.normal(loc=0, scale=dropout_std, size=1))[0]
        data_len = len(self.data)
        dropout_number = min(int(data_len * dropout_percent), data_len - 1)
        dropout_indexes = set(sample(range(0, data_len), dropout_number))
        select_indexes = [i for i in range(0, data_len) if i not in dropout_indexes]
        return self.select_frames(select_indexes), select_indexes

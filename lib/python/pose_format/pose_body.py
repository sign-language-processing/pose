from random import sample
from typing import List, Tuple

import numpy as np
from .utils.reader import BufferReader, ConstStructs
from .pose_header import PoseHeader


POINTS_DIMS = (2, 1, 0, 3)


class PoseBody:
    tensor_reader = 'ABSTRACT-DO-NOT-USE'

    def __init__(self, fps: int, data, confidence):
        self.fps = fps
        self.data = data  # Shape (frames, people, points, dims) - eg (93, 1, 137, 2)
        self.confidence = confidence  # Shape (frames, people, points) - eg (93, 1, 137)

    @classmethod
    def read_v0_0(cls, header: PoseHeader, reader: BufferReader):
        raise NotImplementedError("Reading v0.0 is not implemented for class " + cls.__name__)

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


    @classmethod
    def read(cls, header: PoseHeader, reader: BufferReader):
        if header.version == 0:
            return cls.read_v0_0(header, reader)
        elif round(header.version, 3) == 0.1:
            return cls.read_v0_1(header, reader)

        raise NotImplementedError("Unknown version - " + str(header.version))

    def matmul(self, matrix: np.ndarray) -> __qualname__:
        raise NotImplementedError("Must implement matmul")

    def points_perspective(self):
        raise NotImplementedError("Must implement points_perspective")

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

    def flatten(self):
        raise NotImplementedError("Must implement flatten")
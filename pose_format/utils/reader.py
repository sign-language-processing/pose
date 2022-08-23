import struct
from typing import Tuple

import numpy as np
from dataclasses import dataclass
from tqdm import tqdm


@dataclass
class ConstStructs:
    float: struct.Struct = struct.Struct("<f")

    short: struct.Struct = struct.Struct("<h")
    ushort: struct.Struct = struct.Struct("<H")

    double_ushort: struct.Struct = struct.Struct("<HH")
    triple_ushort: struct.Struct = struct.Struct("<HHH")


class BufferReader:
    def __init__(self, buffer: bytes):
        self.buffer = buffer
        self.read_offset = 0

    def bytes_left(self):
        return len(self.buffer) - self.read_offset

    def unpack_f(self, s_format: str):
        if not hasattr(ConstStructs, s_format):
            le_format: str = "<" + s_format
            setattr(ConstStructs, s_format, struct.Struct(le_format))

        return self.unpack(getattr(ConstStructs, s_format))

    def unpack_numpy(self, s: struct.Struct, shape: Tuple):
        arr = np.ndarray(shape, s.format, self.buffer, self.read_offset).copy()
        self.advance(s, int(np.prod(shape)))
        return arr

    def unpack_torch(self, s: struct.Struct, shape: Tuple):
        import torch

        arr = self.unpack_numpy(s, shape)
        return torch.from_numpy(arr)

    def unpack_tensorflow(self, s: struct.Struct, shape: Tuple):
        import tensorflow as tf

        arr = self.unpack_numpy(s, shape)
        return tf.constant(arr)

    def unpack(self, s: struct.Struct):
        unpack: tuple = s.unpack_from(self.buffer, self.read_offset)
        self.advance(s)
        if len(unpack) == 1:
            return unpack[0]
        return unpack

    def advance(self, s: struct.Struct, times=1):
        self.read_offset += s.size * times

    def unpack_str(self) -> str:
        length: int = self.unpack(ConstStructs.ushort)
        bytes_: bytes = self.unpack_f("%ds" % length)
        return bytes_.decode("utf-8")


if __name__ == "__main__":
    buffer = struct.pack("<H5s", 5, bytes("hello", 'utf8'))
    reader = BufferReader(buffer)

    for _ in tqdm(range(10000)):
        reader.read_offset = 0
        reader.unpack_str()

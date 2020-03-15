import struct
from io import BufferedReader
from typing import Tuple
import numpy as np
from dataclasses import dataclass


@dataclass
class ConstStructs:
    float = struct.Struct("<f")

    short = struct.Struct("<h")
    ushort = struct.Struct("<H")

    double_ushort = struct.Struct("<HH")
    triple_ushort = struct.Struct("<HHH")


class BufferReader:
    def __init__(self, buffer: bytes):
        self.buffer = buffer
        self.read_offset = 0

    def unpack_f(self, s_format: str):
        if not hasattr(ConstStructs, s_format):
            le_format = "<" + s_format
            setattr(ConstStructs, s_format, struct.Struct(le_format))

        return self.unpack(getattr(ConstStructs, s_format))

    def unpack_numpy(self, s: struct.Struct, shape: Tuple):
        arr = np.ndarray(shape, s.format, self.buffer, self.read_offset)
        self.advance(s, int(np.prod(shape)))
        return arr

    def unpack(self, s: struct.Struct):
        unpack = s.unpack_from(self.buffer, self.read_offset)
        self.advance(s)
        if len(unpack) == 1:
            return unpack[0]
        return unpack

    def advance(self, s: struct.Struct, times=1):
        self.read_offset += s.size * times

    def unpack_str(self):
        length = self.unpack(ConstStructs.ushort)
        bytes_ = self.unpack_f("%ds" % length)
        return bytes_.decode("utf-8")

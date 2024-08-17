import struct
from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class ConstStructs:
    """
    Class hold collection of predefined struct formats to reuse
    """
    float: struct.Struct = struct.Struct("<f")
    """
Struct format for floating-point number"""

    short: struct.Struct = struct.Struct("<h")
    """
Struct format for signed short integer,'<h' """

    ushort: struct.Struct = struct.Struct("<H")
    """
Struct format for unsigned short integer"""

    double_ushort: struct.Struct = struct.Struct("<HH")
    """
Struct format for two unsigned short integers"""

    triple_ushort: struct.Struct = struct.Struct("<HHH")
    """
Struct format for three unsigned short integers"""

    uint: struct.Struct = struct.Struct("<I")
    """
Struct format for unsigned integer"""


class BufferReader:
    """
    Class is used to read binary data from buffer
    
    Parameters
    ----------
        buffer: bytes
            buffer from which to read data
        read_offset: int
            current read offset in buffer
    """

    def __init__(self, buffer: bytes):
        self.buffer = buffer
        self.read_offset = 0

    def bytes_left(self):
        """
        gives number of bytes left to read from buffer
        
        Returns
        -------
        int
            The number of bytes left to read.
        """
        return len(self.buffer) - self.read_offset

    def unpack_f(self, s_format: str):
        """
        unpacks data from buffer using given struct format
        
        Parameters
        ----------
        s_format : str
            The struct format to use for unpacking data.
        
        Returns
        -------
        Unpacked data as specified by the struct format.
        """
        if not hasattr(ConstStructs, s_format):
            le_format: str = "<" + s_format
            setattr(ConstStructs, s_format, struct.Struct(le_format))

        return self.unpack(getattr(ConstStructs, s_format))

    def unpack_numpy(self, s: struct.Struct, shape: Tuple):
        """
        unpacks data from buffer into a numpy array using struct format and shape
        
        Parameters
        ----------
        s : struct.Struct
            The struct format to use.
        shape : Tuple[int, ...]
            The shape of the NumPy array.
        
        Returns
        -------
        np.ndarray
            The unpacked NumPy array.
        """
        arr = np.ndarray(shape, s.format, self.buffer, self.read_offset).copy()
        self.advance(s, int(np.prod(shape)))
        return arr

    def unpack_torch(self, s: struct.Struct, shape: Tuple):
        """ unpacks data from buffer into a torch tensor using struct format and shape
        
        Parameters
        ----------
        s : struct.Struct
            The struct format to use.
        shape : Tuple[int, ...]
            The shape of the PyTorch tensor.
        
        Returns
        -------
        torch.Tensor
            The unpacked PyTorch tensor.
        """
        import torch

        arr = self.unpack_numpy(s, shape)
        return torch.from_numpy(arr)

    def unpack_tensorflow(self, s: struct.Struct, shape: Tuple):
        """
        Unpacks into a tensorflow tensor using struct format and shape
        
        Parameters
        ----------
        s : struct.Struct
            The struct format to use.
        shape : Tuple[int, ...]
            The shape of the TensorFlow tensor.
        
        Returns
        -------
        tensorflow.Tensor
            The unpacked TensorFlow tensor.
        """
        import tensorflow as tf

        arr = self.unpack_numpy(s, shape)
        return tf.constant(arr)

    def unpack(self, s: struct.Struct):
        """
        Unpacks data from the buffer using a given struct format.
        
        Parameters
        ----------
        s : struct.Struct
            The struct format to use for unpacking data.
        
        Returns
        -------
        Unpacked data as specified by the struct format.
        """
        unpack: tuple = s.unpack_from(self.buffer, self.read_offset)
        self.advance(s)
        if len(unpack) == 1:
            return unpack[0]
        return unpack

    def advance(self, s: struct.Struct, times=1):
        """
        Updates read_offset by number of times and size of given struct -> advances read offset in buffer
        
        Parameters
        ----------
        s : struct.Struct
            The struct format that determines the data size.
        times : int, optional
            The number of times to advance the read offset. Default is 1.
        """
        self.read_offset += s.size * times

    def unpack_str(self) -> str:
        """
        Unpacks a string from the buffer.
        
        Returns
        -------
        str
            The unpacked string, encoded in UTF-8.
        """
        length: int = self.unpack(ConstStructs.ushort)
        bytes_: bytes = self.unpack_f("%ds" % length)
        return bytes_.decode("utf-8")


if __name__ == "__main__":
    from tqdm import tqdm

    buffer = struct.pack("<H5s", 5, bytes("hello", 'utf8'))
    reader = BufferReader(buffer)

    for _ in tqdm(range(10000)):
        reader.read_offset = 0
        reader.unpack_str()

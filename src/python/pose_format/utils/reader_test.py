import os
import struct
from unittest import TestCase

import numpy as np
import torch

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import tensorflow as tf

from pose_format.utils.reader import BufferReader, ConstStructs


class TestBufferReader(TestCase):
    """ Tests for the BufferReader class"""

    def test_bytes_left(self):
        """ Test that bytes_left returns the correct number of bytes left to read"""
        reader = BufferReader(bytes(range(6)))
        reader.unpack_f("f")
        bytes_left = reader.bytes_left()
        self.assertEqual(bytes_left, 2)

    def test_advance(self):
        """ Test that advance advances the read_offset by the correct number of bytes"""
        reader = BufferReader(bytes())
        reader.advance(ConstStructs.float, 10)
        self.assertEqual(reader.read_offset, 40)

    def test_unpack(self):
        """ Test that unpack returns the correct value"""
        buffer = struct.pack("<f", 5.5)
        reader = BufferReader(buffer)
        unpacked_f = reader.unpack(ConstStructs.float)
        self.assertEqual(unpacked_f, 5.5)

    def test_unpack_f(self):
        """ Test that unpack_f returns the correct value"""
        buffer = struct.pack("<fh", 5.5, 3)
        reader = BufferReader(buffer)
        unpacked_f, unpacked_short = reader.unpack_f("fh")
        self.assertEqual(unpacked_f, 5.5)
        self.assertEqual(unpacked_short, 3)

    def test_unpack_str(self):
        """ Test that unpack_str returns the correct value"""
        s = "hello"
        buffer = struct.pack("<H%ds" % len(s), len(s), bytes(s, 'utf8'))
        reader = BufferReader(buffer)
        unpacked_s = reader.unpack_str()
        self.assertEqual(unpacked_s, s)

    def test_unpack_numpy(self):
        """ Test that unpack_numpy returns the correct value"""
        buffer = struct.pack("<ffff", 1., 2.5, 3.5, 4.5)
        reader = BufferReader(buffer)

        arr = reader.unpack_numpy(ConstStructs.float, (2, 2))

        res = np.array([[1., 2.5], [3.5, 4.5]])
        self.assertTrue(np.all(arr == res), msg="Numpy unpacked array is not equal to expected array")

    def test_unpack_numpy_writeable(self):
        """ Test that unpack_numpy returns a writeable array"""
        buffer = struct.pack("<ffff", 1., 2.5, 3.5, 4.5)
        reader = BufferReader(buffer)

        arr = reader.unpack_numpy(ConstStructs.float, (2, 2))

        # if array is read-only, this will raise a ValueError

        arr -= 0.1

    def test_unpack_torch(self):
        """ Test that unpack_torch returns the correct value"""
        buffer = struct.pack("<ffff", 1., 2.5, 3.5, 4.5)
        reader = BufferReader(buffer)

        arr = reader.unpack_torch(ConstStructs.float, (2, 2))

        res = torch.tensor([[1., 2.5], [3.5, 4.5]])
        self.assertTrue(torch.all(arr == res), msg="Torch unpacked array is not equal to expected array")

    def test_unpack_tensorflow(self):
        """ Test that unpack_tensorflow returns the correct value"""
        buffer = struct.pack("<ffff", 1., 2.5, 3.5, 4.5)
        reader = BufferReader(buffer)

        arr = reader.unpack_tensorflow(ConstStructs.float, (2, 2))

        res = tf.constant([[1., 2.5], [3.5, 4.5]])
        self.assertTrue(tf.reduce_all(tf.equal(arr, res)),
                        msg="Tensorflow unpacked array is not equal to expected array")

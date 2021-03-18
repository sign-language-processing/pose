import struct
from typing import List, Tuple, BinaryIO

import math

from .utils.reader import BufferReader, ConstStructs

VERSION = 0.1


class PoseNormalizationInfo:
    def __init__(self, p1: int, p2: int):
        self.p1 = p1
        self.p2 = p2


class PoseHeaderComponent:
    def __init__(self,
                 name: str,
                 points: List[str],
                 limbs: List[Tuple[int, int]],
                 colors: List[Tuple[int, int, int]],
                 point_format: str):
        self.name = name
        self.points = points
        self.limbs = limbs
        self.colors = colors
        self.format = point_format

        self.relative_limbs = self.get_relative_limbs()

    @staticmethod
    def read(version: float, reader: BufferReader):
        name = reader.unpack_str()
        point_format = reader.unpack_str()
        _points, _limbs, _colors = reader.unpack(ConstStructs.triple_ushort)
        points = [reader.unpack_str() for _ in range(_points)]
        limbs = [reader.unpack(ConstStructs.double_ushort) for _ in range(_limbs)]
        colors = reader.unpack_numpy(ConstStructs.ushort, (_colors, 3))

        return PoseHeaderComponent(name, points, limbs, colors, point_format)

    def _write_str(self, buffer: BinaryIO, s: str):
        buffer.write(struct.pack("<H%ds" % len(s), len(s), bytes(s, 'utf8')))

    def write(self, buffer: BinaryIO):
        self._write_str(buffer, self.name)  # Component Name
        self._write_str(buffer, self.format)  # Point Format

        # Lengths of points, limbs, and colors
        buffer.write(ConstStructs.triple_ushort.pack(len(self.points), len(self.limbs), len(self.colors)))

        for p in self.points:  # Names of Points
            self._write_str(buffer, p)

        for (p1, p2) in self.limbs:  # Indexes of Limbs
            buffer.write(ConstStructs.double_ushort.pack(p1, p2))

        for (r, g, b) in self.colors:  # RGB Colors
            buffer.write(ConstStructs.triple_ushort.pack(r, g, b))

    def get_relative_limbs(self):
        limbs_map = {p2: i for i, (p1, p2) in enumerate(self.limbs)}
        return [limbs_map[p1] if p1 in limbs_map else None for p1, p2 in self.limbs]


class PoseHeaderDimensions:
    def __init__(self, width: int, height: int, depth: int = 0, *args):
        self.width = math.ceil(width)
        self.height = math.ceil(height)
        self.depth = math.ceil(depth)

    @staticmethod
    def read(version: float, reader: BufferReader):
        width, height, depth = reader.unpack(ConstStructs.triple_ushort)
        return PoseHeaderDimensions(width, height, depth)

    def write(self, buffer: BinaryIO):
        buffer.write(ConstStructs.triple_ushort.pack(self.width, self.height, self.depth))


class PoseHeader:
    def __init__(self,
                 version: float,
                 dimensions: PoseHeaderDimensions,
                 components: List[PoseHeaderComponent],
                 is_bbox=False):
        self.version = version
        self.dimensions = dimensions
        self.components = components
        self.is_bbox = is_bbox

    @staticmethod
    def read(reader: BufferReader):
        version = reader.unpack(ConstStructs.float)
        dimensions = PoseHeaderDimensions.read(version, reader)

        _components = reader.unpack(ConstStructs.ushort)
        components = [PoseHeaderComponent.read(version, reader) for _ in range(_components)]

        return PoseHeader(version, dimensions, components)

    def write(self, buffer: BinaryIO):
        buffer.write(ConstStructs.float.pack(VERSION))  # File version
        self.dimensions.write(buffer)  # Width, Height, Depth
        buffer.write(ConstStructs.ushort.pack(len(self.components)))  # Number of components

        for component in self.components:
            component.write(buffer)

    def total_points(self):
        return sum(map(lambda c: len(c.points), self.components))

    def _get_point_index(self, component: str, point: str):
        idx = 0
        for c in self.components:
            if c.name == component:
                idx += c.points.index(point)
                return idx
            else:
                idx += len(c.points)

        raise ValueError("Couldn't find component")

    def normalization_info(self, p1: Tuple[str, str], p2: Tuple[str, str]):
        return PoseNormalizationInfo(p1=self._get_point_index(*p1), p2=self._get_point_index(*p2))

    def bbox(self):
        # Convert Header to boxes
        box_points = ['TOP_LEFT', 'BOTTOM_RIGHT']
        box_limbs = [(0, 1)]
        box_colors = [(255, 0, 0)]
        components = [PoseHeaderComponent(c.name, box_points, box_limbs, box_colors, c.format)
                      for c in self.components]

        return PoseHeader(self.version, self.dimensions, components, True)

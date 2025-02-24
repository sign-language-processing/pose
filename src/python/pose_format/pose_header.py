import hashlib
import math
import struct
from typing import BinaryIO, List, Tuple, Optional, Union

from .utils.reader import BufferReader, ConstStructs

VERSION = 0.2


class PoseNormalizationInfo:
    """ This class represents is used for normalization info for pose.
        
        Parameters
        ----------
        p1 : int
            First pose value
        p2 : int
            Second pose value.
        p3 : int, optional
            Third pose value. Defaults to None.
    """

    def __init__(self, p1: int, p2: int, p3: Optional[int] = None):
        """Initialize a PoseNormalizationInfo instance."""
        self.p1 = p1
        self.p2 = p2
        self.p3 = p3


class PoseHeaderComponent:
    """
    Class for pose header component

    Parameters
    ----------
    name : str
        Name of the pose header component
    points : List[str]
        List of point names.
    limbs : List[Tuple[int, int]]
        List of limb indices.
    colors : List[Tuple[int, int, int]]
        List of RGB colors for each limb.
    point_format : str
        Format for the points.

    Note
    ----
        Limbs and colors should have the same length. 
        The index in the limbs list corresponds to a color in the colors list.
    
    """

    def __init__(self, name: str, points: List[str], limbs: List[Tuple[int, int]], colors: List[Tuple[int, int, int]],
                 point_format: str):
        """
        Initializes PoseHeadComponent
        """
        self.name = name
        self.points = points
        self.limbs = limbs
        self.colors = colors
        self.format = point_format

        self.relative_limbs = self.get_relative_limbs()

    @staticmethod
    def read(version: float, reader: BufferReader) -> 'PoseHeaderComponent':
        """
        Reads pose header dimensions from reader (BufferReader).

        Parameters
        ----------
        version : float
            Version information.
        reader : BufferReader
            Reader object.

        Returns
        -------
        PoseHeaderDimensions
            instance of PoseHeaderDimensions.
        """
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
        """
        Writes pose header dimensions to a buffer (BinaryIO).

        Parameters
        ----------
        buffer : BinaryIO
            Buffer to write data info.

        Raises
        ------
        ValueError
            If dimension value is out of bounds.
        """
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
        """
        Get relative limbs mapping.

        Constructs a mapping from the second point in each limb tuple to its index in the limbs list. 
        Then, it attempts to map each first point in the limbs tuple to its corresponding index.

        Returns
        -------
        list
            List of relative limb indices or None if the limb does not have a relative mapping.

        Note
        ----
        returned list is based on the `self.limbs` of the instance, its structure is expected to be a list of tuples, where each tuple represents a limb with two points.

        """
        limbs_map = {p2: i for i, (p1, p2) in enumerate(self.limbs)}
        return [limbs_map[p1] if p1 in limbs_map else None for p1, p2 in self.limbs]

    def __str__(self):
        text = f"PoseHeaderComponent: {self.name}\n"
        text += f"  Format: {self.format}\n"
        text += f"  Points: {self.points}\n"
        text += f"  Limbs: {len(self.limbs)}\n"
        text += f"  Colors: {len(self.colors)}\n"
        return text


class PoseHeaderDimensions:
    """
    Represents width, height, and depth dimensions for a pose header.

    Parameters
    ----------
    width : int
        Width of the pose.
    height : int
        Height of the pose.
    depth : int
        Depth of the pose. Defaults to 0.

    Raises
    ------
    ValueError
        If any dimension value is out of bounds (0 to 65535).

    Examples
    --------
    >>> dimensions = PoseHeaderDimensions(10, 20, 5)
    >>> print(dimensions.width)
    10
    """

    def __init__(self, width: int, height: int, depth: int = 0, *args):
        self.width = math.ceil(width)
        self.height = math.ceil(height)
        self.depth = math.ceil(depth)

    @staticmethod
    def read(version: float, reader: BufferReader) -> 'PoseHeaderDimensions':
        """
        Reads and returns a PoseHeaderDimensions object from a buffer reader.

        Parameters
        ----------
        version : float
            Version of the data being read.
        reader : BufferReader
            The reader 

        Returns
        -------
        PoseHeaderDimensions
            Instance of PoseHeaderDimensions with its read dimensions (width, height, depth).
        """
        width, height, depth = reader.unpack(ConstStructs.triple_ushort)
        return PoseHeaderDimensions(width, height, depth)

    def write(self, buffer: BinaryIO):
        """
        Writes dimensions to a buffer.

        Parameters
        ----------
        buffer : BinaryIO
            Buffer to which dimensions (width, height, depth) will be written.

        Raises
        ------
        ValueError
            If any dimension value is out of bounds (0 to 65535).
        """
        if not (0 <= self.width <= (0x7fff * 2 + 1)):
            raise ValueError(f"Width must be between 0 and 65535. Got {self.width}")
        if not (0 <= self.height <= (0x7fff * 2 + 1)):
            raise ValueError(f"Height must be between 0 and 65535. Got {self.height}")
        if not (0 <= self.depth <= (0x7fff * 2 + 1)):
            raise ValueError(f"Depth must be between 0 and 65535. Got {self.depth}")

        buffer.write(ConstStructs.triple_ushort.pack(self.width, self.height, self.depth))

    def __str__(self):
        return f"PoseHeaderDimensions(width={self.width}, height={self.height}, depth={self.depth})"


class PoseHeaderCache:
    start_offset: int = None
    end_offset: int = None
    hash: str = None
    header: 'PoseHeader' = None

    @staticmethod
    def calc_hash(buffer: bytes):
        return hashlib.md5(buffer[PoseHeaderCache.start_offset:PoseHeaderCache.end_offset]).hexdigest()

    @staticmethod
    def check_cache(buffer: bytes) -> 'PoseHeader':
        if PoseHeaderCache.hash is None:
            return None

        if PoseHeaderCache.hash == PoseHeaderCache.calc_hash(buffer):
            return PoseHeaderCache.header

    @staticmethod
    def clear_cache():
        PoseHeaderCache.start_offset = None
        PoseHeaderCache.end_offset = None
        PoseHeaderCache.hash = None
        PoseHeaderCache.header = None

    @staticmethod
    def set_cache(header: 'PoseHeader', buffer: bytes, start_offset: int, end_offset: int):
        PoseHeaderCache.start_offset = start_offset
        PoseHeaderCache.end_offset = end_offset
        PoseHeaderCache.header = header
        PoseHeaderCache.hash = PoseHeaderCache.calc_hash(buffer)


class PoseHeader:
    """
    Main header for a pose.

    Parameters
    ----------
    version : float
        Version of the pose header.
    dimensions : PoseHeaderDimensions
        Dimensions of the pose header.
    components : List[PoseHeaderComponent]
        List of pose header components.
    is_bbox : bool, optional
        If bounding box needed. Default is False.
    Note
    ----
    - Use the `read` method to generate an instance from a BufferReader.
    - `total_points` method returns the total number of points across all components.
    - `num_dims` method returns the number of dimensions (X, Y, Z, ...).
    - Convert the header to bounding boxes using the `bbox` method.

    Examples
    --------
    >>> header = PoseHeader(1.0, PoseHeaderDimensions(10, 20, 5), [PoseHeaderComponent(...)], is_bbox=True)
    >>> print(header.is_bbox)
    True
    """

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
    def read(reader: BufferReader) -> 'PoseHeader':
        """
        Reads pose header data from a reader (BufferReader).


        Parameters
        ----------
        reader : BufferReader
            Reader object.

        Returns
        -------
        PoseHeader
            An instance of PoseHeader.
        """
        cached_header = PoseHeaderCache.check_cache(reader.buffer)
        if cached_header is not None:
            reader.read_offset = PoseHeaderCache.end_offset
            return cached_header

        start_offset = reader.read_offset
        version = reader.unpack(ConstStructs.float)
        dimensions = PoseHeaderDimensions.read(version, reader)

        _components = reader.unpack(ConstStructs.ushort)
        components = [PoseHeaderComponent.read(version, reader) for _ in range(_components)]
        end_offset = reader.read_offset

        pose_header = PoseHeader(version, dimensions, components)
        PoseHeaderCache.set_cache(pose_header, reader.buffer, start_offset, end_offset)

        return pose_header

    def write(self, buffer: BinaryIO):
        """
        Writes the pose header to a buffer (BinaryIO).

        Parameters
        ----------
        buffer : BinaryIO
            Buffer to write data into.
        """
        buffer.write(ConstStructs.float.pack(VERSION))  # File version
        self.dimensions.write(buffer)  # Width, Height, Depth
        buffer.write(ConstStructs.ushort.pack(len(self.components)))  # Number of components

        for component in self.components:
            component.write(buffer)

    def total_points(self):
        """
        Returns number of points

        Returns
        -------
        int
            Total number of points.
        """
        return sum(map(lambda c: len(c.points), self.components))

    def num_dims(self):
        """
        Returns number of dimensions

        Returns
        -------
        int
            Total number of dimensions (X, Y, Z, ...).
        """
        return max([len(c.format) for c in self.components]) - 1

    def _get_point_index(self, component: str, point: str):
        idx = 0
        for c in self.components:
            if c.name == component:
                idx += c.points.index(point)
                return idx
            else:
                idx += len(c.points)

        raise ValueError("Couldn't find component")

    def get_point_index(self, component: str, point: str) -> int:
        """
        Returns the index of a given point within the pose.

        Args:
            component (str): The name of the component containing the point.
            point (str): The name of the point whose index is to be retrieved.

        Raises:
            ValueError: If the specified component or point is not found.
        """
        return self._get_point_index(component, point)

    def normalization_info(self, p1: Tuple[str, str], p2: Tuple[str, str], p3: Tuple[str, str] = None):
        """
        Normalization info for given points.

        Parameters
        ----------
        p1 : Tuple[str, str]
            First point.
        p2 : Tuple[str, str]
            Second point.
        p3 : Tuple[str, str], optional
            Third point.

        Returns
        -------
        PoseNormalizationInfo
            Normalization information for the points.
        """
        return PoseNormalizationInfo(p1=self.get_point_index(*p1),
                                     p2=self.get_point_index(*p2),
                                     p3=None if p3 is None else self.get_point_index(*p3))

    def bbox(self):
        """
        Converts header to bounding boxes (bbox).

        Returns
        -------
        PoseHeader
            PoseHeader with bounding box information.
        """
        # Convert Header to boxes
        box_points = ['TOP_LEFT', 'BOTTOM_RIGHT']
        box_limbs = [(0, 1)]
        box_colors = [(255, 0, 0)]
        components = [PoseHeaderComponent(c.name, box_points, box_limbs, box_colors, c.format) for c in self.components]

        return PoseHeader(self.version, self.dimensions, components, True)

    def __str__(self):
        text = "PoseHeader\n"
        text += f"Version: {self.version}\n"
        text += str(self.dimensions) + "\n"
        text += f"Bounding Box: {self.is_bbox}\n"

        text += "Components:\n"
        for c in self.components:
            text += str(c) + "\n"
        return text

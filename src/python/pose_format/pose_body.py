import math
from random import sample
from typing import BinaryIO, List, Tuple, Optional

import numpy as np


from pose_format.pose_header import PoseHeader
from pose_format.utils.reader import BufferReader, ConstStructs

POINTS_DIMS = (2, 1, 0, 3)


class PoseBody:
    """
    Class for body data of a pose.

    Parameters
    ----------
    fps : float
        Frames per second.
    data: 
        Data in the format (Frames, People, Points, Dims) e.g., (93, 1, 137, 2).
    confidence: 
        Confidence data in the format (Frames, People, Points) e.g., (93, 1, 137).


    """
    tensor_reader = 'ABSTRACT-DO-NOT-USE'

    def __init__(self, fps: float, data, confidence):
        """Initialize a PoseBody instance."""
        self.fps = fps
        self.data = data  # Shape (Frames, People, Points, Dims) - eg (93, 1, 137, 2)
        self.confidence = confidence  # Shape (Frames, People, Points) - eg (93, 1, 137)

    @classmethod
    def read(cls, header: PoseHeader, reader: BufferReader, **kwargs) -> "PoseBody":
        """
        Reads pose data a buffer (BufferReader) based on the header's version.

        Parameters
        ----------
        header : PoseHeader
            Header containing the version of its pose data.
        reader : BufferReader
            Buffer from which to read the pose data.
        **kwargs : dict
            Additional parameters for reading specific versions.
        
        Returns
        -------
        PoseBody
            PoseBody object initialized with the read data.
        
        Raises
        ------
        NotImplementedError
            If header's version is not supported / unknown.
        """

        if header.version == 0:
            return cls.read_v0_0(header, reader, **kwargs)
        if round(header.version, 3) == 0.1:
            return cls.read_v0_1(header, reader, **kwargs)
        if round(header.version, 3) == 0.2:
            return cls.read_v0_2(header, reader, **kwargs)

        raise NotImplementedError("Unknown version - %f" % header.version)

    @classmethod
    def read_v0_0(cls, header: PoseHeader, reader: BufferReader, **unused_kwargs):
        """
        reads version 0.0 pose data.

        Parameters
        ----------
        header : PoseHeader
            Header containing the version of the pose data.
        reader : BufferReader
            Buffer from which to read the pose data.
        unused_kwargs : dict
            Unused additional parameters for this version.

        Raises
        ------
        NotImplementedError
            method for this version is not implemented.
        """
        raise NotImplementedError("'read_v0_0' not implemented on '%s'" % cls.__class__)

    @classmethod
    def read_v0_1_frames(cls,
                         frames: int,
                         shape: List[int],
                         reader: BufferReader,
                         start_frame: Optional[int] = None,
                         end_frame: Optional[int] = None):
        """
        Reads frame data for version 0.1 from a buffer.

        Parameters
        ----------
        frames : int
            Number of frames in the pose data.
        shape : List[int]
            Shape of the pose data.
        reader : BufferReader
            Buffer from which to read the pose data.
        start_frame : int, optional
            Index of the first frame to read. Default is None.
        end_frame : int, optional
            Index of the last frame to read. Default is None.

        Returns
        -------
        ndarray
            Array containing the pose data for the specified frames.
        
        Raises
        ------
        ValueError
            If start_frame is greater than number of frames.
        """
        tensor_reader = reader.__getattribute__(cls.tensor_reader)
        s = ConstStructs.float

        _frames = frames
        if start_frame is not None and start_frame > 0:
            if start_frame >= frames:
                raise ValueError(f"Start frame {start_frame} is greater than the number of frames {frames}")
            # Advance to the start frame
            reader.skip(s, int(np.prod((start_frame, *shape))))
            _frames -= start_frame

        remove_frames = None
        if end_frame is not None:
            end_frame = min(end_frame, frames)  # Do not allow overflow
            remove_frames = frames - end_frame
            _frames -= remove_frames

        tensor = tensor_reader(ConstStructs.float, shape=(_frames, *shape))

        if remove_frames is not None:
            reader.skip(s, int(np.prod((remove_frames, *shape))))

        return tensor

    @classmethod
    def read_v0_1(cls,
                  header: PoseHeader,
                  reader: BufferReader,
                  start_frame: Optional[int] = None,
                  end_frame: Optional[int] = None,
                  **unused_kwargs) -> "PoseBody":
        """
        Reads pose data for version 0.1 from a buffer.

        Parameters
        ----------
        header : PoseHeader
            Header containing the version of the pose data.
        reader : BufferReader
            Buffer from which to read the pose data.
        start_frame : int, optional
            Index of the first frame to read. Default is None.
        end_frame : int, optional
            Index of the last frame to read. Default is None.
        **unused_kwargs : dict
            Unused additional parameters for this version.

        Returns
        -------
        PoseBody
            PoseBody object initialized with the read data for version 0.1.
        """
        fps, _frames = reader.unpack(ConstStructs.double_ushort)

        _people = reader.unpack(ConstStructs.ushort)
        _points = sum(len(c.points) for c in header.components)
        _dims = header.num_dims()

        # _frames is defined as short, which sometimes is not enough! TODO change to int
        _frames = int(reader.bytes_left() / (_people * _points * (_dims + 1) * 4))

        data = cls.read_v0_1_frames(_frames, (_people, _points, _dims), reader, start_frame, end_frame)
        confidence = cls.read_v0_1_frames(_frames, (_people, _points), reader, start_frame, end_frame)

        return cls(fps, data, confidence)

    @classmethod
    def read_v0_2(cls,
                  header: PoseHeader,
                  reader: BufferReader,
                  start_frame: Optional[int] = None,
                  end_frame: Optional[int] = None,
                  start_time: Optional[int] = None,
                  end_time: Optional[int] = None,
                  **unused_kwargs) -> "PoseBody":
        """
        Reads pose data for version 0.2 from a buffer.

        Parameters
        ----------
        header : PoseHeader
            Header containing the version of the pose data.
        reader : BufferReader
            Buffer from which to read the pose data.
        start_frame : int, optional
            Index of the first frame to read. Default is None.
        end_frame : int, optional
            Index of the last frame to read. Default is None.
        start_time : int, optional
            Start time of the pose data (in milliseconds). Default is None.
        end_time : int, optional
            End time of the pose data (in milliseconds). Default is None.
        **unused_kwargs : dict
            Unused additional parameters for this version.

        Returns
        -------
        PoseBody
            PoseBody object initialized with the read data for version 0.2.
        """

        if start_time is not None and start_frame is not None:
            raise ValueError("Cannot specify both start_time and start_frame")
        if end_time is not None and end_frame is not None:
            raise ValueError("Cannot specify both end_time and end_frame")

        fps = reader.unpack(ConstStructs.float)  # Changed from v0.1, uint -> float
        _frames = reader.unpack(ConstStructs.uint)  # Changed from v0.1, ushort -> uint

        _people = reader.unpack(ConstStructs.ushort)
        _points = sum([len(c.points) for c in header.components])
        _dims = header.num_dims()

        if start_time is not None:
            start_frame = math.floor(start_time / 1000 * fps)
        if end_time is not None:
            end_frame = math.ceil(end_time / 1000 * fps)

        data = cls.read_v0_1_frames(_frames, (_people, _points, _dims), reader, start_frame, end_frame)
        confidence = cls.read_v0_1_frames(_frames, (_people, _points), reader, start_frame, end_frame)

        return cls(fps, data, confidence)

    def write(self, version: float, buffer: BinaryIO):
        """
        Writes  data to a file based on version of spec: in docs/spec.

        Parameters
        ----------
        version : float
            Version of the pose data to write.
        buffer : BinaryIO
            Buffer to write the pose data to.
        """
        raise NotImplementedError("'write' not implemented on '%s'" % self.__class__)
    
    def copy(self)->"PoseBody":
        return self.__class__(fps=self.fps,
                          data=self.data,
                          confidence=self.confidence)

    def __getitem__(self, index):
        """
        Gets a version of the PoseBody data and confidence based on the provided index.

        Parameters
        ----------
        index : int or slice
            Index or slice to get data.

        Returns
        -------
        PoseBody
            PoseBody object with the sliced data and confidence.
        """
        # Get the sliced data and confidence
        sliced_data = self.data[index]
        sliced_confidence = self.confidence[index]

        # Create a new PoseBody object with the sliced data and confidence
        return type(self)(self.fps, sliced_data, sliced_confidence)

    def numpy(self):
        """
        Convert the current PoseBody representation to NumpyPoseBody.

        Returns
        -------
        NumpyPoseBody
            The converted PoseBody object.
        
        Raises
        ------
        NotImplementedError
            If numpy is not implemented.
        """
        raise NotImplementedError("'numpy' not implemented on '%s'" % self.__class__)

    def torch(self):
        """
        Converts current PoseBody to TorchPoseBody.

        Returns
        -------
        TorchPoseBody
            The converted PoseBody object.

        Raises
        ------
        NotImplementedError
            If torch is not implemented.
        """
        raise NotImplementedError("'torch' not implemented on '%s'" % self.__class__)

    def tensorflow(self):
        """
        Converts current PoseBody representation to TensorflowPoseBody.

        Returns
        -------
        TensorflowPoseBody
            Converted PoseBody object.

        Raises
        ------
        NotImplementedError
            If tensorflow is not implemented.
        """
        raise NotImplementedError("'tensorflow' not implemented on '%s'" % self.__class__)

    def flatten(self):
        """
        Converts data from the (Frames, People, Points, Dims) masked representation to an array of points.
        
        Every item in the result array contains the following dimensions:
        0. Time in milliseconds
        1. Person ID
        2. Point ID
        3. X dimension
        4. Y dimension
        5. Z dimension (if exists)
        6. Pose estimation confidence

        Returns
        -------
        np.ndarray
            Array of points with detailed dimensions.

        Raises
        ------
        NotImplementedError
            If the method is not implemented for the specific class.
        """
        raise NotImplementedError("'flatten' not implemented on '%s'" % self.__class__)

    def slice_step(self, by: int) -> "PoseBody":
        """
        Slices data by skipping rows. This affects the fps (frames per seconds).

        Parameters
        ----------
        by : int
            Take one row every "by" rows.

        Returns
        -------
        PoseBody
            PoseBody instance with sliced data.
        """
        new_data = self.data[::by]
        new_confidence = self.confidence[::by]
        new_fps = self.fps / by

        return self.__class__(fps=new_fps, data=new_data, confidence=new_confidence)

    def augment2d(self, rotation_std=0.2, shear_std=0.2, scale_std=0.2):
        """
        Augment 2D data with given standard deviations.

        Parameters
        ----------
        rotation_std : float, optional
            Rotation in radians. Default is 0.2.
        shear_std : float, optional
            Shear X in percent. Default is 0.2.
        scale_std : float, optional
            Scale X in percent. Default is 0.2.

        Returns
        -------
        PoseBody
            Augmented PoseBody instance.
        
        Note
        ----
        - The method modifies the PoseBody based on shear, rotation, and scaling.
        - **shear_std** based on https://en.wikipedia.org/wiki/Shear_matrix
        - **rotation_std** based on https://en.wikipedia.org/wiki/Rotation_matrix 
        - **scale_std** based on https://en.wikipedia.org/wiki/Scaling_(geometry)
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
        """
        Creates a new PoseBody instance with data replaced by zeros.

        Returns
        -------
        PoseBody
            PoseBody instance with zero-filled data.

        Raises
        ------
        NotImplementedError
            If the zero_filled is not implemented on class .
        """
        raise NotImplementedError("'zero_filled' not implemented on '%s'" % self.__class__)

    def matmul(self, matrix: np.ndarray) -> __qualname__:
        """
        Multiplies PoseBody data with a numpy.ndarray matrix.
        
        Parameters
        ----------
        matrix : np.ndarray
            The matrix to multiply the PoseBody data with.
            
        Returns
        -------
        PoseBody
            PoseBody instance with data multiplied by a numpy array.
        
        
        Raises
        ------
        NotImplementedError
            If the matmul is not implemented in class.
        """
        raise NotImplementedError("'matmul' not implemented on '%s'" % self.__class__)

    def get_points(self, indexes: List[int]) -> __qualname__:
        """
        Get points from PoseBody.
        
        Parameters
        ----------
        indexes : List[int]
            List of point indices to get from PoseBody.
            
        Returns
        -------
        PoseBody
            PoseBody instance containing only chosen points.
             
        Raises
        ------
        NotImplementedError
            If the `get_points` is not implemented in class.
        """
        raise NotImplementedError("'get_points' not implemented on '%s'" % self.__class__)

    def bbox(self, header: PoseHeader) -> __qualname__:
        """
        For computing bounding box of PoseBody.
        
        Parameters
        ----------
        header : PoseHeader
            Header containing the version of the pose data.
            
        Returns
        -------
        PoseBody
            PoseBody instance with bounding box.
        
        Raises
        ------
        NotImplementedError
            If the `bbox` is not implemented in class.
        """

        raise NotImplementedError("'bbox' not implemented on '%s'" % self.__class__)

    def points_perspective(self):
        """
        Give points in PoseBody as a perspective view.
        
        Returns
        -------
        PoseBody
            PoseBody instance with points adjusted for perspective.
        
        Raises
        ------
        NotImplementedError
            If the method is not implemented for the specific class.
        """
        raise NotImplementedError("'points_perspective' not implemented on '%s'" % self.__class__)

    def select_frames(self, frame_indexes: List[int]) -> "PoseBody":
        """
        Selects specific frames from PoseBody object.

        Parameters
        ----------
        frame_indexes : List[int]
            List of frame indexes to select.

        Returns
        -------
        PoseBody
            PoseBody object containing only the selected frames.

        Raises
        ------
        IndexError
            If any of the specified frame indices are out of the valid range for the current PoseBody data.
        """
        data = self.data[frame_indexes]
        confidence = self.confidence[frame_indexes]
        return self.__class__(fps=self.fps, data=data, confidence=confidence)

    def frame_dropout_given_percent(self, dropout_percent: float) -> Tuple["PoseBody", List[int]]:
        """
        Drop of frames based on  given dropout percentage.

        Parameters
        ----------
        dropout_percent : float
            Percentage of frames to drop. Between 0 and 1 (e.g., 0.2 means drop 20% of the frames).

        Returns
        -------
        Tuple[PoseBody, List[int]]
            - New PoseBody object with the gotten frames.
            - List of frame indexes.
        
        Note
        ----
        Actual number of dropped frames might be slightly different due to rounding!
        """

        data_len = len(self.data)
        dropout_number = min(int(data_len * dropout_percent), int(data_len * 0.99))
        dropout_indexes = set(sample(range(0, data_len), dropout_number))
        select_indexes = [i for i in range(0, data_len) if i not in dropout_indexes]

        return self.select_frames(select_indexes), select_indexes

    def frame_dropout_uniform(self, dropout_min: float = 0.2, dropout_max: float = 1.0) -> Tuple["PoseBody", List[int]]:
        """
        Randomly drops frames depending on a uniform distribution - given minimum and maximum percentages.

        Parameters
        ----------
        dropout_min : float, optional
            Minimum percentage of frames to drop. Default is 0.2.
        dropout_max : float, optional
            Maximum percentage of frames to drop. Default is 1.0.

        Returns
        -------
        Tuple[PoseBody, List[int]]
            - New PoseBody object with dropped frames.
            - List of frame indexes that were retained.
        """
        dropout_percent = np.random.uniform(low=dropout_min, high=dropout_max, size=1)[0]

        return self.frame_dropout_given_percent(dropout_percent)

    def frame_dropout_normal(self, dropout_mean: float = 0.5, dropout_std: float = 0.1) -> Tuple["PoseBody", List[int]]:
        """
        drop frames depending on normal distribution with given mean and standard deviation.

        Parameters
        ----------
        dropout_mean : float, optional
            Mean percentage of frames to drop. Default is 0.5.
        dropout_std : float, optional
            Standard deviation of percentage of frames to drop. Default is 0.1.

        Returns
        -------
        Tuple[PoseBody, List[int]]
            - New PoseBody object with dropped frames.
            - List of retrieved frame indexes.
        """
        dropout_percent = np.abs(np.random.normal(loc=dropout_mean, scale=dropout_std, size=1))[0]

        return self.frame_dropout_given_percent(dropout_percent)

    def __str__(self):
        text = f"{self.__class__.__name__}\n"
        text += f"FPS: {self.fps}\n"
        text += f"Data: {type(self.data)} {self.data.shape}, {self.data.dtype}\n"
        text += f"Confidence shape: {type(self.confidence)} {self.confidence.shape}, {self.data.dtype}\n"
        text += f"Duration (seconds): {len(self.data) / self.fps}\n"
        return text

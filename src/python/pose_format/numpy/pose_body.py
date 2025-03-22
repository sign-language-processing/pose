from typing import BinaryIO, List, Union

import numpy as np
import numpy.ma as ma

from ..pose_body import POINTS_DIMS, PoseBody
from ..pose_header import PoseHeader
from ..utils.reader import BufferReader, ConstStructs

# import numpy as np
# np.seterr(all='raise')


class NumPyPoseBody(PoseBody):
    """
    Represents pose information leveraging NumPy operations and structures.

     * Inherits from:  `PoseBody`
     * Implements pose info using NumPy operations and structures.
     * Provides method for operations:  matrix, multiplication, interpolation and data type conversions 

    The `NumPyPoseBody` is an implementation of  `PoseBody` base class. 
    This subclass uses NumPy masked arrays to handle pose data. 
    Makes it suitable for applications where you need NumPy-based operations. 
    The masked arrays allow for efficient handling of missing or invalid pose values

    The class also comes with methods to transform, modify, and operate on pose data, including matrix multiplication, interpolation, 
    and conversions to other data types like PyTorch tensors or TensorFlow tensors
    
    Parameters
    ----------
    fps : float
        Frames per second, to represent the temporal aspect of pose data.
    data : Union[ma.MaskedArray, np.ndarray]
        Pose data either as a masked array or a regular numpy array.
    confidence : np.ndarray
        confidence array of the pose keypoints.
    """

    """Specifies the method name for unpacking a numpy array (Value: 'unpack_numpy')."""
    tensor_reader = 'unpack_numpy'

    def __init__(self, fps: float, data: Union[ma.MaskedArray, np.ndarray], confidence: np.ndarray):
        """
        Initializes the NumPyPoseBody instance
        """
        if isinstance(data, np.ndarray):  # If array is not masked
            mask = confidence == 0  # 0 means no-mask, 1 means with-mask
            stacked_mask = np.stack([mask] * data.shape[-1], axis=3)
            data = ma.masked_array(data, mask=stacked_mask)

        super().__init__(fps, data, confidence)

    @classmethod
    def read_v0_0(cls, header: PoseHeader, reader: BufferReader, **unused_kwargs):
        """
        Reads pose data from a given buffer reader using a specified data format version (see: ``docs/specs``).

        Parameters
        ----------
        header : PoseHeader
            Pose header information
        reader : BufferReader
            binary buffer reader

        Returns
        -------
        NumPyPoseBody
            Instance of NumPyPoseBody with read pose data.
        """
        fps, _frames = reader.unpack(ConstStructs.double_ushort)

        _dims = max([len(c.format) for c in header.components]) - 1
        _points = sum([len(c.points) for c in header.components])

        frames_d = []
        frames_c = []
        for _ in range(_frames):
            _people = reader.unpack(ConstStructs.ushort)
            people_d = []
            people_c = []
            for pid in range(_people):
                reader.advance(ConstStructs.short)  # Skip Person ID
                person_d = []
                person_c = []
                for component in header.components:
                    points = np.array(
                        reader.unpack_numpy(ConstStructs.float, (len(component.points), len(component.format))))
                    dimensions, confidence = np.split(points, [-1], axis=1)
                    boolean_confidence = np.where(confidence > 0, 0, 1)  # To create the mask
                    mask = np.column_stack(tuple([boolean_confidence] * (len(component.format) - 1)))

                    person_d.append(ma.masked_array(dimensions, mask=mask))
                    person_c.append(np.squeeze(confidence, axis=-1))

                if pid == 0:
                    people_d.append(ma.concatenate(person_d))
                    people_c.append(np.concatenate(person_c))

            # In case no person, should all be zeros
            if len(people_d) == 0:
                people_d.append(np.zeros((_points, _dims)))
                people_c.append(np.zeros(_points))

            frames_d.append(ma.stack(people_d))
            frames_c.append(np.stack(people_c))

        return cls(fps, ma.stack(frames_d), ma.stack(frames_c))

    def write(self, version: float, buffer: BinaryIO):
        """
        Writes pose data to a binary buffer using specified data format version.

        Parameters
        ----------
        version : float
            Version of the data format.
        buffer : BinaryIO
            The binary buffer to write to.
        """
        _frames, _people, _points, _dims = self.data.shape
        if _frames > 4_294_967_295: # about 4.5 years of video at 30fps
            raise ValueError("Too many frames to write. Maximum is 2^32 - 1.")
        buffer.write(ConstStructs.float.pack(self.fps))
        buffer.write(ConstStructs.uint.pack(_frames))
        buffer.write(ConstStructs.ushort.pack(_people))

        buffer.write(np.array(self.data.data, dtype=np.float32).tobytes())
        buffer.write(np.array(self.confidence, dtype=np.float32).tobytes())

    def copy(self) -> 'NumPyPoseBody':
        return type(self)(fps=self.fps,
                          data=self.data.copy(),
                          confidence=self.confidence.copy())

    @property
    def mask(self):
        """ Returns  mask associated with data. """
        return self.data.mask

    def torch(self):
        """
        converts current instance into a TorchPoseBody instance.

        Returns
        -------
        TorchPoseBody
            The pose body data represented in PyTorch tensors.
        """
        try:
            import torch
        except ImportError:
            raise ImportError("Please install torch. https://pytorch.org/")

        import torch

        from ..torch.pose_body import TorchPoseBody

        torch_confidence = torch.from_numpy(self.confidence)
        torch_data = torch.from_numpy(self.data.data)
        return TorchPoseBody(self.fps, torch_data, torch_confidence)

    def tensorflow(self):
        """
        converts current instance into a TensorflowPoseBody instance

        Returns
        -------
        TensorflowPoseBody
            pose body data represented in TensorFlow tensors
        """
        import tensorflow

        from ..tensorflow.pose_body import TensorflowPoseBody

        tf_confidence = tensorflow.constant(self.confidence)
        tf_data = tensorflow.constant(self.data.data)
        return TensorflowPoseBody(self.fps, tf_data, tf_confidence)

    def zero_filled(self):
        """
        fills missing values with zeros.

        Returns
        -------
        NumPyPoseBody
            changed pose body data.
        """
        copy = self.copy()
        copy.data = ma.array(copy.data.filled(0), mask=copy.data.mask)
        return copy

    def matmul(self, matrix: np.ndarray):
        """
        Performs matrix multiplication on pose data.

        Parameters
        ----------
        matrix : np.ndarray
            matrix to multiply the pose data with

        Returns
        -------
        NumPyPoseBody
            transformed pose body data
        """
        data = ma.dot(self.data, matrix)
        return NumPyPoseBody(self.fps, data, self.confidence)

    def flip(self, axis=0):
        """
        flips pose data across a specified axis

        Parameters
        ----------
        axis : int, optional
            axis along which the pose data should be flipped.

        Returns
        -------
        NumPyPoseBody
            flipped pose body data
        """
        vec = np.ones(self.data.shape[-1])
        vec[axis] = -1

        data = self.data * vec
        return NumPyPoseBody(self.fps, data, self.confidence)

    def points_perspective(self):
        """
        Transforms pose data to get a perspective based on points.

        Returns
        -------
        ma.MaskedArray
            Transformed pose data
        """
        return ma.transpose(self.data, axes=POINTS_DIMS)

    def get_points(self, indexes: List[int]):
        """
        Get points (keypoints) based on given indexes.

        Parameters
        ----------
        indexes : List[int]
             List of indices representing the keypoints to get.

        Returns
        -------
        NumPyPoseBody
            Pose body data containing only a specified points.
        """
        data = ma.transpose(self.data, axes=POINTS_DIMS)
        new_data = ma.transpose(data[indexes], axes=POINTS_DIMS)

        confidence_reshape = (2, 1, 0)
        confidence = np.transpose(self.confidence, axes=confidence_reshape)
        new_confidence = np.transpose(confidence[indexes], axes=confidence_reshape)

        return NumPyPoseBody(self.fps, new_data, new_confidence)

    def bbox(self, header: PoseHeader):
        """
        Computes the bounding boxes for each component based on the pose data.

        Parameters
        ----------
        header : PoseHeader
            Pose header information.

        Returns
        -------
        NumPyPoseBody
            Pose body data representing bounding boxes.
        """
        data = ma.transpose(self.data, axes=POINTS_DIMS)

        # Split data by components, `ma` doesn't support ".split"
        components = []
        idx = 0
        for component in header.components:
            components.append(data[list(range(idx, idx + len(component.points)))])
            idx += len(component.points)

        boxes = [ma.stack([ma.min(c, axis=0), ma.max(c, axis=0)]) for c in components]
        boxes_cat = ma.concatenate(boxes)
        if type(boxes_cat.mask) == np.bool_:  # Sometimes, it doesn't concatenate the mask...
            boxes_mask = ma.concatenate([b.mask for b in boxes])
            boxes_cat = ma.array(boxes_cat, mask=boxes_mask)

        new_data = ma.transpose(boxes_cat, axes=POINTS_DIMS)

        confidence_mask = np.split(new_data.mask, [-1], axis=3)[0]
        confidence_mask = np.squeeze(confidence_mask, axis=-1)
        confidence = np.where(confidence_mask == True, 0, 1)

        return NumPyPoseBody(self.fps, new_data, confidence)

    def interpolate(self, new_fps: int = None, kind='cubic'):
        """
        Interpolates the pose data to match a new frame rate.

        Parameters
        ----------
        new_fps : int, optional
            The desired frame rate for interpolation.
        kind : str, optional
            The type of interpolation. Options include: "linear", "quadratic", and "cubic".

        Returns
        -------
        NumPyPoseBody
            Interpolated pose body data.
        """
        try:
            from scipy.interpolate import interp1d
        except ImportError:
            raise ImportError("Please install scipy with: pip install scipy")

        if new_fps is None:
            new_fps = self.fps

        _frames = self.data.shape[0]
        if _frames == 1:
            raise ValueError("Can't interpolate single frame")

        _new_frames = round(_frames * new_fps / self.fps)
        steps = np.linspace(0, 1, _frames)
        new_steps = np.linspace(0, 1, _new_frames)

        transposed = self.points_perspective()  # (points, people, frames, dims)
        masked_confidence = ma.array(self.confidence, mask=self.confidence == 0)
        confidence = ma.expand_dims(masked_confidence.transpose(), axis=3)  # (points, people, frames, 1)
        points = ma.concatenate([transposed, confidence], axis=3)

        new_people = []
        for people in points:
            new_frames = []
            for frames in people:
                mask = frames.transpose()[-1].mask # takes mask from confidence value

                partial_steps = ma.array(steps, mask=mask).compressed()

                if partial_steps.shape[0] == 0:  # No data for this point
                    new_frames.append(np.zeros((_new_frames, frames.shape[1])))
                else:
                    partial_frames = frames.compressed().reshape(partial_steps.shape[0], frames.shape[1])

                    if len(partial_steps) == 1:
                        f = lambda l: partial_frames
                    else:
                        this_kind = kind if len(partial_steps) > 3 \
                            else "quadratic" if len(partial_steps) > 2 and kind == "cubic" \
                            else "linear"  # Can't do something fancy for 2 points
                        f = interp1d(partial_steps, partial_frames, axis=0, kind=this_kind)

                    first_step = partial_steps[0]
                    last_step = partial_steps[-1]
                    if first_step == 0 and last_step == 1:
                        new_frames.append(f(new_steps))
                    else:
                        first_step_where = np.argwhere(new_steps >= first_step)
                        first_step_index = first_step_where[0][0] if len(first_step_where) > 0 else 0

                        last_step_where = np.argwhere(new_steps > last_step)
                        last_step_index = last_step_where[0][0] if len(last_step_where) > 0 else len(new_steps)

                        if first_step_index == last_step_index:
                            new_frames.append(np.zeros((len(new_steps), frames.shape[1])))
                        else:
                            frame_data = f(new_steps[first_step_index:last_step_index])
                            new_frames.append(
                                np.concatenate([
                                    np.zeros((first_step_index, frames.shape[1])),
                                    np.array(frame_data),
                                    np.zeros((len(new_steps) - last_step_index, frames.shape[1]))
                                ]))
            new_people.append(np.stack(new_frames, axis=0))

        new_data = np.stack(new_people, axis=0).transpose([2, 1, 0, 3])
        dimensions, confidence = np.split(new_data, [-1], axis=3)
        confidence = np.squeeze(confidence, axis=3)

        return NumPyPoseBody(fps=new_fps, data=dimensions, confidence=confidence)

    def flatten(self):
        """
        Flattens data and confidence arrays.

        method reshapes data and confidence arrays to a two-dimensional array.
        The flattened array is filtered to remove rows where confidence is zero.

        Returns
        -------
        numpy.ndarray
            flattened and filtered version of the data array.

        """
        shape = self.data.shape
        data = self.data.data.reshape(-1, shape[-1])  # Not masked data
        confidence = self.confidence.flatten()
        indexes = list(np.ndindex(shape[:-1]))
        flat = np.c_[indexes, confidence, data]
        # Filter data from flat
        flat = flat[confidence != 0]
        # Scale the first axis by fps
        scalar = np.ones(len(shape) + shape[-1])
        scalar[0] = 1 / self.fps
        return flat * scalar

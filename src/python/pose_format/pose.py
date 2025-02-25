from io import BytesIO
from itertools import chain
from typing import BinaryIO, Dict, List, Tuple, Type, Union

import numpy as np
import numpy.ma as ma
from pose_format.numpy import NumPyPoseBody
from pose_format.pose_body import PoseBody
from pose_format.pose_header import (PoseHeader, PoseHeaderComponent,
                                     PoseHeaderDimensions,
                                     PoseNormalizationInfo, PoseHeaderCache)
from pose_format.utils.fast_math import distance_batch
from pose_format.utils.reader import BufferReader, BytesIOReader



class Pose:
    """
    File IO for '.pose' file format, including the header and body.
    
    Parameters
    ----------
    header : PoseHeader
        Header information for the pose.
    body : PoseBody
        Body data for the pose.
    """

    def __init__(self, header: PoseHeader, body: PoseBody):
        self.header = header
        self.body = body

    @staticmethod
    def read(buffer: Union[bytes, BytesIO], pose_body: Type[PoseBody] = NumPyPoseBody, **kwargs):
        """
        Read Pose object from buffer.

        Parameters
        ----------
        buffer : bytes
            The input buffer.
        pose_body : Type[PoseBody], optional
            The type of pose body to be read. Defaults to NumPyPoseBody.

        Returns
        -------
        Pose
            Pose object.
        """

        # Use BytesIO reader optimization only when start/end is specified, otherwise, it is faster to read from buffer
        if isinstance(buffer, bytes):
            reader = BufferReader(buffer)
        else:
            if kwargs.get("start_frame", None) or kwargs.get("end_frame", None) or kwargs.get("start_time", None) or kwargs.get("end_time", None):
                reader = BytesIOReader(buffer)
            else:
                reader = BufferReader(buffer.read())

        reader.expect_to_read((PoseHeaderCache.end_offset or 10 * 1024) + 100) # Expect to read the header at least (or 10kb)
        header = PoseHeader.read(reader)
        body = pose_body.read(header, reader, **kwargs)

        return Pose(header, body)

    def write(self, buffer: BinaryIO):
        """
        Write Pose object to buffer.

        Parameters
        ----------
        buffer : BinaryIO
            buffer
        """

        # Sanity check: The body should have 4 dimensions
        if len(self.body.data.shape) != 4:
            raise ValueError(f"Body data should have 4 dimensions, not {len(self.body.data.shape)}")

        # Sanity check: Body should have as many dimensions as header
        header_dims = self.header.num_dims()
        body_dims = self.body.data.shape[-1]
        if header_dims != body_dims:
            raise ValueError(f"Header has {header_dims} dimensions, but body has {body_dims}")

        self.header.write(buffer)
        self.body.write(self.header.version, buffer)

    def focus(self):
        """
        Gets the pose to start at (0,0) and have dimensions as big as needed
        """
        mins = ma.min(self.body.data, axis=(0, 1, 2))
        maxs = ma.max(self.body.data, axis=(0, 1, 2))

        if np.count_nonzero(mins) > 0:  # Only translate if there is a number to translate by
            self.body.data = ma.subtract(self.body.data, mins)

        dimensions = (maxs - mins).tolist()
        self.header.dimensions = PoseHeaderDimensions(*dimensions)

    def normalize(self, info: Union[PoseNormalizationInfo,None]=None, scale_factor: float = 1) -> "Pose":
        """
        Normalize the points to a fixed distance between two particular points.

        Parameters
        ----------
        info : PoseNormalizationInfo
            Information for normalization.
        scale_factor : float, optional
            Scaling factor. Defaults to 1.

        Returns
        -------
        Pose
            The normalized Pose object.
        """
        if info is None:
            from pose_format.utils.generic import pose_normalization_info
            info = pose_normalization_info(self.header)

        transposed = self.body.points_perspective()

        p1s = transposed[info.p1]
        p2s = transposed[info.p2]

        # Move all points so center is (0,0)
        center = ((p2s + p1s) / 2).mean(axis=(0, 1))

        self.body.data -= center

        mean_distance = distance_batch(p1s, p2s).mean()

        # scale all points to dist/scale
        scale = scale_factor / mean_distance

        self.body.data = self.body.data * scale

        return self

    def normalize_distribution(self, mu=None, std=None, axis=(0, 1)):
        """
        Normalize points distribution.

        Parameters
        ----------
        mu : np.ndarray, optional
            Mean values for normalization. If None, it will be computed.
        std : np.ndarray, optional
            Standard deviation values for normalization. If None, it will be computed.
        axis : tuple of int, optional
            Axes for mean and std computation. Defaults to (0, 1).

        Returns
        -------
        tuple of np.ndarray
            Calculated mean and standard deviation.
        """

        mu = mu if mu is not None else self.body.data.mean(axis=axis)
        std = std if std is not None else self.body.data.std(axis=axis)

        self.body.data = (self.body.data - mu) / std

        return mu, std

    def unnormalize_distribution(self, mu, std):
        """
        Given mean, standard deviationn unnormalization applied to the pose points distribution.

        Parameters
        ----------
        mu : np.ndarray
            The mean values used for normalization.
        std : np.ndarray
            The standard deviation values used for normalization.
        """
        self.body.data = (self.body.data * std) + mu

    def frame_dropout_uniform(self, dropout_min: float = 0.2, dropout_max: float = 1.0) -> Tuple["Pose", List[int]]:
        """
        Perform uniform frame dropout on Pose

        Parameters
        ----------
        dropout_min : float, optional
            Minimum dropout value. Defaults to 0.2.
        dropout_max : float, optional
            Maximum dropout value. Defaults to 1.0.

        Returns
        -------
        tuple
            a tuple containing Pose with dropped frames and a list of selected indexes.
        """
        body, selected_indexes = self.body.frame_dropout_uniform(dropout_min=dropout_min, dropout_max=dropout_max)
        return Pose(header=self.header, body=body), selected_indexes

    def frame_dropout_normal(self, dropout_mean: float = 0.5, dropout_std: float = 0.1) -> Tuple["Pose", List[int]]:
        """
        Normal frame dropout on Pose.

        Parameters
        ----------
        dropout_mean : float, optional
            Mean value for dropout. Defaults to 0.5.
        dropout_std : float, optional
            Standard deviation for dropout. Defaults to 0.1.

        Returns
        -------
        tuple
            a tuple with Pose of dropped frames and a list of selected indexes.
        """
        body, selected_indexes = self.body.frame_dropout_normal(dropout_mean=dropout_mean, dropout_std=dropout_std)
        return Pose(header=self.header, body=body), selected_indexes
    
    
    def remove_components(self, components_to_remove: Union[str, List[str]], points_to_remove: Union[Dict[str, List[str]],None] = None):
        
        if isinstance(components_to_remove, str):
            components_to_remove = [components_to_remove]

        components_to_keep = []
        points_dict = {}

        for component in self.header.components:
            if component.name not in components_to_remove:
                components_to_keep.append(component.name)
                if points_to_remove:
                    points_to_remove_list = points_to_remove.get(component.name, []) 
                    points_dict[component.name] = [point for point in component.points if point not in points_to_remove_list]
                else:
                    points_dict[component.name] = component.points[:]

        return self.get_components(components_to_keep, points_dict)
        
    

    def get_components(self, components: List[str], points: Union[Dict[str, List[str]],None] = None):
        """
        get pose components based on criteria.

        Parameters
        ----------
        components : List[str]
            List of component names to get.
        points : Dict[str, List[str]], optional
            Mapping of component names to lists of point names to get.

        Returns
        -------
        Pose
            Pose object containing new components
        """
        indexes = {}
        new_components = {}

        idx = 0
        for component in self.header.components:
            if component.name in components:
                new_component = PoseHeaderComponent(component.name, component.points, component.limbs, component.colors,
                                                    component.format)
                if points is not None and component.name in points:  # copy and permute points
                    new_component.points = points[component.name]
                    point_index_mapping = {
                        component.points.index(point): i for i, point in enumerate(new_component.points)
                    }
                    old_indexes_set = set(point_index_mapping.keys())
                    new_component.limbs = [(point_index_mapping[l1], point_index_mapping[l2])
                                           for l1, l2 in component.limbs
                                           if l1 in old_indexes_set and l2 in old_indexes_set]

                    indexes[component.name] = [idx + component.points.index(p) for p in new_component.points]
                else:  # Copy component as is
                    indexes[component.name] = list(range(idx, len(component.points) + idx))

                new_components[component.name] = new_component

            idx += len(component.points)

        new_components_order = [new_components[c] for c in components]
        indexes_order = [indexes[c] for c in components]

        new_header = PoseHeader(self.header.version, self.header.dimensions, new_components_order)
        flat_indexes = list(chain.from_iterable(indexes_order))
        new_body = self.body.get_points(flat_indexes)

        return Pose(header=new_header, body=new_body)
    

    def copy(self):
        return self.__class__(self.header, self.body.copy())

    def bbox(self):
        """
        Calculates bounding box for Pose.

        Returns
        -------
        Pose
            Pose object representing bounding box (bbox).
        """
        body = self.body.bbox(self.header)
        header = self.header.bbox()
        return Pose(header=header, body=body)

    pass_through_methods = {
        "augment2d",  # Augment 2D points
        "flip",  # Flip pose on axis
        "interpolate",  # Interpolate missing pose points
        "torch",  # Convert body to torch
        "tensorflow",  # Convert body to tensorflow
        "slice_step",  # Step through the data
    }
    """
A set of method names which define actions that can be applied to the pose data.

    Parameters
    ----------
    augment2d : str
        Represents a method to augment 2D points.
    flip : str
        Represents a method to flip the pose on an axis.
    interpolate : str
        Represents a method to interpolate missing pose points.
    torch : str
        Represents a method to convert the body data to torch format.
    tensorflow : str
        Represents a method to convert the body data to TensorFlow format.
    slice_step : str
        Represents a method to step through the data.
    """

    def __getattr__(self, attr):
        """
        for dynamic method resolution on the PoseBody

        Parameters
        ----------
        attr : str
            Name of the attribute or method to get.

        Returns
        -------
        Callable
            callable method if found.

        Raises
        ------
        AttributeError
            If the attribute does not exist.
        """
        if attr not in Pose.pass_through_methods:
            raise AttributeError("Attribute '%s' doesn't exist on class Pose" % attr)

        def func(*args, **kwargs):
            prop = getattr(self.body, attr)
            body_res = prop(*args, **kwargs)

            if isinstance(body_res, PoseBody):
                header = self.header
                if hasattr(header, attr):
                    header_res = getattr(header, attr)(*args, **kwargs)
                    if isinstance(header_res, PoseHeader):
                        header = header_res

                return Pose(header, body_res)

            return body_res

        return func

    def __str__(self):
        return f"Pose\n{self.header}\n{self.body}"

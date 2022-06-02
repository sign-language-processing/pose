from itertools import chain
from typing import List, BinaryIO, Dict, Type, Tuple

import numpy as np
import numpy.ma as ma

from pose_format.numpy import NumPyPoseBody
from pose_format.pose_body import PoseBody
from pose_format.pose_header import PoseHeader, PoseHeaderDimensions, PoseNormalizationInfo, PoseHeaderComponent
from pose_format.utils.fast_math import distance_batch
from pose_format.utils.reader import BufferReader


class Pose:
    """File IO for '.pose' file format, including the header and body"""

    def __init__(self, header: PoseHeader, body: PoseBody):
        """
        :param header: PoseHeader
        :param body: PoseBody
        """
        self.header = header
        self.body = body

    @staticmethod
    def read(buffer: bytes, pose_body: Type[PoseBody] = NumPyPoseBody):
        reader = BufferReader(buffer)
        header = PoseHeader.read(reader)
        body = pose_body.read(header, reader)

        return Pose(header, body)

    def write(self, buffer: BinaryIO):
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

    def normalize(self, info: PoseNormalizationInfo, scale_factor: float = 1) -> "Pose":
        """
        Normalize the points to a fixed distance between two particular points.

        """
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

        :param mu:
        :param std:
        :param axis:
        :return:
        """

        mu = mu if mu is not None else self.body.data.mean(axis=axis)
        std = std if std is not None else self.body.data.std(axis=axis)

        self.body.data = (self.body.data - mu) / std

        return mu, std

    def unnormalize_distribution(self, mu, std):
        self.body.data = (self.body.data * std) + mu

    def frame_dropout_uniform(self,
                              dropout_min: float = 0.2,
                              dropout_max: float = 1.0) -> Tuple["Pose", List[int]]:
        body, selected_indexes = self.body.frame_dropout_uniform(dropout_min=dropout_min, dropout_max=dropout_max)
        return Pose(header=self.header, body=body), selected_indexes

    def frame_dropout_normal(self,
                             dropout_mean: float = 0.5,
                             dropout_std: float = 0.1) -> Tuple["Pose", List[int]]:
        body, selected_indexes = self.body.frame_dropout_normal(dropout_mean=dropout_mean, dropout_std=dropout_std)
        return Pose(header=self.header, body=body), selected_indexes

    def get_components(self, components: List[str], points: Dict[str, List[str]] = None):
        indexes = {}
        new_components = {}

        idx = 0
        for component in self.header.components:
            if component.name in components:
                new_component = PoseHeaderComponent(component.name, component.points,
                                                    component.limbs, component.colors, component.format)
                if points is not None and component.name in points:  # copy and permute points
                    new_component.points = points[component.name] 
                    point_index_mapping = {component.points.index(point): i for i, point in enumerate(new_component.points)}
                    old_indexes_set = set(point_index_mapping.keys())
                    new_component.limbs = [(point_index_mapping[l1], point_index_mapping[l2]) for l1, l2 in component.limbs if l1 in old_indexes_set and l2 in old_indexes_set]

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

    def bbox(self):
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

    def __getattr__(self, attr):
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

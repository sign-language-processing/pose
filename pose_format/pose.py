from typing import List, BinaryIO

import numpy as np
import numpy.ma as ma

from pose_format.numpy import NumPyPoseBody
from pose_format.pose_body import PoseBody
from pose_format.pose_header import PoseHeader, PoseHeaderDimensions, PoseNormalizationInfo
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
    def read(buffer: bytes, pose_body: PoseBody = NumPyPoseBody):
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

    def normalize(self, info: PoseNormalizationInfo, scale_factor: float = 1):
        """
        Normalize the point to a fixed distance between two points
        """
        mask = self.body.data.mask
        transposed = self.body.zero_filled().points_perspective()

        p1s = transposed[info.p1]
        p2s = transposed[info.p2]

        if transposed.shape[1] == 0:
            p1s = p1s[0]
            p2s = p2s[0]
        else:
            p1s = ma.concatenate(p1s)
            p2s = ma.concatenate(p2s)

        # Move all points so center is (0,0)
        center = np.mean((p2s + p1s) / 2, axis=0)
        self.body.data -= center

        mean_distance = np.mean(distance_batch(p1s, p2s))

        scale = scale_factor / mean_distance  # scale all points to dist/scale

        if round(scale, 5) != 1:
            self.body.data = ma.multiply(self.body.data, scale)

        self.body.data = ma.array(self.body.data, mask=mask)

        return self

    def frame_dropout(self, dropout_std=0.1):
        body, selected_indexes = self.body.frame_dropout(dropout_std=dropout_std)
        return Pose(header=self.header, body=body), selected_indexes

    def get_components(self, components: List[str]):
        indexes = []
        new_components = []

        idx = 0
        for component in self.header.components:
            if component.name in components:
                new_components.append(component)
                indexes += list(range(idx, len(component.points) + idx))
            idx += len(component.points)

        new_header = PoseHeader(self.header.version, self.header.dimensions, new_components)
        new_body = self.body.get_points(indexes)

        return Pose(header=new_header, body=new_body)

    def bbox(self):
        body = self.body.bbox(self.header)
        header = self.header.bbox()
        return Pose(header=header, body=body)

    pass_through_methods = {
        "augment2d",  # Augment 2D points
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

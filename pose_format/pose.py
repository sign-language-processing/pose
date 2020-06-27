import inspect
from typing import List, BinaryIO

from .pose_body import PoseBody
from .pose_header import PoseHeader, PoseHeaderDimensions, PoseNormalizationInfo
from .utils.reader import BufferReader

import numpy as np
import numpy.ma as ma

from .utils.fast_math import distance_batch


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
    def read(buffer: bytes, pose_body: PoseBody):
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

        # try:
        mean_distance = np.mean(distance_batch(p1s, p2s))
        # except FloatingPointError:
        #     print(self.body.data)
        #     print(p1s)
        #     print(p2s)

        scale = scale_factor / mean_distance  # scale all points to dist/scale

        if round(scale, 5) != 1:
            self.body.data = ma.multiply(self.body.data, scale)

        self.body.data = ma.array(self.body.data, mask=mask)

    def frame_dropout(self, dropout_std=0.1):
        body, selected_indexes = self.body.frame_dropout(dropout_std=dropout_std)
        return Pose(header=self.header, body=body), selected_indexes

    def augment2d(self, rotation_std=0.2, shear_std=0.2, scale_std=0.2):
        """
        :param rotation_std: Rotation in radians
        :param shear_std: Shear X in percent
        :param scale_std: Scale X in percent
        :return:
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
            scale_matrix[0][0] += np.random.normal(loc=0, scale=scale_std, size=1)[0]
            matrix = np.dot(matrix, scale_matrix)

        body = self.body.matmul(matrix.astype(dtype=np.float32))
        return Pose(header=self.header, body=body)

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
        header = self.header.bbox()
        body = self.body.bbox(header)
        return Pose(header=header, body=body)

    pass_through_methods = {
        "interpolate",  # Interpolate missing pose points
        "torch",  # Convert body to torch
        # "bbox",  # Replace every component with its bounding box
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

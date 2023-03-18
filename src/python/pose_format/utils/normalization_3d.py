import math
from typing import Tuple

import numpy as np
import numpy.ma as ma
from pose_format.pose_header import PoseNormalizationInfo
from scipy.spatial.transform import Rotation


class PoseNormalizer:
    def __init__(self,
                 plane: PoseNormalizationInfo,
                 line: PoseNormalizationInfo,
                 size: float = 1):
        self.size = size
        self.plane = plane
        self.line = line

    def rotate_to_normal(self, pose: np.ndarray, normal: np.ndarray, around: np.ndarray):
        # Let's rotate the points such that the normal is the new Z axis
        # Following https://stackoverflow.com/questions/1023948/rotate-normal-vector-onto-axis-plane
        old_x_axis = np.array([1, 0, 0])

        z_axis = normal
        y_axis = np.cross(old_x_axis, z_axis)
        x_axis = np.cross(z_axis, y_axis)

        axis = np.stack([x_axis, y_axis, z_axis])

        return np.dot(pose - around, axis.T)

    def get_normal(self, pose: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        triangle = pose[[self.plane.p1, self.plane.p2, self.plane.p3]]

        v1 = triangle[1] - triangle[0]
        v2 = triangle[2] - triangle[0]

        normal = np.cross(v1, v2)
        normal /= np.linalg.norm(normal)

        return normal, triangle[0]

    def get_rotation_angle(self, pose: ma.masked_array) -> float:
        p1 = pose[self.line.p1]
        p2 = pose[self.line.p2]
        vec = p2 - p1

        return 90 + math.degrees(math.atan2(vec[1], vec[0]))

    def rotate(self, pose: ma.masked_array, angle: float) -> ma.masked_array:
        r = Rotation.from_euler('z', angle, degrees=True)
        return ma.dot(pose, r.as_matrix())

    def scale(self, pose: ma.masked_array) -> ma.masked_array:
        p1 = pose[self.line.p1]
        p2 = pose[self.line.p2]
        current_size = np.sqrt(np.power(p2 - p1, 2).sum())

        pose *= self.size / current_size
        pose -= pose[self.line.p1]  # move to first point of the line
        return pose

    def normalize_pose(self, pose: ma.masked_array) -> ma.masked_array:
        if ma.all(pose == 0):
            return pose

        # First rotate to normal
        normal, base = self.get_normal(pose)
        pose = self.rotate_to_normal(pose, normal, base)

        # Then rotate on the X-Y plane such that the line is on the Y axis
        angle = self.get_rotation_angle(pose.data)
        pose = self.rotate(pose.data, angle)

        # Scale pose such that the line is of size self.size
        pose = self.scale(pose)

        return pose

    def __call__(self, poses: ma.masked_array) -> ma.masked_array:
        return ma.array([[self.normalize_pose(p) for p in ps] for ps in poses], mask=poses.mask)

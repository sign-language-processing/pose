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

    def rotate_to_normal(self, pose: ma.masked_array, normal: ma.masked_array, around: ma.masked_array):
        # Move pose to origin
        pose = pose - around[:, np.newaxis]

        old_x_axis = np.array([1, 0, 0])

        z_axis = normal
        y_axis = np.cross(old_x_axis, z_axis, axis=-1)
        x_axis = np.cross(z_axis, y_axis, axis=-1)

        axis = np.stack([x_axis, y_axis, z_axis], axis=1)

        rotated = np.einsum('...ij,...kj->...ik', pose, axis)
        return ma.masked_array(rotated, pose.mask)

    def get_normal(self, pose: ma.masked_array) -> Tuple[ma.masked_array, ma.masked_array]:
        triangle = pose[:, [self.plane.p1, self.plane.p2, self.plane.p3]]

        v1 = triangle[:, 1] - triangle[:, 0]
        v2 = triangle[:, 2] - triangle[:, 0]

        normal = np.cross(v1, v2, axisa=-1)
        normal /= np.linalg.norm(normal, axis=-1, keepdims=True)

        normal = ma.masked_array(normal, pose[:, 0].mask)
        return normal, triangle[:, 0]

    def get_rotation_angle(self, pose: ma.masked_array) -> ma.masked_array:
        p1 = pose[:, self.line.p1]
        p2 = pose[:, self.line.p2]
        vec = p2 - p1

        return 90 + np.degrees(np.arctan2(vec[..., 1], vec[..., 0]))

    def rotate(self, pose: ma.masked_array, angle: np.ndarray) -> ma.masked_array:
        r = Rotation.from_euler('z', -angle[..., np.newaxis], degrees=True)  # Clockwise rotation
        rotated = np.einsum('...ij,...kj->...ik', pose, r.as_matrix()).reshape(pose.shape)
        return ma.masked_array(rotated, pose.mask)

    def scale(self, pose: ma.masked_array) -> ma.masked_array:
        p1 = pose[:, self.line.p1]
        p2 = pose[:, self.line.p2]
        current_size = ma.sqrt(ma.power(p2 - p1, 2).sum(axis=-1))
        scale = self.size / current_size
        pose *= scale.reshape(-1, 1, 1)
        pose -= pose[:, [self.line.p1]]  # move to first point of the line
        return pose

    def normalize_pose(self, pose: ma.masked_array) -> ma.masked_array:
        # First rotate to normal
        normal, base = self.get_normal(pose)
        pose = self.rotate_to_normal(pose, normal, base)

        # Then rotate on the X-Y plane such that the line is on the Y axis
        angle = self.get_rotation_angle(pose)
        pose = self.rotate(pose, angle)

        # Scale pose such that the line is of size self.size
        pose = self.scale(pose)

        # Filled with zeros
        pose = ma.array(pose.filled(0), mask=pose.mask)

        return pose

    def __call__(self, poses: ma.masked_array) -> ma.masked_array:
        frames, people, joints, dims = poses.shape
        poses = poses.reshape(-1, joints, dims)
        poses = self.normalize_pose(poses)
        return poses.reshape(frames, people, joints, dims)

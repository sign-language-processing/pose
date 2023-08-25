from typing import Tuple

import numpy as np
import numpy.ma as ma
from scipy.spatial.transform import Rotation

from pose_format.pose_header import PoseNormalizationInfo


class PoseNormalizer:
    """
    Class to normalize pose using normalization information.
    
    :param plane: Plane normalization information
    :type plane: PoseNormalizationInfo
    :param line: Line normalization information
    :type line: PoseNormalizationInfo
    :param size: The desired size after normalization, defaults to 1
    :type size: float
    """

    def __init__(self, plane: PoseNormalizationInfo, line: PoseNormalizationInfo, size: float = 1):

        self.size = size
        self.plane = plane
        self.line = line

    def rotate_to_normal(self, pose: ma.masked_array, normal: ma.masked_array, around: ma.masked_array):
        """
        Rotate pose so that its normal vector aligns with z-axis.

        Parameters
        ----------
        pose : ma.masked_array
            Original pose data
        normal : ma.masked_array
            Normal vector with respect to which the pose will be aligned.
        around : ma.masked_array
            Points to rotate around

        Returns
        -------
        ma.masked_array
            The rotated pose
        
        Raises
        ------
        ValueError:
            if the shapes of pose, normal, and around aren't compatible.

        Examples
        --------
        >>> pose = ma.masked_array([[1, 1], [2, 2], [3, 3]])
        >>> normal = ma.masked_array([0, 0, 1])
        >>> around = ma.masked_array([1, 1])
        >>> rotated_pose = normalizer.rotate_to_normal(pose, normal, around)
        """
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
        """
        Get normal vector based on pose "plane"

        Parameters
        ----------
        pose : ma.masked_array
            Pose data.

        Returns
        -------
        normal : ma.masked_array
            Normal vector for pose.
        base : ma.masked_array
            Base point -> triangle[:,0] used to compute normal
        
        Note
        ----
        Important that plane attributes (p1, p2, p3) are correctly initialized for normal to be correctly computed
        """
        triangle = pose[:, [self.plane.p1, self.plane.p2, self.plane.p3]]

        v1 = triangle[:, 1] - triangle[:, 0]
        v2 = triangle[:, 2] - triangle[:, 0]

        normal = np.cross(v1, v2, axisa=-1)
        normal /= np.linalg.norm(normal, axis=-1, keepdims=True)

        normal = ma.masked_array(normal, pose[:, 0].mask)
        return normal, triangle[:, 0]

    def get_rotation_angle(self, pose: ma.masked_array) -> ma.masked_array:
        """
        Gets rotation angle required to rotate pose such that the line is on the Y axis.

        Parameters
        ----------
        pose : ma.masked_array
            Pose data

        Returns
        -------
        ma.masked_array
            Angles (degrees) needed for each pose in the array
        """
        p1 = pose[:, self.line.p1]
        p2 = pose[:, self.line.p2]
        vec = p2 - p1

        return 90 + np.degrees(np.arctan2(vec[..., 1], vec[..., 0]))

    def rotate(self, pose: ma.masked_array, angle: np.ndarray) -> ma.masked_array:
        """
        Rotate pose in the X-Y plane by a custom angle (np.ndarray).

        Parameters
        ----------
        pose : ma.masked_array
            Original pose data
        angle : np.ndarray
            Angles to rotate poses, in degrees.

        Returns
        -------
        ma.masked_array
            rotated pose
        """
        r = Rotation.from_euler('z', -angle[..., np.newaxis], degrees=True)  # Clockwise rotation
        rotated = np.einsum('...ij,...kj->...ik', pose, r.as_matrix()).reshape(pose.shape)
        return ma.masked_array(rotated, pose.mask)

    def scale(self, pose: ma.masked_array) -> ma.masked_array:
        """
        Scaling of pose

        Parameters
        ----------
        pose : ma.masked_array
            pose to scale

        Returns
        -------
        ma.masked_array
            scaled pose
        """
        p1 = pose[:, self.line.p1]
        p2 = pose[:, self.line.p2]
        current_size = ma.sqrt(ma.power(p2 - p1, 2).sum(axis=-1))
        scale = self.size / current_size
        pose *= scale.reshape(-1, 1, 1)
        pose -= pose[:, [self.line.p1]]  # move to first point of the line
        return pose

    def normalize_pose(self, pose: ma.masked_array) -> ma.masked_array:
        """
        Fully normalizes the pose - rotates to match normals, then rotates in the 
        X-Y plane, and finally scales.

        Parameters
        ----------
        pose : ma.masked_array
            original pose data

        Returns
        -------
        ma.masked_array
            fully normalized pose
        """
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
        """
        Normalization to a batch of poses. 

        TReshapes the input to combine frames and people dimensions, 
        applies pose normalization, then reshapes back to the original structure.

        Parameters
        ----------
        poses : ma.masked_array
            4D masked array with dimensions [frames, people, joints, dims] 
            representing a batch of poses needed to be normalized.

        Returns
        -------
        ma.masked_array
            4D masked array with dimensions [frames, people, joints, dims] 
            containing normalized poses.

        """
        frames, people, joints, dims = poses.shape
        poses = poses.reshape(-1, joints, dims)
        poses = self.normalize_pose(poses)
        return poses.reshape(frames, people, joints, dims)

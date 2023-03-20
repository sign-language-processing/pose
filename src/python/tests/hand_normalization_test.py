import numpy as np
from unittest import TestCase

from numpy import ma
from pose_format.pose import Pose
from pose_format.pose_header import PoseNormalizationInfo

from pose_format.utils.normalization_3d import PoseNormalizer


class Test3DNormalization(TestCase):
    def test_normal(self):
        """
        https://sites.math.washington.edu/~king/coursedir/m445w04/notes/vector/normals-planes.html#:~:text=Thus%20for%20a%20plane%20(or,4%2B4)%20%3D%203.
        Example (Plane Equation Example revisited) Given,
        P = (1, 1, 1), Q = (1, 2, 0), R = (-1, 2, 1).
        The normal vector A is the cross product (Q - P) x (R - P) = (1, 2, 2)
        """
        p1 = (1, 1, 1)
        p2 = (1, 2, 0)
        p3 = (-1, 2, 1)

        gold_normal = (1, 2, 2)

        plane = PoseNormalizationInfo(p1=0, p2=1, p3=2)
        normalizer = PoseNormalizer(plane=plane, line=None)
        tensor = ma.array([p1, p2, p3], dtype=np.float32)
        normal, _ = normalizer.get_normal(tensor)

        gold_vec = ma.array(gold_normal) / np.linalg.norm(gold_normal)
        self.assertEqual(ma.allequal(normal, gold_vec), True)

    def test_hand_normalization(self):
        with open('data/mediapipe.pose', 'rb') as f:
            pose = Pose.read(f.read())
            pose = pose.get_components(["RIGHT_HAND_LANDMARKS"])

        plane = pose.header.normalization_info(
            p1=("RIGHT_HAND_LANDMARKS", "WRIST"),
            p2=("RIGHT_HAND_LANDMARKS", "PINKY_MCP"),
            p3=("RIGHT_HAND_LANDMARKS", "INDEX_FINGER_MCP")
        )
        line = pose.header.normalization_info(
            p1=("RIGHT_HAND_LANDMARKS", "WRIST"),
            p2=("RIGHT_HAND_LANDMARKS", "MIDDLE_FINGER_MCP")
        )
        normalizer = PoseNormalizer(plane=plane, line=line, size=100)
        tensor = normalizer(pose.body.data)

        pose.body.data = tensor
        pose.focus()

        with open('data/mediapipe_hand_normalized.pose', 'rb') as f:
            pose_gold = Pose.read(f.read())

        self.assertTrue(ma.allclose(pose.body.data, pose_gold.body.data))

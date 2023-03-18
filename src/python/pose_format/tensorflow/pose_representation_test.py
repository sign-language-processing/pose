from unittest import TestCase

import tensorflow as tf

from pose_format.pose_header import PoseHeaderDimensions, PoseHeader
from pose_format.tensorflow.pose_representation import TensorflowPoseRepresentation
from pose_format.tensorflow.representation.angle import AngleRepresentation
from pose_format.tensorflow.representation.distance import DistanceRepresentation
from pose_format.tensorflow.representation.inner_angle import InnerAngleRepresentation
from pose_format.tensorflow.representation.point_line_distance import PointLineDistanceRepresentation
from pose_format.utils.openpose import OpenPose_Hand_Component

dimensions = PoseHeaderDimensions(width=0, height=0, depth=0)
components = [OpenPose_Hand_Component("hand_left_keypoints_2d")]
header: PoseHeader = PoseHeader(version=0.1, dimensions=dimensions, components=components)

representation = TensorflowPoseRepresentation(
    header=header,
    rep_modules1=[],
    rep_modules2=[AngleRepresentation(), DistanceRepresentation()],
    rep_modules3=[InnerAngleRepresentation(), PointLineDistanceRepresentation()]
)


class TestTensorflowPoseRepresentation(TestCase):
    def test_input_size(self):
        input_size = representation.input_size
        self.assertEqual(input_size, 21)

    def test_calc_output_size(self):
        output_size = representation.calc_output_size()
        self.assertEqual(output_size, 70)

    def test_call_return_shape(self):
        input_size = representation.input_size
        output_size = representation.calc_output_size()

        points = tf.random.normal(shape=(1, 2, input_size, 3), dtype=tf.float32)
        rep = representation(points)
        self.assertEqual(rep.shape, (1, 2, output_size))

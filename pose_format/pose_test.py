import string
import random

import numpy as np
import numpy.ma as ma
import tensorflow as tf

from unittest import TestCase
from typing import Optional, Tuple

from pose_format.pose import Pose
from pose_format.pose_header import PoseHeader, PoseHeaderDimensions, PoseHeaderComponent
from pose_format.numpy.pose_body import NumPyPoseBody
from pose_format.tensorflow.masked.tensor import MaskedTensor
from pose_format.tensorflow.pose_body import TensorflowPoseBody


def _create_pose_header_component(name: str,
                                  num_keypoints: int) -> PoseHeaderComponent:
    """

    :return:
    """

    points = string.ascii_letters[:num_keypoints]
    points = [name + "_" + point for point in points]

    limbs = []
    colors = []

    for _ in points:
        limb = (random.randint(0, num_keypoints), random.randint(0, num_keypoints))
        limbs.append(limb)

        color = (random.randint(0, 256),
                 random.randint(0, 256),
                 random.randint(0, 256))

        colors.append(color)

    component = PoseHeaderComponent(name=name,
                                    points=points,
                                    limbs=limbs,
                                    colors=colors,
                                    point_format="XYC")

    return component


def _create_pose_header(width: int,
                        height: int,
                        depth: int,
                        num_components: int,
                        num_keypoints: int) -> PoseHeader:
    """

    :param width:
    :param height:
    :param depth:
    :param num_components:
    :return:
    """
    dimensions = PoseHeaderDimensions(width=width, height=height, depth=depth)

    components = [_create_pose_header_component(name=str(index), num_keypoints=num_keypoints) for index in range(num_components)]

    header = PoseHeader(version=1.0, dimensions=dimensions, components=components)

    return header


def _create_random_tensorflow_data(frames_min: Optional[int] = None,
                                   frames_max: Optional[int] = None,
                                   num_frames: Optional[int] = None,
                                   num_keypoints: int = 137,
                                   num_dimensions: int = 2) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """

    :param frames_min:
    :param frames_max:
    :param num_frames:
    :param num_keypoints:
    :param num_dimensions:
    :return:
    """
    if num_frames is None:
        assert None not in [frames_min, frames_max]
        num_frames = np.random.randint(frames_min, frames_max + 1)
    else:
        assert frames_min is None and frames_max is None

    # (Frames, People, Points, Dims) - eg (93, 1, 137, 2)
    tensor = tf.random.normal((num_frames, 1, num_keypoints, num_dimensions))

    # (Frames, People, Points) - eg (93, 1, 137)
    confidence = tf.random.uniform((num_frames, 1, num_keypoints), minval=0.0, maxval=1.0, dtype=tf.dtypes.float32)

    # (Frames, People, Points, Dims) - eg (93, 1, 137, 2)
    mask = tf.random.uniform((num_frames, 1, num_keypoints, num_dimensions), minval=0, maxval=2, dtype=tf.int32)
    mask = tf.cast(mask, tf.bool)

    return tensor, mask, confidence


def _create_random_numpy_data(frames_min: Optional[int] = None,
                              frames_max: Optional[int] = None,
                              num_frames: Optional[int] = None,
                              num_keypoints: int = 137,
                              num_dimensions: int = 2,
                              probability_for_masked: float = 0.2) -> Tuple[np.array, np.array, np.array]:
    """

    :param frames_min:
    :param frames_max:
    :param num_frames:
    :param num_keypoints:
    :param num_dimensions:
    :param probability_for_masked:
    :return:
    """
    if num_frames is None:
        assert None not in [frames_min, frames_max]
        num_frames = np.random.randint(frames_min, frames_max + 1)
    else:
        assert frames_min is None and frames_max is None

    # (Frames, People, Points, Dims) - eg (93, 1, 137, 2)
    tensor = np.random.random_sample(size=(num_frames, 1, num_keypoints, num_dimensions))

    # (Frames, People, Points) - eg (93, 1, 137)
    confidence = np.random.uniform(size=(num_frames, 1, num_keypoints), low=0.0, high=1.0)

    # (Frames, People, Points, Dims) - eg (93, 1, 137, 2)
    mask = np.random.choice(a=[False, True],
                            size=(num_frames, 1, num_keypoints, num_dimensions),
                            p=[probability_for_masked, 1 - probability_for_masked])

    return tensor, mask, confidence


def _get_random_pose_object_with_tf_posebody(num_keypoints: int) -> Pose:

    tensor, mask, confidence = _create_random_tensorflow_data(frames_min=1,
                                                              frames_max=10,
                                                              num_keypoints=num_keypoints)

    masked_tensor = MaskedTensor(tensor=tensor, mask=mask)
    body = TensorflowPoseBody(fps=10, data=masked_tensor, confidence=confidence)

    header = _create_pose_header(width=10, height=7, depth=0, num_components=3, num_keypoints=num_keypoints)

    return Pose(header=header, body=body)


def _get_random_pose_object_with_numpy_posebody(num_keypoints: int) -> Pose:

    tensor, mask, confidence = _create_random_numpy_data(frames_min=1,
                                                         frames_max=10,
                                                         num_keypoints=num_keypoints)

    masked_array = ma.array(tensor, mask=np.logical_not(mask))

    body = NumPyPoseBody(fps=10, data=masked_array, confidence=confidence)

    header = _create_pose_header(width=10, height=7, depth=0, num_components=3, num_keypoints=num_keypoints)

    return Pose(header=header, body=body)


class TestPose(TestCase):
    def test_pose_object_should_be_callable(self):
        assert callable(Pose)

    def test_pose_tf_posebody_normalize_eager_mode(self):

        pose = _get_random_pose_object_with_tf_posebody(num_keypoints=5)

        # in the mock data header components are named 0, 1, and so on
        # individual points are named 0_a, 0_b, and so on
        pose.normalize(pose.header.normalization_info(
            p1=("0", "0_a"),
            p2=("0", "0_b")
        ))

    def test_pose_numpy_posebody_normalize(self):

        pose = _get_random_pose_object_with_numpy_posebody(num_keypoints=5)

        # in the mock data header components are named 0, 1, and so on
        # individual points are named 0_a, 0_b, and so on
        pose.normalize(pose.header.normalization_info(
            p1=("0", "0_a"),
            p2=("0", "0_b")
        ))

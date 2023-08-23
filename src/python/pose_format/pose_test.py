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
    Create a PoseHeaderComponent with randomized limbs and colors.

    Parameters
    ----------
    name : str
        Name of the component.
    num_keypoints : int
        Number of keypoints for the component.

    Returns
    -------
    PoseHeaderComponent
        Component with specified name, randomized limbs and colors.
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
    """Creats a PoseHeader with given dimensions and components.

    Parameters
    ----------
    width : int
        Width dimension.
    height : int
        Height dimension.
    depth : int
        Depth dimension.
    num_components : int
        Number of components.
    num_keypoints : int
        Number of keypoints for each component.

    Returns
    -------
    PoseHeader
        Header with specified dimensions and components.
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
    Creates random TensorFlow data for poses with optional frame ranges and default settings

    Parameters
    ----------
    frames_min : int, optional
        Minimum number of frames.
    frames_max : int, optional
        Maximum number of frames.
    num_frames : int, optional
        Specific number of frames.
    num_keypoints : int, default=137
        Number of keypoints.
    num_dimensions : int, default=2
        Number of dimensions (e.g., 2 for x, y coordinates).

    Returns
    -------
    tuple of tf.Tensor
        - tensor: random poses tensor, shape (num_frames, 1, num_keypoints, num_dimensions).
        - mask: bool mask tensor with the same shape as tensor.
        - confidence: confidence scores tensor, shape (num_frames, 1, num_keypoints).
    """
    if num_frames is None:
        assert None not in [frames_min, frames_max]
        num_frames = np.random.randint(frames_min, frames_max + 1)
    else:
        assert frames_min is None and frames_max is None

    # avoid a mean of zero to test certain pose methods
    # (Frames, People, Points, Dims) - eg (93, 1, 137, 2)
    tensor = tf.random.normal(shape=(num_frames, 1, num_keypoints, num_dimensions), mean=1.0)

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
    Create random NumPy data for poses with optional frame ranges and default settings.

    Parameters
    ----------
    frames_min : int, optional
        Minimum number of frames.
    frames_max : int, optional
        Maximum number of frames.
    num_frames : int, optional
        Specific number of frames.
    num_keypoints : int, default=137
        Number of keypoints.
    num_dimensions : int, default=2
        Number of dimensions (e.g., 2 for x, y coordinates).
    probability_for_masked : float, default=0.2
        Probability for point to be masked.

    Returns
    -------
    tuple of np.array
        - tensor: Random poses numpy array, shape (num_frames, 1, num_keypoints, num_dimensions).
        - mask: Boolean mask numpy array and same shape as tensor.
        - confidence: Confidence scores numpy array, shape (num_frames, 1, num_keypoints).
    """
    if num_frames is None:
        assert None not in [frames_min, frames_max]
        num_frames = np.random.randint(frames_min, frames_max + 1)
    else:
        assert frames_min is None and frames_max is None

    # avoid a mean of zero to test certain pose methods
    # (Frames, People, Points, Dims) - eg (93, 1, 137, 2)
    tensor = np.random.normal(loc=1.0, size=(num_frames, 1, num_keypoints, num_dimensions))

    # (Frames, People, Points) - eg (93, 1, 137)
    confidence = np.random.uniform(size=(num_frames, 1, num_keypoints), low=0.0, high=1.0)

    # (Frames, People, Points, Dims) - eg (93, 1, 137, 2)
    mask = np.random.choice(a=[False, True],
                            size=(num_frames, 1, num_keypoints, num_dimensions),
                            p=[probability_for_masked, 1 - probability_for_masked])

    return tensor, mask, confidence


def _get_random_pose_object_with_tf_posebody(num_keypoints: int,
                                             frames_min: int = 1,
                                             frames_max: int = 10) -> Pose:

    tensor, mask, confidence = _create_random_tensorflow_data(frames_min=frames_min,
                                                              frames_max=frames_max,
                                                              num_keypoints=num_keypoints)

    masked_tensor = MaskedTensor(tensor=tensor, mask=mask)
    body = TensorflowPoseBody(fps=10, data=masked_tensor, confidence=confidence)

    header = _create_pose_header(width=10, height=7, depth=0, num_components=3, num_keypoints=num_keypoints)

    return Pose(header=header, body=body)


def _get_random_pose_object_with_numpy_posebody(num_keypoints: int,
                                                frames_min: int = 1,
                                                frames_max: int = 10) -> Pose:

    tensor, mask, confidence = _create_random_numpy_data(frames_min=frames_min,
                                                         frames_max=frames_max,
                                                         num_keypoints=num_keypoints)

    masked_array = ma.array(tensor, mask=np.logical_not(mask))

    body = NumPyPoseBody(fps=10, data=masked_array, confidence=confidence)

    header = _create_pose_header(width=10, height=7, depth=0, num_components=3, num_keypoints=num_keypoints)

    return Pose(header=header, body=body)


class TestPose(TestCase):
    """
    A suite of tests for the Pose object.
    """
    def test_pose_object_should_be_callable(self):
        """
        Tests if the Pose object is callable.
        """
        assert callable(Pose)


class TestPoseTensorflowPoseBody(TestCase):
    """
    Tests for Pose objects containing TensorFlow PoseBody data.
    """

    def test_pose_tf_posebody_normalize_eager_mode_preserves_shape(self):
        """
        Tests if the normalization of Pose object with TensorFlow PoseBody in eager mode preserves tensor shape.
        """
        pose = _get_random_pose_object_with_tf_posebody(num_keypoints=5)

        shape_before = pose.body.data.tensor.shape

        # in the mock data header components are named 0, 1, and so on
        # individual points are named 0_a, 0_b, and so on
        pose.normalize(pose.header.normalization_info(
            p1=("0", "0_a"),
            p2=("0", "0_b")
        ))

        shape_after = pose.body.data.tensor.shape

        self.assertEqual(shape_before, shape_after, "Normalize did not preserve tensor shape.")

    def test_pose_tf_posebody_normalize_distribution_eager_mode_correct_result(self):
        """
        Tests if the normalization of distribution for Pose object with TensorFlow PoseBody returns expected result.
        """
        pose = _get_random_pose_object_with_tf_posebody(num_keypoints=5, frames_min=2)

        tensor_as_numpy = pose.body.data.tensor.numpy()
        mask_as_numpy = pose.body.data.mask.numpy()
        masked_numpy_array = ma.array(tensor_as_numpy, mask=np.logical_not(mask_as_numpy))

        axis = (0, 1)

        numpy_mean = ma.mean(masked_numpy_array, axis=axis)
        numpy_std = ma.std(masked_numpy_array, axis=axis)
        expected_tensor = (masked_numpy_array - numpy_mean) / numpy_std
        expected_tensor = expected_tensor.filled(0)

        pose.normalize_distribution(axis=axis)

        actual_tensor_nan_fixed = pose.body.data.fix_nan()
        actual_tensor_zero_filled = actual_tensor_nan_fixed.zero_filled()
        actual_tensor_as_numpy = actual_tensor_zero_filled.numpy()

        self.assertTrue(np.allclose(actual_tensor_as_numpy, expected_tensor),
                        "Normalize distribution did not return the expected result.")

    def test_pose_tf_posebody_frame_dropout_normal_eager_mode_num_frames_not_zero(self):
        """
        Tests if frame dropout using normal distribution in Pose object with TensorFlow PoseBody preserves frame count.
        """
        pose = _get_random_pose_object_with_tf_posebody(num_keypoints=5, frames_min=3)

        pose_after_dropout, _ = pose.frame_dropout_normal()

        num_frames = pose_after_dropout.body.data.tensor.shape[0]

        self.assertNotEqual(num_frames, 0, "Number of frames after dropout can never be 0.")

    def test_pose_tf_posebody_frame_dropout_uniform_eager_mode_num_frames_not_zero(self):
        """
        Checks if frame dropout using uniform distribution in Pose object with TensorFlow PoseBody preserves frame count.
        """
        pose = _get_random_pose_object_with_tf_posebody(num_keypoints=5, frames_min=3)

        pose_after_dropout, _ = pose.frame_dropout_uniform()

        num_frames = pose_after_dropout.body.data.tensor.shape[0]

        self.assertNotEqual(num_frames, 0, "Number of frames after dropout can never be 0.")

    # test if pose object methods can be used as tf.data.Dataset operations

    def test_pose_normalize_can_be_used_in_tf_dataset_map(self):
        """
        Tests if the normalize method of Pose object can be used within a tf.data.Dataset map operation.
        """
        num_examples = 10

        dataset = tf.data.Dataset.range(num_examples)

        def create_and_normalize_pose(example: tf.Tensor) -> tf.Tensor:

            pose = _get_random_pose_object_with_tf_posebody(num_keypoints=5)

            _ = pose.normalize(pose.header.normalization_info(
                p1=("0", "0_a"),
                p2=("0", "0_b")
            ))

            return example

        dataset.map(create_and_normalize_pose)

    def test_pose_normalize_distribution_can_be_used_in_tf_dataset_map(self):
        """
        Tests if the normalize_distribution method of Pose object can be used within a tf.data.Dataset map operation.
        """
        num_examples = 10

        dataset = tf.data.Dataset.range(num_examples)

        def create_pose_and_normalize_distribution(example: tf.Tensor) -> tf.Tensor:

            pose = _get_random_pose_object_with_tf_posebody(num_keypoints=5)

            _ = pose.normalize_distribution(axis=(0, 1))

            return example

        dataset.map(create_pose_and_normalize_distribution)

    def test_pose_frame_dropout_normal_can_be_used_in_tf_dataset_map(self):

        """
        Tests if the frame_dropout_normal method of Pose object can be used within a tf.data.Dataset map operation.
        """
        num_examples = 10

        dataset = tf.data.Dataset.range(num_examples)

        def create_pose_and_frame_dropout_normal(example: tf.Tensor) -> tf.Tensor:

            pose = _get_random_pose_object_with_tf_posebody(num_keypoints=5)

            _, _ = pose.frame_dropout_normal()

            return example

        dataset.map(create_pose_and_frame_dropout_normal)

    def test_pose_frame_dropout_uniform_can_be_used_in_tf_dataset_map(self):
        """
        Tests if the frame_dropout_uniform method of Pose object can be used within a tf.data.Dataset map operation.
        """
        num_examples = 10

        dataset = tf.data.Dataset.range(num_examples)

        def create_pose_and_frame_dropout_uniform(example: tf.Tensor) -> tf.Tensor:

            pose = _get_random_pose_object_with_tf_posebody(num_keypoints=5)

            _, _ = pose.frame_dropout_uniform()

            return example

        dataset.map(create_pose_and_frame_dropout_uniform)


class TestPoseNumpyPoseBody(TestCase):
    """
    Testcases for Pose objects containing NumPy PoseBody data.
    """

    def test_pose_numpy_posebody_normalize_preserves_shape(self):
        """
        Tests if the normalization of Pose object with NumPy PoseBody preserves array shape.
        """
        pose = _get_random_pose_object_with_numpy_posebody(num_keypoints=5, frames_min=3)

        shape_before = pose.body.data.shape

        # in the mock data header components are named 0, 1, and so on
        # individual points are named 0_a, 0_b, and so on
        pose.normalize(pose.header.normalization_info(
            p1=("0", "0_a"),
            p2=("0", "0_b")
        ))

        shape_after = pose.body.data.shape

        self.assertEqual(shape_before, shape_after, "Normalize did not preserve tensor shape.")

    def test_pose_numpy_posebody_frame_dropout_normal_eager_mode_num_frames_not_zero(self):
        """
        Tests if frame dropout using normal distribution in Pose object with NumPy PoseBody preserves frame count.
        """
        pose = _get_random_pose_object_with_numpy_posebody(num_keypoints=5, frames_min=3)

        pose_after_dropout, _ = pose.frame_dropout_normal()

        num_frames = pose_after_dropout.body.data.shape[0]

        self.assertNotEqual(num_frames, 0, "Number of frames after dropout can never be 0.")

    def test_pose_numpy_posebody_frame_dropout_uniform_eager_mode_num_frames_not_zero(self):
        """
        Tests if frame dropout using uniform distribution in Pose object with NumPy PoseBody preserves frame count.
        """
        pose = _get_random_pose_object_with_numpy_posebody(num_keypoints=5, frames_min=3)

        pose_after_dropout, _ = pose.frame_dropout_uniform()

        num_frames = pose_after_dropout.body.data.shape[0]

        self.assertNotEqual(num_frames, 0, "Number of frames after dropout can never be 0.")

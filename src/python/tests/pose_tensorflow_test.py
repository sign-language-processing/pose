from typing import Optional, Tuple
from unittest import TestCase

import numpy as np
import numpy.ma as ma
import tensorflow as tf
from pose_format import Pose

from pose_format.tensorflow.masked.tensor import MaskedTensor as TensorflowMaskedTensor
from pose_format.tensorflow.pose_body import TensorflowPoseBody

from .pose_test import _create_pose_header


def _create_random_tensorflow_data(frames_min: Optional[int] = None,
                                   frames_max: Optional[int] = None,
                                   num_frames: Optional[int] = None,
                                   num_keypoints: int = 137,
                                   num_dimensions: int = 2) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """
    Creates random TensorFlow data for testing.

    Parameters
    ----------
    frames_min : Optional[int], default=None
        Minimum number of frames for random generation if `num_frames` is not specified.
    frames_max : Optional[int], default=None
        Maximum number of frames for random generation if `num_frames` is not specified.
    num_frames : Optional[int], default=None
        Specific number of frames.
    num_keypoints : int, default=137
        Number of keypoints in the pose data.
    num_dimensions : int, default=2
        Number of dimensions in the pose data.

    Returns
    -------
    Tuple[tf.Tensor, tf.Tensor, tf.Tensor]
        Random tensor data, mask, and confidence values.
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


def _get_random_pose_object_with_tf_posebody(num_keypoints: int, frames_min: int = 1, frames_max: int = 10) -> Pose:
    """
    Generates a random Pose object with TensorFlow pose body for testing.

    Parameters
    ----------
    num_keypoints : int
        Number of keypoints in the pose data.
    frames_min : int, default=1
        Minimum number of frames for random generation.
    frames_max : int, default=10
        Maximum number of frames for random generation.

    Returns
    -------
    Pose
        Randomly generated Pose object.
    """

    tensor, mask, confidence = _create_random_tensorflow_data(frames_min=frames_min,
                                                              frames_max=frames_max,
                                                              num_keypoints=num_keypoints)

    masked_tensor = TensorflowMaskedTensor(tensor=tensor, mask=mask)
    body = TensorflowPoseBody(fps=10, data=masked_tensor, confidence=confidence)

    header = _create_pose_header(width=10, height=7, depth=0, num_components=3, num_keypoints=num_keypoints)

    return Pose(header=header, body=body)


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
        pose.normalize(pose.header.normalization_info(p1=("0", "0_a"), p2=("0", "0_b")))

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

            _ = pose.normalize(pose.header.normalization_info(p1=("0", "0_a"), p2=("0", "0_b")))

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

    def test_pose_tf_posebody_copy_creates_deepcopy(self):
        pose = _get_random_pose_object_with_tf_posebody(num_keypoints=5)
        self.assertIsInstance(pose.body, TensorflowPoseBody)
        self.assertIsInstance(pose.body.data, TensorflowMaskedTensor)

        pose_copy = pose.copy()
        self.assertIsInstance(pose_copy.body, TensorflowPoseBody)
        self.assertIsInstance(pose_copy.body.data, TensorflowMaskedTensor)

        # Check that pose and pose_copy are not the same object
        self.assertNotEqual(pose, pose_copy, "Copy of pose should not be 'equal' to original")

        # Ensure the data tensors are equal but independent
        self.assertTrue(tf.reduce_all(pose.body.data == pose_copy.body.data), "Copy's data should match original")

        # Modify original pose's data and check that copy remains unchanged
        pose.body.data = TensorflowMaskedTensor(tf.zeros_like(pose.body.data.tensor),
                                                tf.zeros_like(pose.body.data.mask))

        self.assertFalse(tf.reduce_all(pose.body.data == pose_copy.body.data),
                         "Copy's data should not match original after original is replaced")

        # Create another copy and ensure it matches the first copy
        pose = pose_copy.copy()
        self.assertNotEqual(pose, pose_copy, "Copy of pose should not be 'equal' to original")

        self.assertTrue(tf.reduce_all(pose.body.data == pose_copy.body.data), "Copy's data should match original again")

        # Modify the copy and check that the original remains unchanged
        pose_copy.body.data.tensor = tf.zeros(pose_copy.body.data.tensor.shape)

        self.assertFalse(tf.reduce_all(pose.body.data == pose_copy.body.data),
                         "Copy's data should not match original after copy is modified")

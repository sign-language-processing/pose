import random
import string
from typing import Optional, Tuple
from unittest import TestCase

import numpy as np
import numpy.ma as ma
import tensorflow as tf
import torch

from pose_format.numpy.pose_body import NumPyPoseBody
from pose_format.pose import Pose
from pose_format.pose_header import (PoseHeader, PoseHeaderComponent,
                                     PoseHeaderDimensions)
from pose_format.tensorflow.masked.tensor import MaskedTensor as TensorflowMaskedTensor
from pose_format.torch.masked import MaskedTensor as TorchMaskedTensor
from pose_format.tensorflow.pose_body import TensorflowPoseBody
from pose_format.torch.pose_body import TorchPoseBody



def _create_pose_header_component(name: str, num_keypoints: int) -> PoseHeaderComponent:
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

        color = (random.randint(0, 256), random.randint(0, 256), random.randint(0, 256))

        colors.append(color)

    component = PoseHeaderComponent(name=name, points=points, limbs=limbs, colors=colors, point_format="XYC")

    return component


def _distribute_points_among_components(component_count: int, total_keypoint_count: int):
    if component_count <= 0 or total_keypoint_count < component_count + 1:
        raise ValueError("Total keypoints must be at least component count+1 (so that 0 can have two), and component count must be positive")

    # Step 1: Initialize with required minimum values
    keypoint_counts = [2] + [1] * (component_count - 1)  # Ensure first is 2, others at least 1

    # Step 2: Distribute remaining points
    remaining_points = total_keypoint_count - sum(keypoint_counts)
    for _ in range(remaining_points):
        keypoint_counts[random.randint(0, component_count - 1)] += 1  # Add randomly

    return keypoint_counts

def _create_pose_header(width: int, height: int, depth: int, num_components: int, num_keypoints: int) -> PoseHeader:
    """
    Create a PoseHeader with given dimensions and components.

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

    keypoints_per_component = _distribute_points_among_components(num_components, num_keypoints)

    components = [
        _create_pose_header_component(name=str(index), num_keypoints=keypoints_per_component[index]) for index in range(num_components)
    ]

    header = PoseHeader(version=1.0, dimensions=dimensions, components=components)

    return header


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




def _create_random_numpy_data(frames_min: Optional[int] = None,
                              frames_max: Optional[int] = None,
                              num_frames: Optional[int] = None,
                              num_keypoints: int = 137,
                              num_dimensions: int = 2,
                              probability_for_masked: float = 0.2) -> Tuple[np.array, np.array, np.array]:
    """
    Creates random Numpy data for testing.

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
    probability_for_masked : float, default=0.2
        Probability for generating masked values.

    Returns
    -------
    Tuple[np.array, np.array, np.array]
        Random numpy array data, mask, and confidence values.
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

def _create_random_torch_data(frames_min: Optional[int] = None,
                              frames_max: Optional[int] = None,
                              num_frames: Optional[int] = None,
                              num_keypoints: int = 137,
                              num_dimensions: int = 2) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Creates random PyTorch data for testing.

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
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        Random tensor data, mask, and confidence values.
    """
    if num_frames is None:
        assert None not in [frames_min, frames_max]
        num_frames = np.random.randint(frames_min, frames_max + 1)
    else:
        assert frames_min is None and frames_max is None

    # Avoid a mean of zero to test certain pose methods
    tensor = torch.randn((num_frames, 1, num_keypoints, num_dimensions)) + 1.0  # Shape: (Frames, People, Points, Dims)

    confidence = torch.rand((num_frames, 1, num_keypoints), dtype=torch.float32)  # Shape: (Frames, People, Points)

    mask = torch.randint(0, 2, (num_frames, 1, num_keypoints, num_dimensions), dtype=torch.bool)  # Bool mask

    return tensor, mask, confidence


def _get_random_pose_object_with_torch_posebody(num_keypoints: int, frames_min: int = 1, frames_max: int = 10) -> Pose:
    """
    Generates a random Pose object with PyTorch pose body for testing.

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

    tensor, mask, confidence = _create_random_torch_data(frames_min=frames_min,
                                                         frames_max=frames_max,
                                                         num_keypoints=num_keypoints)

    masked_tensor = TorchMaskedTensor(tensor=tensor, mask=mask)
    body = TorchPoseBody(fps=10, data=masked_tensor, confidence=confidence)

    header = _create_pose_header(width=10, height=7, depth=0, num_components=3, num_keypoints=num_keypoints)

    return Pose(header=header, body=body)

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


def _get_random_pose_object_with_numpy_posebody(num_keypoints: int, frames_min: int = 1, frames_max: int = 10, num_components=3) -> Pose:
    """
    Creates a random Pose object with Numpy pose body for testing.

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

    tensor, mask, confidence = _create_random_numpy_data(frames_min=frames_min,
                                                         frames_max=frames_max,
                                                         num_keypoints=num_keypoints)

    masked_array = ma.array(tensor, mask=np.logical_not(mask))

    body = NumPyPoseBody(fps=10, data=masked_array, confidence=confidence)

    header = _create_pose_header(width=10, height=7, depth=0, num_components=num_components, num_keypoints=num_keypoints)

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

    def test_get_index(self):
        pose = _get_random_pose_object_with_numpy_posebody(num_keypoints=5)
        expected_index = 0
        self.assertEqual(0, pose.header.get_point_index("0", "0_a"))
        for component in pose.header.components:
            for point in component.points:
                self.assertEqual(expected_index, pose.header.get_point_index(component.name, point))
                expected_index +=1

        with self.assertRaises(ValueError):
            pose.header.get_point_index("component that doesn't exist", "")

        with self.assertRaises(ValueError):
            pose.header.get_point_index("0", "point that doesn't exist")

        

    def test_pose_remove_components(self):
        pose = _get_random_pose_object_with_numpy_posebody(num_keypoints=5)
        assert pose.body.data.shape[-2] == 5
        assert pose.body.data.shape[-1] == 2 # XY dimensions

        self.assertEqual(len(pose.header.components), 3)
        self.assertEqual(sum(len(c.points) for c in pose.header.components), 5)
        self.assertEqual(pose.header.components[0].name, "0")
        self.assertEqual(pose.header.components[1].name, "1")
        self.assertEqual(pose.header.components[0].points[0], "0_a")
        self.assertIn("1_a", pose.header.components[1].points)
        self.assertNotIn("1_f", pose.header.components[1].points)
        self.assertNotIn("4", pose.header.components)

        # test that we can remove a component
        component_to_remove = "0"
        pose_copy = pose.copy()
        self.assertIn(component_to_remove, [c.name for c in pose_copy.header.components])
        pose_copy = pose_copy.remove_components(component_to_remove)
        self.assertNotIn(component_to_remove, [c.name for c in pose_copy.header.components])
        self.assertEqual(pose_copy.header.components[0].name, "1")
        # quickly check to make sure other components/points weren't removed
        self.assertIn("1_a", pose_copy.header.components[0].points)
        self.assertEqual(pose_copy.header.components[0].points, pose.header.components[1].points)        


        # Remove a point only
        point_to_remove = "0_a"
        pose_copy = pose.copy()
        self.assertIn(point_to_remove, pose_copy.header.components[0].points)
        pose_copy = pose_copy.remove_components([], {point_to_remove[0]:[point_to_remove]})
        self.assertNotIn(point_to_remove, pose_copy.header.components[0].points)
        # quickly check to make sure other components/points weren't removed
        self.assertIn("1_a", pose_copy.header.components[1].points)
        self.assertEqual(pose_copy.header.components[1].points, pose.header.components[1].points)        


        # Can we remove two things at once
        pose_copy = pose.copy()
        component_to_remove = "1"
        point_to_remove = "2_a"
        component_to_remove_point_from = "2"
        
        self.assertIn(component_to_remove, [c.name for c in pose_copy.header.components])
        self.assertIn(component_to_remove_point_from, [c.name for c in pose_copy.header.components])
        self.assertIn(point_to_remove, pose_copy.header.components[2].points)
        pose_copy = pose_copy.remove_components([component_to_remove], {component_to_remove_point_from:[point_to_remove]})
        self.assertNotIn(component_to_remove, [c.name for c in pose_copy.header.components])
        self.assertIn(component_to_remove_point_from, [c.name for c in pose_copy.header.components]) # this should still be around
        self.assertIn("0_a", pose_copy.header.components[0].points)
        self.assertEqual(pose_copy.header.components[0].points, pose.header.components[0].points) # should be unaffected

        # can we remove a component and a point FROM that component without crashing
        component_to_remove = "0"
        point_to_remove = "0_a"
        pose_copy = pose.copy()
        self.assertIn(point_to_remove, pose_copy.header.components[0].points)
        pose_copy = pose_copy.remove_components([component_to_remove], {component_to_remove:[point_to_remove]})
        self.assertNotIn(component_to_remove, [c.name for c in pose_copy.header.components])
        self.assertNotIn(point_to_remove, pose_copy.header.components[0].points)
        self.assertEqual(pose_copy.header.components[0].points, pose.header.components[1].points) # should be unaffected


        # can we "remove" a component that doesn't exist without crashing
        component_to_remove = "NOT EXISTING"
        pose_copy = pose.copy()
        initial_count = len(pose_copy.header.components)
        pose_copy = pose_copy.remove_components([component_to_remove])
        self.assertEqual(initial_count, len(pose_copy.header.components))
        for c_orig, c_copy in zip(pose.header.components, pose_copy.header.components):
            self.assertNotEqual(c_copy, c_orig) # should be a new object...
            self.assertEqual(c_copy.name, c_orig.name) # with the same name
            self.assertEqual(c_copy.points, c_orig.points) # and the same points


        

        # can we "remove" a point that doesn't exist from a component that does without crashing
        point_to_remove = "2_x"
        component_to_remove_point_from = "2"
        pose_copy = pose.copy()
        self.assertNotIn(point_to_remove, pose_copy.header.components[2].points)
        pose_copy = pose_copy.remove_components([], {component_to_remove_point_from:[point_to_remove]})
        self.assertNotIn(point_to_remove, pose_copy.header.components[2].points)
        for c_orig, c_copy in zip(pose.header.components, pose_copy.header.components):
            self.assertNotEqual(c_copy, c_orig) # should be a new object...
            self.assertEqual(c_copy.name, c_orig.name) # with the same name
            self.assertEqual(c_copy.points, c_orig.points) # and the same points


        # can we "remove" an empty list of points
        component_to_remove_point_from = "2"
        pose_copy = pose.copy()
        initial_component_count = len(pose_copy.header.components)
        initial_point_count = len(pose_copy.header.components[2].points)
        pose_copy = pose_copy.remove_components([], {component_to_remove_point_from:[]})
        self.assertEqual(initial_component_count, len(pose_copy.header.components))
        self.assertEqual(len(pose_copy.header.components[2].points), initial_point_count)
        for c_orig, c_copy in zip(pose.header.components, pose_copy.header.components):
            self.assertNotEqual(c_copy, c_orig) # should be a new object...
            self.assertEqual(c_copy.name, c_orig.name) # with the same name
            self.assertEqual(c_copy.points, c_orig.points) # and the same points

        # can we remove a point from a component that doesn't exist
        point_to_remove = "2_x"
        component_to_remove_point_from = "NOT EXISTING"
        pose_copy = pose.copy()
        self.assertNotIn(point_to_remove, pose_copy.header.components[2].points)
        pose_copy = pose_copy.remove_components([], {component_to_remove_point_from:[point_to_remove]})
        self.assertNotIn(point_to_remove, pose_copy.header.components[2].points)
        for c_orig, c_copy in zip(pose.header.components, pose_copy.header.components):
            self.assertNotEqual(c_copy, c_orig) # should be a new object...
            self.assertEqual(c_copy.name, c_orig.name) # with the same name
            self.assertEqual(c_copy.points, c_orig.points) # and the same points




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
        pose.body.data = TensorflowMaskedTensor(tf.zeros_like(pose.body.data.tensor), tf.zeros_like(pose.body.data.mask))

        self.assertFalse(tf.reduce_all(pose.body.data == pose_copy.body.data), "Copy's data should not match original after original is replaced")

        # Create another copy and ensure it matches the first copy
        pose = pose_copy.copy()
        self.assertNotEqual(pose, pose_copy, "Copy of pose should not be 'equal' to original")
        
        self.assertTrue(tf.reduce_all(pose.body.data == pose_copy.body.data), "Copy's data should match original again")

        # Modify the copy and check that the original remains unchanged
        pose_copy.body.data.tensor = tf.zeros(pose_copy.body.data.tensor.shape)

        self.assertFalse(tf.reduce_all(pose.body.data == pose_copy.body.data), "Copy's data should not match original after copy is modified")




class TestPoseNumpyPoseBody(TestCase):
    """
    Testcases for Pose objects containing NumPy PoseBody data.
    """

    def test_pose_numpy_generated_with_correct_shape(self):
        pose = _get_random_pose_object_with_numpy_posebody(num_keypoints=5, frames_min=3)

        # does the header match the body?
        expected_keypoints_count_from_header = sum(len(c.points) for c in pose.header.components)
        self.assertEqual(expected_keypoints_count_from_header, pose.body.data.shape[-2])


    def test_pose_numpy_posebody_normalize_preserves_shape(self):
        """
        Tests if the normalization of Pose object with NumPy PoseBody preserves array shape.
        """
        pose = _get_random_pose_object_with_numpy_posebody(num_keypoints=5, frames_min=3)

        shape_before = pose.body.data.shape

        # in the mock data header components are named 0, 1, and so on
        # individual points are named 0_a, 0_b, and so on
        pose.normalize(pose.header.normalization_info(p1=("0", "0_a"), p2=("0", "0_b")))

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

    def test_pose_numpy_posebody_copy_creates_deepcopy(self):

        pose = _get_random_pose_object_with_numpy_posebody(num_keypoints=5, frames_min=3)

        pose_copy = pose.copy()
        self.assertNotEqual(pose, pose_copy, "Copy of pose should not be 'equal' to original")
        
        self.assertTrue(np.array_equal(pose.body.data, pose_copy.body.data), "Copy's data should match original")

        pose.body.data = ma.zeros(pose.body.data.shape)

        self.assertFalse(np.array_equal(pose.body.data, pose_copy.body.data), "Copy's data should not match original after original is replaced")

        pose = pose_copy.copy()

        self.assertTrue(np.array_equal(pose.body.data, pose_copy.body.data), "Copy's data should match original again")

        pose_copy.body.data[:] = 3.14

        self.assertFalse(np.array_equal(pose.body.data, pose_copy.body.data), "Copy's data should not match original after copy is modified")



        
class TestPoseTorchPoseBody(TestCase):

    def test_pose_torch_posebody_copy_tensors_detached(self):
        pose = _get_random_pose_object_with_torch_posebody(num_keypoints=5)
        pose_copy = pose.copy()

        self.assertFalse(pose.body.data.data.requires_grad, "Copied data should be detached from computation graph")
        self.assertFalse(pose_copy.body.data.mask.requires_grad, "Copied mask should be detached from computation graph")

    def test_pose_torch_posebody_copy_creates_deepcopy(self):
        pose = _get_random_pose_object_with_torch_posebody(num_keypoints=5)
        self.assertIsInstance(pose.body, TorchPoseBody)
        self.assertIsInstance(pose.body.data, TorchMaskedTensor)

        pose_copy = pose.copy()
        self.assertIsInstance(pose_copy.body, TorchPoseBody)
        self.assertIsInstance(pose_copy.body.data, TorchMaskedTensor)

        self.assertNotEqual(pose, pose_copy, "Copy of pose should not be 'equal' to original")
        self.assertTrue(pose.body.data.tensor.equal(pose_copy.body.data.tensor), "Copy's data should match original")
        self.assertTrue(pose.body.data.mask.equal(pose_copy.body.data.mask), "Copy's mask should match original")

        pose.body.data = TorchMaskedTensor(tensor=torch.zeros_like(pose.body.data.tensor),
                                                   mask=torch.ones_like(pose.body.data.mask))


        self.assertFalse(pose.body.data.tensor.equal(pose_copy.body.data.tensor), "Copy's data should not match original after original is replaced")
        self.assertFalse(pose.body.data.mask.equal(pose_copy.body.data.mask), "Copy's mask should not match original after original is replaced")

        pose = pose_copy.copy()

        self.assertTrue(pose.body.data.tensor.equal(pose_copy.body.data.tensor), "Copy's data should match original again")
        self.assertTrue(pose.body.data.mask.equal(pose_copy.body.data.mask), "Copy's mask should match original again")

        pose_copy.body.data.tensor.fill_(3.14)

        self.assertFalse(pose.body.data.tensor.equal(pose_copy.body.data.tensor), "Copy's data should not match original after copy is modified")

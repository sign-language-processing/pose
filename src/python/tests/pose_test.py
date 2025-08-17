import random
import string
from pathlib import Path
from typing import Optional, Tuple
from unittest import TestCase

import numpy as np
import numpy.ma as ma

from pose_format.numpy.pose_body import NumPyPoseBody
from pose_format.pose import Pose
from pose_format.pose_header import (PoseHeader, PoseHeaderComponent,
                                     PoseHeaderDimensions)


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
        raise ValueError(
            "Total keypoints must be at least component count+1 (so that 0 can have two), and component count must be positive")

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
        _create_pose_header_component(name=str(index), num_keypoints=keypoints_per_component[index]) for index in
        range(num_components)
    ]

    header = PoseHeader(version=1.0, dimensions=dimensions, components=components)

    return header


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


def _get_random_pose_object_with_numpy_posebody(num_keypoints: int, frames_min: int = 1, frames_max: int = 10,
                                                num_components=3) -> Pose:
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

    header = _create_pose_header(width=10, height=7, depth=0, num_components=num_components,
                                 num_keypoints=num_keypoints)

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
                expected_index += 1

        with self.assertRaises(ValueError):
            pose.header.get_point_index("component that doesn't exist", "")

        with self.assertRaises(ValueError):
            pose.header.get_point_index("0", "point that doesn't exist")

    def test_pose_remove_components(self):
        pose = _get_random_pose_object_with_numpy_posebody(num_keypoints=5)
        assert pose.body.data.shape[-2] == 5
        assert pose.body.data.shape[-1] == 2  # XY dimensions

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
        pose_copy = pose_copy.remove_components([], {point_to_remove[0]: [point_to_remove]})
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
        pose_copy = pose_copy.remove_components([component_to_remove],
                                                {component_to_remove_point_from: [point_to_remove]})
        self.assertNotIn(component_to_remove, [c.name for c in pose_copy.header.components])
        self.assertIn(component_to_remove_point_from,
                      [c.name for c in pose_copy.header.components])  # this should still be around
        self.assertIn("0_a", pose_copy.header.components[0].points)
        self.assertEqual(pose_copy.header.components[0].points,
                         pose.header.components[0].points)  # should be unaffected

        # can we remove a component and a point FROM that component without crashing
        component_to_remove = "0"
        point_to_remove = "0_a"
        pose_copy = pose.copy()
        self.assertIn(point_to_remove, pose_copy.header.components[0].points)
        pose_copy = pose_copy.remove_components([component_to_remove], {component_to_remove: [point_to_remove]})
        self.assertNotIn(component_to_remove, [c.name for c in pose_copy.header.components])
        self.assertNotIn(point_to_remove, pose_copy.header.components[0].points)
        self.assertEqual(pose_copy.header.components[0].points,
                         pose.header.components[1].points)  # should be unaffected

        # can we "remove" a component that doesn't exist without crashing
        component_to_remove = "NOT EXISTING"
        pose_copy = pose.copy()
        initial_count = len(pose_copy.header.components)
        pose_copy = pose_copy.remove_components([component_to_remove])
        self.assertEqual(initial_count, len(pose_copy.header.components))
        for c_orig, c_copy in zip(pose.header.components, pose_copy.header.components):
            self.assertNotEqual(c_copy, c_orig)  # should be a new object...
            self.assertEqual(c_copy.name, c_orig.name)  # with the same name
            self.assertEqual(c_copy.points, c_orig.points)  # and the same points

        # can we "remove" a point that doesn't exist from a component that does without crashing
        point_to_remove = "2_x"
        component_to_remove_point_from = "2"
        pose_copy = pose.copy()
        self.assertNotIn(point_to_remove, pose_copy.header.components[2].points)
        pose_copy = pose_copy.remove_components([], {component_to_remove_point_from: [point_to_remove]})
        self.assertNotIn(point_to_remove, pose_copy.header.components[2].points)
        for c_orig, c_copy in zip(pose.header.components, pose_copy.header.components):
            self.assertNotEqual(c_copy, c_orig)  # should be a new object...
            self.assertEqual(c_copy.name, c_orig.name)  # with the same name
            self.assertEqual(c_copy.points, c_orig.points)  # and the same points

        # can we "remove" an empty list of points
        component_to_remove_point_from = "2"
        pose_copy = pose.copy()
        initial_component_count = len(pose_copy.header.components)
        initial_point_count = len(pose_copy.header.components[2].points)
        pose_copy = pose_copy.remove_components([], {component_to_remove_point_from: []})
        self.assertEqual(initial_component_count, len(pose_copy.header.components))
        self.assertEqual(len(pose_copy.header.components[2].points), initial_point_count)
        for c_orig, c_copy in zip(pose.header.components, pose_copy.header.components):
            self.assertNotEqual(c_copy, c_orig)  # should be a new object...
            self.assertEqual(c_copy.name, c_orig.name)  # with the same name
            self.assertEqual(c_copy.points, c_orig.points)  # and the same points

        # can we remove a point from a component that doesn't exist
        point_to_remove = "2_x"
        component_to_remove_point_from = "NOT EXISTING"
        pose_copy = pose.copy()
        self.assertNotIn(point_to_remove, pose_copy.header.components[2].points)
        pose_copy = pose_copy.remove_components([], {component_to_remove_point_from: [point_to_remove]})
        self.assertNotIn(point_to_remove, pose_copy.header.components[2].points)
        for c_orig, c_copy in zip(pose.header.components, pose_copy.header.components):
            self.assertNotEqual(c_copy, c_orig)  # should be a new object...
            self.assertEqual(c_copy.name, c_orig.name)  # with the same name
            self.assertEqual(c_copy.points, c_orig.points)  # and the same points

    def test_pose_bbox(self):
        data_dir = Path(__file__).parent / "data"
        with open(data_dir / 'mediapipe.pose', 'rb') as f:
            pose = Pose.read(f)

        bbox = pose.bbox()

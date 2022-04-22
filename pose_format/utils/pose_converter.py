from typing import List

import numpy as np

from pose_format import Pose, PoseHeader
from pose_format.numpy import NumPyPoseBody
from pose_format.pose_header import PoseHeaderComponent
from pose_format.pose_visualizer import PoseVisualizer
from pose_format.utils.holistic import HAND_POINTS as HOLISTIC_HAND_POINTS, holistic_components

from pose_format.utils.openpose import HAND_POINTS as OPENPOSE_HAND_POINTS

LEFT_HAND_MAP = [(("hand_left_keypoints_2d", k1), ("LEFT_HAND_LANDMARKS", k2))
                 for k1, k2 in zip(OPENPOSE_HAND_POINTS, HOLISTIC_HAND_POINTS)]
RIGHT_HAND_MAP = [(("hand_right_keypoints_2d", k1), ("RIGHT_HAND_LANDMARKS", k2))
                  for k1, k2 in zip(OPENPOSE_HAND_POINTS, HOLISTIC_HAND_POINTS)]
BODY_MAP = [
    {("pose_keypoints_2d", "Nose"), ("POSE_LANDMARKS", "NOSE")},
    {("pose_keypoints_2d", "Neck"), ("POSE_LANDMARKS", ("RIGHT_SHOULDER", "LEFT_SHOULDER"))},
    {("pose_keypoints_2d", "RShoulder"), ("POSE_LANDMARKS", "RIGHT_SHOULDER")},
    {("pose_keypoints_2d", "RElbow"), ("POSE_LANDMARKS", "RIGHT_ELBOW")},
    {("pose_keypoints_2d", "RWrist"), ("POSE_LANDMARKS", "RIGHT_WRIST")},
    {("pose_keypoints_2d", "LShoulder"), ("POSE_LANDMARKS", "LEFT_SHOULDER")},
    {("pose_keypoints_2d", "LElbow"), ("POSE_LANDMARKS", "LEFT_ELBOW")},
    {("pose_keypoints_2d", "LWrist"), ("POSE_LANDMARKS", "LEFT_WRIST")},
    {("pose_keypoints_2d", "MidHip"), ("POSE_LANDMARKS", ("RIGHT_HIP", "LEFT_HIP"))},
    {("pose_keypoints_2d", "RHip"), ("POSE_LANDMARKS", "RIGHT_HIP")},
    {("pose_keypoints_2d", "RKnee"), ("POSE_LANDMARKS", "RIGHT_KNEE")},
    {("pose_keypoints_2d", "RAnkle"), ("POSE_LANDMARKS", "RIGHT_ANKLE")},
    {("pose_keypoints_2d", "LHip"), ("POSE_LANDMARKS", "LEFT_HIP")},
    {("pose_keypoints_2d", "LKnee"), ("POSE_LANDMARKS", "LEFT_KNEE")},
    {("pose_keypoints_2d", "LAnkle"), ("POSE_LANDMARKS", "LEFT_ANKLE")},
    {("pose_keypoints_2d", "REye"), ("POSE_LANDMARKS", "RIGHT_EYE")},
    {("pose_keypoints_2d", "LEye"), ("POSE_LANDMARKS", "LEFT_EYE")},
    {("pose_keypoints_2d", "REar"), ("POSE_LANDMARKS", "RIGHT_EAR")},
    {("pose_keypoints_2d", "LEar"), ("POSE_LANDMARKS", "LEFT_EAR")},
    {("pose_keypoints_2d", "LHeel"), ("POSE_LANDMARKS", "LEFT_HEEL")},
    {("pose_keypoints_2d", "RHeel"), ("POSE_LANDMARKS", "RIGHT_HEEL")},
]

POSES_MAP = BODY_MAP + LEFT_HAND_MAP + RIGHT_HAND_MAP


def convert_pose(pose: Pose, pose_components: List[PoseHeaderComponent]) -> Pose:
    pose_header = PoseHeader(version=pose.header.version,
                             dimensions=pose.header.dimensions,
                             components=pose_components)

    base_shape = (pose.body.data.shape[0], pose.body.data.shape[1], pose_header.total_points())
    data = np.zeros(shape=(*base_shape, len(pose_components[0].format) - 1), dtype=np.float)
    conf = np.zeros(shape=base_shape, dtype=np.float)

    original_components = set([c.name for c in pose.header.components])
    new_components = set([c.name for c in pose_components])

    # Create a mapping
    mapping = {}
    for points in POSES_MAP:
        original_point = None
        new_point = None
        for component, point in points:
            if component in original_components:
                original_point = (component, point)

            if component in new_components and isinstance(point, str):
                new_point = (component, point)

        if original_point is not None and new_point is not None:
            mapping[new_point] = original_point

    dims = min(len(pose_header.components[0].format), len(pose.header.components[0].format)) - 1
    for (c1, p1), (c2, p2) in mapping.items():
        p2 = tuple([p2]) if isinstance(p2, str) else p2
        p2s = [pose.header._get_point_index(c2, p) for p in list(p2)]
        p1_index = pose_header._get_point_index(c1, p1)
        data[:, :, p1_index, :dims] = pose.body.data[:, :, p2s, :dims].mean(axis=2)
        conf[:, :, p1_index] = pose.body.confidence[:, :, p2s].mean(axis=2)

    pose_body = NumPyPoseBody(fps=pose.body.fps, data=data, confidence=conf)

    return Pose(pose_header, pose_body)


def save_image(pose: Pose, name: str):
    visualizer = PoseVisualizer(pose, thickness=1)
    frame = next(iter(visualizer.draw(background_color=(255, 255, 255))))
    visualizer.save_frame(name, frame)


if __name__ == "__main__":
    with open("sample-data/video/sample.pose", "rb") as f:
        original_pose = Pose.read(f.read(), NumPyPoseBody)
        original_pose.focus()

    conv_holistic = convert_pose(original_pose, holistic_components())
    conv_openpose1 = convert_pose(original_pose, original_pose.header.components)
    conv_openpose2 = convert_pose(conv_holistic, original_pose.header.components)

    save_image(original_pose, "original.png")
    save_image(conv_holistic, "conv_holistic.png")
    save_image(conv_openpose1, "conv_openpose1.png")
    save_image(conv_openpose2, "conv_openpose2.png")

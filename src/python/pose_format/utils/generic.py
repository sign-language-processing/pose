from typing import Tuple

import numpy as np
from numpy import ma
from pose_format import Pose
from pose_format.numpy import NumPyPoseBody
from pose_format.pose_header import PoseHeader, PoseHeaderDimensions
from pose_format.utils.normalization_3d import PoseNormalizer
from pose_format.utils.openpose import OpenPose_Components


def pose_hide_legs(pose: Pose):
    if pose.header.components[0].name == "POSE_LANDMARKS":
        point_names = ["KNEE", "ANKLE", "HEEL", "FOOT_INDEX"]
        # pylint: disable=protected-access
        points = [
            pose.header._get_point_index("POSE_LANDMARKS", side + "_" + n)
            for n in point_names
            for side in ["LEFT", "RIGHT"]
        ]
        pose.body.data[:, :, points, :] = 0
        pose.body.confidence[:, :, points] = 0
    elif pose.header.components[0].name == "pose_keypoints_2d":
        point_names = ["Hip", "Knee", "Ankle", "BigToe", "SmallToe", "Heel"]
        # pylint: disable=protected-access
        points = [
            pose.header._get_point_index("pose_keypoints_2d", side + n)
            for n in point_names
            for side in ["L", "R"]
        ]
        pose.body.data[:, :, points, :] = 0
        pose.body.confidence[:, :, points] = 0
    else:
        raise ValueError("Unknown pose header schema for hiding legs")


def pose_shoulders(pose_header: PoseHeader):
    if pose_header.components[0].name == "POSE_LANDMARKS":
        return ("POSE_LANDMARKS", "RIGHT_SHOULDER"), ("POSE_LANDMARKS", "LEFT_SHOULDER")

    if pose_header.components[0].name == "BODY_135":
        return ("BODY_135", "RShoulder"), ("BODY_135", "LShoulder")

    if pose_header.components[0].name == "pose_keypoints_2d":
        return ("pose_keypoints_2d", "RShoulder"), ("pose_keypoints_2d", "LShoulder")

    raise ValueError("Unknown pose header schema for normalization")


def hands_indexes(pose_header: PoseHeader):
    if pose_header.components[0].name == "POSE_LANDMARKS":
        return [pose_header._get_point_index("LEFT_HAND_LANDMARKS", "MIDDLE_FINGER_MCP"),
                pose_header._get_point_index("RIGHT_HAND_LANDMARKS", "MIDDLE_FINGER_MCP")]

    if pose_header.components[0].name == "pose_keypoints_2d":
        return [pose_header._get_point_index("hand_left_keypoints_2d", "M_CMC"),
                pose_header._get_point_index("hand_right_keypoints_2d", "M_CMC")]


def pose_normalization_info(pose_header: PoseHeader):
    (c1, p1), (c2, p2) = pose_shoulders(pose_header)
    return pose_header.normalization_info(p1=(c1, p1), p2=(c2, p2))


def hands_components(pose_header: PoseHeader):
    if pose_header.components[0].name in ["POSE_LANDMARKS", "LEFT_HAND_LANDMARKS", "RIGHT_HAND_LANDMARKS"]:
        return ("LEFT_HAND_LANDMARKS", "RIGHT_HAND_LANDMARKS"), \
               ("WRIST", "PINKY_MCP", "INDEX_FINGER_MCP"), \
               ("WRIST", "MIDDLE_FINGER_MCP")

    if pose_header.components[0].name in ["pose_keypoints_2d", "hand_left_keypoints_2d", "hand_right_keypoints_2d"]:
        return ("hand_left_keypoints_2d", "hand_right_keypoints_2d"), \
               ("BASE", "P_CMC", "I_CMC"), \
               ("BASE", "M_CMC")

    raise ValueError("Unknown pose header")


def normalize_component_3d(pose, component_name: str, plane: Tuple[str, str, str], line: Tuple[str, str]):
    hand_pose = pose.get_components([component_name])
    plane = hand_pose.header.normalization_info(p1=(component_name, plane[0]),
                                                p2=(component_name, plane[1]),
                                                p3=(component_name, plane[2]))
    line = hand_pose.header.normalization_info(p1=(component_name, line[0]),
                                               p2=(component_name, line[1]))
    normalizer = PoseNormalizer(plane=plane, line=line)
    normalized_hand = normalizer(hand_pose.body.data)

    # Add normalized hand to pose
    pose.body.data = ma.concatenate([pose.body.data, normalized_hand], axis=2).astype(np.float32)
    pose.body.confidence = np.concatenate([pose.body.confidence, hand_pose.body.confidence], axis=2)


def normalize_hands_3d(pose: Pose, left_hand=True, right_hand=True):
    (left_hand_component, right_hand_component), plane, line = hands_components(pose.header)
    if left_hand:
        normalize_component_3d(pose, left_hand_component, plane, line)
    if right_hand:
        normalize_component_3d(pose, right_hand_component, plane, line)


def fake_pose(num_frames: int, fps=25, dims=2, components=OpenPose_Components):
    dimensions = PoseHeaderDimensions(width=1, height=1, depth=1)
    header = PoseHeader(version=0.1, dimensions=dimensions, components=components)

    total_points = header.total_points()
    data = np.random.randn(num_frames, 1, total_points, dims)
    confidence = np.random.randn(num_frames, 1, total_points)
    masked_data = ma.masked_array(data)

    body = NumPyPoseBody(fps=int(fps), data=masked_data, confidence=confidence)

    return Pose(header, body)


def correct_wrist(pose: Pose, hand: str) -> Pose:
    wrist_index = pose.header._get_point_index(f'{hand}_HAND_LANDMARKS', 'WRIST')
    wrist = pose.body.data[:, :, wrist_index]
    wrist_conf = pose.body.confidence[:, :, wrist_index]

    body_wrist_index = pose.header._get_point_index('POSE_LANDMARKS', f'{hand}_WRIST')
    body_wrist = pose.body.data[:, :, body_wrist_index]
    body_wrist_conf = pose.body.confidence[:, :, body_wrist_index]

    new_wrist_data = ma.where(wrist.data == 0, body_wrist, wrist)
    new_wrist_conf = ma.where(wrist_conf == 0, body_wrist_conf, wrist_conf)

    pose.body.data[:, :, body_wrist_index] = ma.masked_equal(new_wrist_data, 0)
    pose.body.confidence[:, :, body_wrist_index] = new_wrist_conf
    return pose


def correct_wrists(pose: Pose) -> Pose:
    pose = correct_wrist(pose, 'LEFT')
    pose = correct_wrist(pose, 'RIGHT')
    return pose


def reduce_holistic(pose: Pose) -> Pose:
    if pose.header.components[0].name != "POSE_LANDMARKS":
        return pose

    import mediapipe as mp
    points_set = set([p for p_tup in list(mp.solutions.holistic.FACEMESH_CONTOURS) for p in p_tup])
    face_contours = [str(p) for p in sorted(points_set)]

    ignore_names = [
        "EAR", "NOSE", "MOUTH", "EYE",  # Face
        "THUMB", "PINKY", "INDEX",  # Hands
        "KNEE", "ANKLE", "HEEL", "FOOT_INDEX"  # Feet
    ]

    body_component = [c for c in pose.header.components if c.name == 'POSE_LANDMARKS'][0]
    body_no_face_no_hands = [p for p in body_component.points if all([i not in p for i in ignore_names])]

    components = [c.name for c in pose.header.components if c.name != 'POSE_WORLD_LANDMARKS']
    return pose.get_components(components, {
        "FACE_LANDMARKS": face_contours,
        "POSE_LANDMARKS": body_no_face_no_hands
    })

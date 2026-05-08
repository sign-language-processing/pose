import numpy as np
from ..numpy.pose_body import NumPyPoseBody
from ..pose import Pose
from ..pose_header import PoseHeader, PoseHeaderComponent, PoseHeaderDimensions
from .alphapose import (
    FACE_POINTS, FACE_LIMBS_NAMES, GENERAL_HAND_POINTS,
    LEFT_HAND_POINTS, RIGHT_HAND_POINTS, HAND_LIMBS_NAMES,
    _map_limbs, load_alphapose_json, parse_keypoints_and_confidence, _apply_metadata,
)

# 133-keypoint body (no neck, head_top, or pelvis compared to 136)
BODY_POINTS = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
    "left_big_toe", "left_small_toe", "left_heel",
    "right_big_toe", "right_small_toe", "right_heel",
]

BODY_LIMBS_NAMES = [
    ("left_ankle", "left_knee"), ("left_knee", "left_hip"),
    ("right_ankle", "right_knee"), ("right_knee", "right_hip"),
    ("left_hip", "right_hip"),
    ("left_shoulder", "left_hip"), ("right_shoulder", "right_hip"),
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"), ("right_shoulder", "right_elbow"),
    ("left_elbow", "left_wrist"), ("right_elbow", "right_wrist"),
    ("left_eye", "right_eye"), ("nose", "left_eye"), ("nose", "right_eye"),
    ("left_eye", "left_ear"), ("right_eye", "right_ear"),
    ("left_ear", "left_shoulder"), ("right_ear", "right_shoulder"),
    ("left_ankle", "left_big_toe"), ("left_ankle", "left_small_toe"), ("left_ankle", "left_heel"),
    ("right_ankle", "right_big_toe"), ("right_ankle", "right_small_toe"), ("right_ankle", "right_heel"),
]


def get_alphapose_133_components():
    """
    Returns AlphaPose WholeBody-133 component definitions.

    Returns
    -------
    list of PoseHeaderComponent
        Components for body, face, left hand, and right hand.
    """
    hand_limbs = _map_limbs(GENERAL_HAND_POINTS, HAND_LIMBS_NAMES)
    return [
        PoseHeaderComponent(
            name="BODY_133",
            points=BODY_POINTS,
            limbs=_map_limbs(BODY_POINTS, BODY_LIMBS_NAMES),
            colors=[(0, 255, 0)],
            point_format="XYC"
        ),
        PoseHeaderComponent(
            name="FACE_133",
            points=FACE_POINTS,
            limbs=_map_limbs(FACE_POINTS, FACE_LIMBS_NAMES),
            colors=[(255, 255, 255)],
            point_format="XYC"
        ),
        PoseHeaderComponent(
            name="LEFT_HAND_133",
            points=GENERAL_HAND_POINTS,
            limbs=hand_limbs,
            colors=[(0, 255, 0)],
            point_format="XYC"
        ),
        PoseHeaderComponent(
            name="RIGHT_HAND_133",
            points=GENERAL_HAND_POINTS,
            limbs=hand_limbs,
            colors=[(255, 128, 0)],
            point_format="XYC"
        ),
    ]


AlphaPose133_Components = get_alphapose_133_components()


def load_alphapose_wholebody_from_json(input_path: str,
                                       version: float = 0.2,
                                       fps: float = 24,
                                       width: int = 1000,
                                       height: int = 1000,
                                       depth: int = 0) -> Pose:
    """
    Load an AlphaPose WholeBody-133 JSON file into a Pose object.

    Raises ValueError if the file contains 136-keypoint data; use
    pose_format.utils.alphapose.load_alphapose_wholebody_from_json for
    auto-detection.

    Parameters
    ----------
    input_path : str
        Path to the AlphaPose JSON file.
    version : float
        Pose format version written to the header.
    fps : float
        Frames per second. Overridden by JSON metadata if present.
    width : int
        Frame width in pixels. Overridden by JSON metadata if present.
    height : int
        Frame height in pixels. Overridden by JSON metadata if present.
    depth : int
        Depth dimension size (0 for 2D poses).

    Returns
    -------
    Pose
        Loaded pose with header and body.

    Raises
    ------
    ValueError
        If the JSON contains 136-keypoint data.
    """
    frames, metadata = load_alphapose_json(input_path)
    fps, width, height = _apply_metadata(metadata, fps, width, height)

    frames_xy = []
    frames_conf = []

    for item in frames:
        xy, conf, n_keypoints = parse_keypoints_and_confidence(item["keypoints"])
        if n_keypoints == 136:
            raise ValueError(
                "This file contains 136-keypoint AlphaPose data. "
                "Use pose_format.utils.alphapose.load_alphapose_wholebody_from_json instead."
            )
        frames_xy.append(xy)
        frames_conf.append(conf)

    xy_data = np.stack(frames_xy, axis=0)[:, None, :, :]
    conf_data = np.stack(frames_conf, axis=0)[:, None, :]

    header = PoseHeader(
        version=version,
        dimensions=PoseHeaderDimensions(width=width, height=height, depth=depth),
        components=AlphaPose133_Components,
    )
    body = NumPyPoseBody(fps=fps, data=xy_data, confidence=conf_data)
    return Pose(header, body)

import numpy as np
from ..numpy.pose_body import NumPyPoseBody
from ..pose import Pose
from ..pose_header import PoseHeader, PoseHeaderComponent, PoseHeaderDimensions
from pose_format.utils.alphapose import load_alphapose_json, parse_keypoints_and_confidence, _map_limbs

BODY_POINTS = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    "left_big_toe",
    "left_small_toe",
    "left_heel",
    "right_big_toe",
    "right_small_toe",
    "right_heel",
]

FACE_POINTS = [f"face_{i}" for i in range(68)]
LEFT_HAND_POINTS = [f"left_hand_{i}" for i in range(21)]
RIGHT_HAND_POINTS = [f"right_hand_{i}" for i in range(21)]
GENERAL_HAND_POINTS = [f"hand_{i}" for i in range(21)]

BODY_LIMBS_NAMES = [
    ("left_ankle", "left_knee"),
    ("left_knee", "left_hip"),
    ("right_ankle", "right_knee"),
    ("right_knee", "right_hip"),
    ("left_hip", "right_hip"),
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("right_shoulder", "right_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_elbow", "right_wrist"),
    ("left_eye", "right_eye"),
    ("nose", "left_eye"),
    ("nose", "right_eye"),
    ("left_eye", "left_ear"),
    ("right_eye", "right_ear"),
    ("left_ear", "left_shoulder"),
    ("right_ear", "right_shoulder"),
    ("left_ankle", "left_big_toe"),
    ("left_ankle", "left_small_toe"),
    ("left_ankle", "left_heel"),
    ("right_ankle", "right_big_toe"),
    ("right_ankle", "right_small_toe"),
    ("right_ankle", "right_heel"),
]

LEFT_HAND_LIMBS_NAMES = [
    ("left_hand_0", "left_hand_1"),
    ("left_hand_1", "left_hand_2"),
    ("left_hand_2", "left_hand_3"),
    ("left_hand_3", "left_hand_4"),
    ("left_hand_0", "left_hand_5"),
    ("left_hand_5", "left_hand_6"),
    ("left_hand_6", "left_hand_7"),
    ("left_hand_7", "left_hand_8"),
    ("left_hand_0", "left_hand_9"),
    ("left_hand_9", "left_hand_10"),
    ("left_hand_10", "left_hand_11"),
    ("left_hand_11", "left_hand_12"),
    ("left_hand_0", "left_hand_13"),
    ("left_hand_13", "left_hand_14"),
    ("left_hand_14", "left_hand_15"),
    ("left_hand_15", "left_hand_16"),
    ("left_hand_0", "left_hand_17"),
    ("left_hand_17", "left_hand_18"),
    ("left_hand_18", "left_hand_19"),
    ("left_hand_19", "left_hand_20"),
]

RIGHT_HAND_LIMBS_NAMES = [
    ("right_hand_0", "right_hand_1"),
    ("right_hand_1", "right_hand_2"),
    ("right_hand_2", "right_hand_3"),
    ("right_hand_3", "right_hand_4"),
    ("right_hand_0", "right_hand_5"),
    ("right_hand_5", "right_hand_6"),
    ("right_hand_6", "right_hand_7"),
    ("right_hand_7", "right_hand_8"),
    ("right_hand_0", "right_hand_9"),
    ("right_hand_9", "right_hand_10"),
    ("right_hand_10", "right_hand_11"),
    ("right_hand_11", "right_hand_12"),
    ("right_hand_0", "right_hand_13"),
    ("right_hand_13", "right_hand_14"),
    ("right_hand_14", "right_hand_15"),
    ("right_hand_15", "right_hand_16"),
    ("right_hand_0", "right_hand_17"),
    ("right_hand_17", "right_hand_18"),
    ("right_hand_18", "right_hand_19"),
    ("right_hand_19", "right_hand_20"),
]


def get_alphapose_133_components():
    """
    Returns AlphaPose WholeBody-133 component definitions.

    Returns
    -------
    list of PoseHeaderComponent
        Components for body, face, left hand, and right hand.
    """
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
            limbs=[],
            colors=[(255, 255, 255)],
            point_format="XYC"
        ),
        PoseHeaderComponent(
            name="LEFT_HAND_133",
            points=GENERAL_HAND_POINTS,
            limbs=_map_limbs(LEFT_HAND_POINTS, LEFT_HAND_LIMBS_NAMES),
            colors=[(0, 255, 0)],
            point_format="XYC"
        ),
        PoseHeaderComponent(
            name="RIGHT_HAND_133",
            points=GENERAL_HAND_POINTS,
            limbs=_map_limbs(RIGHT_HAND_POINTS, RIGHT_HAND_LIMBS_NAMES),
            colors=[(255, 128, 0)],
            point_format="XYC"
        ),
    ]


AlphaPose133_Components = get_alphapose_133_components()


def reorder_133_kpts(xy, conf):
    """
    Reorder 133-keypoint flat arrays into BODY + FACE + LEFT_HAND + RIGHT_HAND.

    AlphaPose 133 layout: BODY 0-22, FACE 23-90, LH 91-111, RH 112-132.
    """
    xy_reordered = np.concatenate([xy[0:23], xy[23:91], xy[91:112], xy[112:133]], axis=0)
    conf_reordered = np.concatenate([conf[0:23], conf[23:91], conf[91:112], conf[112:133]], axis=0)
    return xy_reordered, conf_reordered


def load_alphapose_wholebody_from_json(input_path: str,
                                       version: float = 0.2,
                                       fps: float = 24,
                                       width: int = 1000,
                                       height: int = 1000,
                                       depth: int = 0) -> Pose:
    """
    Load an AlphaPose WholeBody-133 JSON file into a Pose object.

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
        If the JSON contains 136-keypoint data. Use alphapose.py instead.
    """
    frames, metadata = load_alphapose_json(input_path)

    if metadata is not None:
        if metadata.get("fps") is not None:
            fps = metadata["fps"]
        if metadata.get("width") is not None:
            width = metadata["width"]
        if metadata.get("height") is not None:
            height = metadata["height"]

    frames_xy = []
    frames_conf = []

    for item in frames:
        xy, conf, n_keypoints = parse_keypoints_and_confidence(item["keypoints"])
        if n_keypoints == 136:
            raise ValueError(
                f"This file contains 136-keypoint AlphaPose data. "
                f"Use pose_format.utils.alphapose.load_alphapose_wholebody_from_json instead."
            )
        xy_ord, conf_ord = reorder_133_kpts(xy, conf)
        frames_xy.append(xy_ord)
        frames_conf.append(conf_ord)

    xy_data = np.stack(frames_xy, axis=0)[:, None, :, :]
    conf_data = np.stack(frames_conf, axis=0)[:, None, :]

    header = PoseHeader(
        version=version,
        dimensions=PoseHeaderDimensions(width=width, height=height, depth=depth),
        components=get_alphapose_133_components(),
    )
    body = NumPyPoseBody(fps=fps, data=xy_data, confidence=conf_data)
    return Pose(header, body)

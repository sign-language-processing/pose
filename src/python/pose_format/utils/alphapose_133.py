import numpy as np
from ..numpy.pose_body import NumPyPoseBody
from ..pose import Pose
from ..pose_header import PoseHeader, PoseHeaderComponent, PoseHeaderDimensions
from .alphapose import load_alphapose_json, parse_keypoints_and_confidence, _apply_metadata
from .cocowholebody133_header import (
    BODY_POINTS, FACE_POINTS, HAND_POINTS,
    BODY_LIMBS, FACE_LIMBS, HAND_LIMBS,
)


def get_alphapose_133_components():
    """
    Returns AlphaPose WholeBody-133 component definitions.

    AlphaPose 133 is the COCO Wholebody 133-keypoint format. Point lists and
    limb connectivity are shared with cocowholebody133_header; only the component
    names differ (suffixed with _133 for backward compatibility).

    Returns
    -------
    list of PoseHeaderComponent
        Components for body, face, left hand, and right hand.
    """
    return [
        PoseHeaderComponent(
            name="BODY_133",
            points=BODY_POINTS,
            limbs=BODY_LIMBS,
            colors=[(0, 255, 0)],
            point_format="XYC"
        ),
        PoseHeaderComponent(
            name="FACE_133",
            points=FACE_POINTS,
            limbs=FACE_LIMBS,
            colors=[(255, 255, 255)],
            point_format="XYC"
        ),
        PoseHeaderComponent(
            name="LEFT_HAND_133",
            points=HAND_POINTS,
            limbs=HAND_LIMBS,
            colors=[(0, 255, 0)],
            point_format="XYC"
        ),
        PoseHeaderComponent(
            name="RIGHT_HAND_133",
            points=HAND_POINTS,
            limbs=HAND_LIMBS,
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

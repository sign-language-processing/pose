import json
import math
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from numpy import ma

from ..numpy.pose_body import NumPyPoseBody
from ..pose import Pose
from ..pose_header import PoseHeader, PoseHeaderComponent, PoseHeaderDimensions

BODY_POINTS = [
    "Nose", "Neck", "RShoulder", "RElbow", "RWrist", "LShoulder", "LElbow", "LWrist", "MidHip", "RHip", "RKnee",
    "RAnkle", "LHip", "LKnee", "LAnkle", "REye", "LEye", "REar", "LEar", "LBigToe", "LSmallToe", "LHeel", "RBigToe",
    "RSmallToe", "RHeel"
]

# Based on https://github.com/CMU-Perceptual-Computing-Lab/openpose/raw/master/.github/media/keypoints_pose_25.png
# Everything sprouts out of the neck
BODY_LIMBS = [
    # Body
    ("Neck", "RShoulder"),
    ("RShoulder", "RElbow"),
    ("RElbow", "RWrist"),
    ("Neck", "LShoulder"),
    ("LShoulder", "LElbow"),
    ("LElbow", "LWrist"),
    ("Neck", "MidHip"),
    # Face
    ("Nose", "LEye"),
    ("Nose", "REye"),
    ("Nose", "LEar"),
    ("Nose", "REar"),
    ("Neck", "Nose"),
    # Legs
    ("MidHip", "RHip"),
    ("RHip", "RKnee"),
    ("RKnee", "RAnkle"),
    ("MidHip", "LHip"),
    ("LHip", "LKnee"),
    ("LKnee", "LAnkle"),
    # Feet
    ("RAnkle", "RHeel"),
    ("RAnkle", "RBigToe"),
    ("RBigToe", "RSmallToe"),
    ("LAnkle", "LHeel"),
    ("LAnkle", "LBigToe"),
    ("LBigToe", "LSmallToe"),
]

#        8   12  16  20
#        |   |   |   |
#        7   11  15  19
#    4   |   |   |   |
#    |   6   10  14  18
#    3   |   |   |   |
#    |   5---9---13--17
#    2    \         /
#     \    \       /
#      1    \     /
#       \    \   /
#        ------0-

# Anatomy guide https://www.assh.org/handcare/blog/anatomy-101-finger-joints
HAND_POINTS = [
    "BASE",
    "T_STT",
    "T_BCMC",
    "T_MCP",
    "T_IP",  # Thumb
    "I_CMC",
    "I_MCP",
    "I_PIP",
    "I_DIP",  # Index
    "M_CMC",
    "M_MCP",
    "M_PIP",
    "M_DIP",  # Middle
    "R_CMC",
    "R_MCP",
    "R_PIP",
    "R_DIP",  # Ring
    "P_CMC",
    "P_MCP",
    "P_PIP",
    "P_DIP",  # Pinky
]

# Based on https://github.com/CMU-Perceptual-Computing-Lab/openpose/raw/master/.github/media/keypoints_hand.png
# Everything sprouts out of the base
HAND_LIMBS = [
    ("BASE", "T_STT"),
    ("BASE", "I_CMC"),
    ("BASE", "M_CMC"),
    ("BASE", "R_CMC"),
    ("BASE", "P_CMC"),  # Base
    ("T_STT", "T_BCMC"),
    ("T_BCMC", "T_MCP"),
    ("T_MCP", "T_IP"),  # Thumb
    ("I_CMC", "I_MCP"),
    ("I_MCP", "I_PIP"),
    ("I_PIP", "I_DIP"),  # Index
    ("M_CMC", "M_MCP"),
    ("M_MCP", "M_PIP"),
    ("M_PIP", "M_DIP"),  # Middle
    ("R_CMC", "R_MCP"),
    ("R_MCP", "R_PIP"),
    ("R_PIP", "R_DIP"),  # Ring
    ("P_CMC", "P_MCP"),
    ("P_MCP", "P_PIP"),
    ("P_PIP", "P_DIP"),  # Pinky
]

# Based on https://github.com/CMU-Perceptual-Computing-Lab/openpose/raw/master/.github/media/keypoints_face.png
# Border
FACE_BORDER_POINTS = ["FB_" + str(i) for i in range(17)]
FACE_BORDER_LIMBS_LEFT = [("FB_" + str(i), "FB_" + str(i - 1)) for i in reversed(range(1, 9))]
FACE_BORDER_LIMBS_RIGHT = [("FB_" + str(i), "FB_" + str(i + 1)) for i in range(8, 16)]

# Lips
FACE_OUTER_LIPS_POINTS = ["FLO_" + str(i) for i in range(48, 60)]
FACE_OUTER_LIPS_LIMBS = [("FLO_" + str(i), "FLO_" + str(i + 1)) for i in range(48, 59)] + [("FLO_59", "FLO_48")]
FACE_INNER_LIPS_POINTS = ["FLI_" + str(i) for i in range(60, 68)]
FACE_INNER_LIPS_LIMBS = [("FLI_" + str(i), "FLI_" + str(i + 1)) for i in range(60, 67)] + [("FLI_67", "FLI_60")]

# Nose
FACE_NOSE_POINTS = ["FN_" + str(i) for i in range(27, 36)]
FACE_NOSE_BRIDGE_LIMBS = [("FN_" + str(i), "FN_" + str(i + 1)) for i in range(27, 31)]
FACE_NOSE_HORIZONTAL_LIMBS = [("FN_" + str(i), "FN_" + str(i + 1)) for i in range(31, 35)]
FACE_NOSE_LIMBS = FACE_NOSE_BRIDGE_LIMBS + FACE_NOSE_HORIZONTAL_LIMBS + [("FN_30", "FN_33")]

# Eyebrows
FACE_EYE_POINTS = ["FE_" + str(i) for i in range(36, 48)]
FACE_EYE_LEFT_LIMBS = [("FE_" + str(i), "FE_" + str(i + 1)) for i in range(36, 41)] + [("FE_41", "FE_36")]
FACE_EYE_RIGHT_LIMBS = [("FE_" + str(i), "FE_" + str(i + 1)) for i in range(42, 47)] + [("FE_47", "FE_42")]
FACE_PUPILS_POINTS = ["FP_68", "FP_69"]

# Eyes
FACE_EYEBROWS_POINTS = ["FEB_" + str(i) for i in range(17, 27)]
FACE_EYEBROW_LEFT_LIMBS = [("FEB_" + str(i), "FEB_" + str(i + 1)) for i in range(17, 21)]
FACE_EYEBROW_RIGHT_LIMBS = [("FEB_" + str(i), "FEB_" + str(i + 1)) for i in range(22, 26)]

# Face points, in order
FACE_POINTS = FACE_BORDER_POINTS + FACE_EYEBROWS_POINTS + FACE_NOSE_POINTS + FACE_EYE_POINTS + FACE_OUTER_LIPS_POINTS + FACE_INNER_LIPS_POINTS + FACE_PUPILS_POINTS
FACE_LIMBS: List[Tuple[str, str]] = FACE_BORDER_LIMBS_LEFT + FACE_BORDER_LIMBS_RIGHT + FACE_OUTER_LIPS_LIMBS + \
                                    FACE_INNER_LIPS_LIMBS + FACE_NOSE_LIMBS + FACE_EYEBROW_LEFT_LIMBS + \
                                    FACE_EYEBROW_RIGHT_LIMBS + FACE_EYE_LEFT_LIMBS + FACE_EYE_RIGHT_LIMBS

HAND_POINTS_COLOR = [[192, 0, 0], [0, 0, 192], [0, 192, 0], [0, 192, 192], [192, 127, 0], [127, 127, 127]]

OPENPOSE_FRAME_PATTERN = "(?:^|\D)(\d+)\\_keypoints.json"


# Definition of OpenPose Components


def limbs_index(limbs: List[Tuple[str, str]], points: List[str]) -> List[Tuple[int, int]]:
    """
    Convert limb names to indices based on a list of points.
    
    Parameters
    ----------
    limbs : list of tuple of str
        limbs defined by point names
    points : list of str (List[str])
        list of point names
    
    Returns
    -------
    list of tuple of int
        List of limbs defined by point indices
    """
    return [(points.index(p1), points.index(p2)) for p1, p2 in limbs]


hand_colors = [
    tuple([math.floor(x + 35 * (i % 4)) for x in HAND_POINTS_COLOR[i // 4]])
    for i in range(-1, len(HAND_POINTS) - 1)
]
# Override pinky, for accessibility
hand_colors[17] = hand_colors[20] = (255, 128, 0)
hand_colors[19] = (255, 153, 51)
hand_colors[18] = (255, 178, 102)

OpenPose_Hand_Component = lambda name: PoseHeaderComponent(
    name=name, points=HAND_POINTS, limbs=limbs_index(HAND_LIMBS, HAND_POINTS), colors=hand_colors, point_format="XYC")
OpenPose_Hand_Component.__doc__ = """
This "lambda" function creates a PoseHeaderComponent using 'name' and
a constants for points, limbs, colors, and format.
"""

#     {
#     "points": HAND_POINTS,
#     "colors": [[math.floor(x + 35 * (i % 4)) for x in HAND_POINTS_COLOR[i // 4]] for i in
#                range(-1, len(HAND_POINTS) - 1)],
#     "limbs": HAND_LIMBS,
#     "point_format": {"X": 0, "Y": 1, "C": 2}
# }

OpenPose_Components = [
    PoseHeaderComponent(name="pose_keypoints_2d",
                        points=BODY_POINTS,
                        limbs=limbs_index(BODY_LIMBS, BODY_POINTS),
                        colors=[(255, 0, 0)],
                        point_format="XYC"),
    PoseHeaderComponent(name="face_keypoints_2d",
                        points=FACE_POINTS,
                        limbs=limbs_index(FACE_LIMBS, FACE_POINTS),
                        colors=[(128, 0, 0)],
                        point_format="XYC"),
    OpenPose_Hand_Component("hand_left_keypoints_2d"),
    OpenPose_Hand_Component("hand_right_keypoints_2d"),
]

OpenPoseFrame = Dict[str, Any]
OpenPoseFrames = Dict[int, OpenPoseFrame]


def load_openpose(frames: OpenPoseFrames,
                  fps: float = 24,
                  width: int = 1000,
                  height: int = 1000,
                  depth: int = 0,
                  num_frames: Optional[int] = None) -> Pose:
    """
    Loads a dictionary of OpenPose frames into a Pose object.
    
    Parameters
    ----------
    frames : dict
        Dictionary where keys are frame IDs, and values are individual frames. Each individual frame is also a dictionary.
    fps : float, optional
        Framerate, default is 24.
    width : int, optional
        Width of pose space, default is 1000.
    height : int, optional
        Height of pose space, default is 1000.
    depth : int, optional
        Depth of pose space, default is 0.
    num_frames : int, optional
        Number of frames when it's known and cannot be derived from OpenPose files. That is the case if the last frame(s) of a video are missing from the OpenPose output.
        Default is None.

    Returns
    -------
    Pose
        Pose object with a header specific to OpenPose and a body that contains a single array.
    """
    dimensions = PoseHeaderDimensions(width=width, height=height, depth=depth)

    header: PoseHeader = PoseHeader(version=0.2, dimensions=dimensions, components=OpenPose_Components)

    total_points = header.total_points()

    if num_frames is None:
        # take the maximum of all frame IDs because some frames could be missing
        num_frames = max(frames.keys()) + 1

    # array dimensions: (frames, person, points, dimensions)
    people = max([len(frame["people"]) for frame in frames.values()])
    data = np.zeros(shape=(num_frames, people, total_points, 2), dtype=np.float32)
    confidence = np.zeros(shape=(num_frames, people, total_points), dtype=np.float32)

    for frame_id, frame in frames.items():
        for person_id, person in enumerate(frame["people"]):
            keypoint_id = 0
            for component in header.components:
                numbers = person[component.name]
                for k in range(0, len(numbers), len(component.format)):
                    data[frame_id, person_id, keypoint_id, 0] = numbers[k + 0]
                    data[frame_id, person_id, keypoint_id, 1] = numbers[k + 1]
                    confidence[frame_id, person_id, keypoint_id] = numbers[k + 2]
                    keypoint_id += 1

    # Mask data
    mask = confidence == 0  # 0 means no-mask, 1 means with-mask
    stacked_confidence = np.stack([mask, mask], axis=3)
    masked_data = ma.masked_array(data, mask=stacked_confidence)

    body = NumPyPoseBody(fps=int(fps), data=masked_data, confidence=confidence)

    return Pose(header, body)


def get_frame_id(filename: str, pattern: str) -> int:
    """
    Parses a filename to find the ID of a frame. Example file name for frame with ID 17: `CAM2_000000000017_keypoints.json`.

    Parameters
    ----------
    filename : str
        Name of the openpose frame file.
    pattern : str, optional
        Regex pattern to extract frame ID, default is OPENPOSE_FRAME_PATTERN.

    Returns
    -------
    int
        Frame ID as an integer.
    """
    m = re.findall(pattern, filename)
    frame_id = int(m[-1])

    return frame_id


def load_frames_directory_dict(directory: str, pattern: str) -> OpenPoseFrames:
    """
    Load a pose directory where each frame's pose data is stored in a separate file 
    following a specific naming scheme. 
    Filenames must adhere to the format: `[ARBITRARY CHARACTERS]_[FRAME_ID]_keypoints.json`.
    Example: For a frame with ID 17, the filename would be `CAM2_000000000017_keypoints.json`.

    Parameters
    ----------
    directory : str
        Path to the folder containing pose files.
    pattern : str, optional
        Regular expression pattern to identify and parse frame filenames. The default pattern expects 
        filenames of the form `[ARBITRARY CHARACTERS]_[FRAME_ID]_keypoints.json`.

    Returns
    -------
    OpenPoseFrames
        Dictionary where keys are frame IDs (int) and values are the corresponding frames (dict).

    Examples
    --------
    >>> frames = load_frames_directory_dict("path/to/frames")
    >>> print(frames[17])
    {...}  # content of CAM2_000000000017_keypoints.json
    """
    frames = {}  # type: OpenPoseFrames

    with os.scandir(directory) as entry_iterator:
        for entry in entry_iterator:  # type: os.DirEntry
            with open(entry.path, "r") as f:
                frame_id = get_frame_id(entry.name, pattern=pattern)
                frame_dict = json.load(f)
                frames[frame_id] = frame_dict

    return frames


def load_openpose_directory(directory: str,
                            fps: float = 24,
                            width: int = 1000,
                            height: int = 1000,
                            depth: int = 0,
                            num_frames: Optional[int] = None) -> Pose:
    """
    Loads pose data from a directory containing OpenPose files and return a `Pose` object.

    Parameters
    ----------
    directory : str
        Path to the folder that contains pose files.
    fps : float, optional
        Framerate. Default is 24.
    width : int, optional
        Width of pose space. Default 1000.
    height : int, optional
        Height of pose space, default; 1000.
    depth : int, optional
        Depth of pose space, default; 0.
    num_frames : int, optional
        Number of frames when known and cannot be derived from OpenPose files. 
        This can be the case if the last frame(s) of a video are missing from the OpenPose output.
        Default is None.

    Returns
    -------
    Pose
        Pose object with a header specific to OpenPose and a body containing a single array.

    Examples
    --------
    >>> pose = load_openpose_directory("path/to/frames")
    >>> print(pose.header)
    PoseHeader(...)
    """
    frames = load_frames_directory_dict(directory=directory, pattern=OPENPOSE_FRAME_PATTERN)

    return load_openpose(frames, fps=fps, width=width, height=height, depth=depth, num_frames=num_frames)

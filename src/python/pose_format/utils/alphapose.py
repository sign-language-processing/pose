import re
import json
import numpy as np
from ..numpy.pose_body import NumPyPoseBody
from ..pose import Pose
from ..pose_header import PoseHeader, PoseHeaderComponent, PoseHeaderDimensions

# --- Shared constants (same across all AlphaPose WholeBody variants) ---

FACE_POINTS = [f"face_{i}" for i in range(68)]
GENERAL_HAND_POINTS = [f"hand_{i}" for i in range(21)]

# Left/right-specific point name lists — exported for reference / downstream use
LEFT_HAND_POINTS = [f"left_hand_{i}" for i in range(21)]
RIGHT_HAND_POINTS = [f"right_hand_{i}" for i in range(21)]

# Single hand limb table using generic point names, shared by both hand components
HAND_LIMBS_NAMES = [
    ("hand_0", "hand_1"), ("hand_1", "hand_2"), ("hand_2", "hand_3"), ("hand_3", "hand_4"),
    ("hand_0", "hand_5"), ("hand_5", "hand_6"), ("hand_6", "hand_7"), ("hand_7", "hand_8"),
    ("hand_0", "hand_9"), ("hand_9", "hand_10"), ("hand_10", "hand_11"), ("hand_11", "hand_12"),
    ("hand_0", "hand_13"), ("hand_13", "hand_14"), ("hand_14", "hand_15"), ("hand_15", "hand_16"),
    ("hand_0", "hand_17"), ("hand_17", "hand_18"), ("hand_18", "hand_19"), ("hand_19", "hand_20"),
]

FACE_LIMBS_NAMES = [
    ("face_0", "face_1"), ("face_1", "face_2"), ("face_2", "face_3"), ("face_3", "face_4"),
    ("face_4", "face_5"), ("face_5", "face_6"), ("face_6", "face_7"), ("face_7", "face_8"),
    ("face_8", "face_9"), ("face_9", "face_10"), ("face_10", "face_11"), ("face_11", "face_12"),
    ("face_12", "face_13"), ("face_13", "face_14"), ("face_14", "face_15"), ("face_15", "face_16"),
    ("face_17", "face_18"), ("face_18", "face_19"), ("face_19", "face_20"), ("face_20", "face_21"),
    ("face_22", "face_23"), ("face_23", "face_24"), ("face_24", "face_25"), ("face_25", "face_26"),
    ("face_27", "face_28"), ("face_28", "face_29"), ("face_29", "face_30"),
    ("face_31", "face_32"), ("face_32", "face_33"), ("face_33", "face_34"), ("face_34", "face_35"),
    ("face_36", "face_37"), ("face_37", "face_38"), ("face_38", "face_39"),
    ("face_39", "face_40"), ("face_40", "face_41"),
    ("face_42", "face_43"), ("face_43", "face_44"), ("face_44", "face_45"),
    ("face_45", "face_46"), ("face_46", "face_47"),
    ("face_48", "face_49"), ("face_49", "face_50"), ("face_50", "face_51"), ("face_51", "face_52"),
    ("face_52", "face_53"), ("face_53", "face_54"), ("face_54", "face_55"), ("face_55", "face_56"),
    ("face_56", "face_57"), ("face_57", "face_58"), ("face_58", "face_59"), ("face_59", "face_60"),
    ("face_60", "face_61"), ("face_61", "face_62"), ("face_62", "face_63"), ("face_63", "face_64"),
    ("face_64", "face_65"), ("face_65", "face_66"), ("face_66", "face_67"),
]

# --- 136-keypoint body (WholeBody-136 has neck, head_top, pelvis) ---

BODY_POINTS = [
    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
    'left_knee', 'right_knee', 'left_ankle', 'right_ankle',
    'head_top', 'neck', 'pelvis',
    'left_big_toe', 'right_big_toe', 'left_small_toe', 'right_small_toe',
    'left_heel', 'right_heel',
]

BODY_LIMBS_NAMES = [
    ("nose", "left_eye"), ("nose", "right_eye"),
    ("left_eye", "left_ear"), ("right_eye", "right_ear"),
    ("head_top", "neck"),
    ("left_shoulder", "neck"), ("right_shoulder", "neck"),
    ("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist"),
    ("neck", "pelvis"), ("pelvis", "left_hip"), ("pelvis", "right_hip"),
    ("left_hip", "left_knee"), ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"), ("right_knee", "right_ankle"),
    ("left_ankle", "left_heel"), ("left_heel", "left_big_toe"), ("left_heel", "left_small_toe"),
    ("right_ankle", "right_heel"), ("right_heel", "right_big_toe"), ("right_heel", "right_small_toe"),
]


def _map_limbs(points, limbs):
    index_map = {name: idx for idx, name in enumerate(points)}
    return [(index_map[a], index_map[b]) for (a, b) in limbs]


def get_alphapose_components():
    """
    Returns AlphaPose WholeBody-136 component definitions.

    Returns
    -------
    list of PoseHeaderComponent
        Components for body, face, left hand, and right hand.
    """
    hand_limbs = _map_limbs(GENERAL_HAND_POINTS, HAND_LIMBS_NAMES)
    return [
        PoseHeaderComponent(
            name="BODY_136",
            points=BODY_POINTS,
            limbs=_map_limbs(BODY_POINTS, BODY_LIMBS_NAMES),
            colors=[(0, 255, 0)],
            point_format="XYC"
        ),
        PoseHeaderComponent(
            name="FACE_136",
            points=FACE_POINTS,
            limbs=_map_limbs(FACE_POINTS, FACE_LIMBS_NAMES),
            colors=[(255, 255, 255)],
            point_format="XYC"
        ),
        PoseHeaderComponent(
            name="LEFT_HAND_136",
            points=GENERAL_HAND_POINTS,
            limbs=hand_limbs,
            colors=[(0, 255, 0)],
            point_format="XYC"
        ),
        PoseHeaderComponent(
            name="RIGHT_HAND_136",
            points=GENERAL_HAND_POINTS,
            limbs=hand_limbs,
            colors=[(255, 128, 0)],
            point_format="XYC"
        ),
    ]


AlphaPose_Components = get_alphapose_components()


def load_alphapose_json(json_path):
    """
    Load AlphaPose results in either of two formats.

    FORMAT A (original list):
        [{"image_id": "0.jpg", "keypoints": [x0, y0, c0, ...]}, ...]

    FORMAT B (extended dict with metadata):
        {"frames": [...], "metadata": {"fps": float, "width": int, "height": int}}

    Returns
    -------
    frames : list
        Sorted list of frame detections.
    meta : dict or None
        Metadata dict if present, else None.
    """
    with open(json_path, "r") as f:
        raw = json.load(f)

    if isinstance(raw, dict) and "frames" in raw:
        frames = raw["frames"]
        meta = {
            "fps": raw.get("metadata", {}).get("fps", None),
            "width": raw.get("metadata", {}).get("width", None),
            "height": raw.get("metadata", {}).get("height", None),
        }
    else:
        frames = raw
        meta = None

    def extract_frame_number(item):
        matches = re.findall(r"\d+", item["image_id"])
        return int(matches[-1]) if matches else -1

    return sorted(frames, key=extract_frame_number), meta


def parse_keypoints_and_confidence(flat):
    """
    Parse AlphaPose flat keypoint list [x0, y0, c0, x1, y1, c1, ...].

    Parameters
    ----------
    flat : list or array-like
        Flat list of keypoint values.

    Returns
    -------
    xy : ndarray, shape (N, 2)
    conf : ndarray, shape (N,)
    n_keypoints : int
    """
    n_values = len(flat)
    if n_values == 136 * 3:
        n_keypoints = 136
    elif n_values == 133 * 3:
        n_keypoints = 133
    else:
        raise ValueError(
            f"Expected 136 (408 values) or 133 (399 values) keypoints, got {n_values} values."
        )
    arr = np.array(flat).reshape(-1, 3)
    return arr[:, :2], arr[:, 2], n_keypoints


def _apply_metadata(metadata, fps, width, height):
    if metadata is not None:
        fps = metadata["fps"] if metadata.get("fps") is not None else fps
        width = metadata["width"] if metadata.get("width") is not None else width
        height = metadata["height"] if metadata.get("height") is not None else height
    return fps, width, height


def load_alphapose_wholebody_from_json(input_path: str,
                                       version: float = 0.2,
                                       fps: float = 24,
                                       width: int = 1000,
                                       height: int = 1000,
                                       depth: int = 0) -> Pose:
    """
    Load an AlphaPose WholeBody JSON file into a Pose object.

    Automatically detects whether the file contains 133 or 136 keypoints and
    builds the appropriate components. Use alphapose_133.py directly to
    enforce strict 133-only loading.

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
    """
    frames, metadata = load_alphapose_json(input_path)
    fps, width, height = _apply_metadata(metadata, fps, width, height)

    frames_xy = []
    frames_conf = []
    n_keypoints_detected = None

    for item in frames:
        xy, conf, n_keypoints = parse_keypoints_and_confidence(item["keypoints"])
        n_keypoints_detected = n_keypoints
        frames_xy.append(xy)
        frames_conf.append(conf)

    xy_data = np.stack(frames_xy, axis=0)[:, None, :, :]
    conf_data = np.stack(frames_conf, axis=0)[:, None, :]

    if n_keypoints_detected == 136:
        components = AlphaPose_Components
    else:
        from .alphapose_133 import AlphaPose133_Components
        components = AlphaPose133_Components

    header = PoseHeader(
        version=version,
        dimensions=PoseHeaderDimensions(width=width, height=height, depth=depth),
        components=components,
    )
    body = NumPyPoseBody(fps=fps, data=xy_data, confidence=conf_data)
    return Pose(header, body)

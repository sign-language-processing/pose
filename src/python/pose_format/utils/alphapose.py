import re
import json
import numpy as np
from ..numpy.pose_body import NumPyPoseBody
from ..pose import Pose
from ..pose_header import PoseHeader, PoseHeaderComponent, PoseHeaderDimensions
from pose_format.utils.openpose import hand_colors



def alphapose_components():
    """
    Creates a list of alphapose components.
    
    Returns
    -------
    list of PoseHeaderComponent
        List of holistic components.
    """
    BODY_POINTS = [
        "nose","left_eye","right_eye","left_ear","right_ear",
        "left_shoulder","right_shoulder","left_elbow","right_elbow",
        "left_wrist","right_wrist","left_hip","right_hip",
        "left_knee","right_knee","left_ankle","right_ankle",
        "left_big_toe","left_small_toe","left_heel",
        "right_big_toe","right_small_toe","right_heel",
    ]

    FACE_POINTS = [f"face-{i}" for i in range(68)]
    LEFT_HAND_POINTS = [f"left_hand_{i}" for i in range(21)]
    RIGHT_HAND_POINTS = [f"right_hand_{i}" for i in range(21)]

    def map_limbs(points, limbs):
        index_map = {name: idx for idx, name in enumerate(points)}
        return [
            (index_map[a], index_map[b])
            for (a, b) in limbs
        ]

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
        ("left_hand_0", "left_hand_1"), ("left_hand_1", "left_hand_2"),
        ("left_hand_2", "left_hand_3"), ("left_hand_3", "left_hand_4"),
        ("left_hand_0", "left_hand_5"), ("left_hand_5", "left_hand_6"),
        ("left_hand_6", "left_hand_7"), ("left_hand_7", "left_hand_8"),
        ("left_hand_0", "left_hand_9"), ("left_hand_9", "left_hand_10"),
        ("left_hand_10", "left_hand_11"), ("left_hand_11", "left_hand_12"),
        ("left_hand_0", "left_hand_13"), ("left_hand_13", "left_hand_14"),
        ("left_hand_14", "left_hand_15"), ("left_hand_15", "left_hand_16"),
        ("left_hand_0", "left_hand_17"), ("left_hand_17", "left_hand_18"),
        ("left_hand_18", "left_hand_19"), ("left_hand_19", "left_hand_20"),
    ]

    RIGHT_HAND_LIMBS_NAMES = [
        ("right_hand_0", "right_hand_1"), ("right_hand_1", "right_hand_2"),
        ("right_hand_2", "right_hand_3"), ("right_hand_3", "right_hand_4"),
        ("right_hand_0", "right_hand_5"), ("right_hand_5", "right_hand_6"),
        ("right_hand_6", "right_hand_7"), ("right_hand_7", "right_hand_8"),
        ("right_hand_0", "right_hand_9"), ("right_hand_9", "right_hand_10"),
        ("right_hand_10", "right_hand_11"), ("right_hand_11", "right_hand_12"),
        ("right_hand_0", "right_hand_13"), ("right_hand_13", "right_hand_14"),
        ("right_hand_14", "right_hand_15"), ("right_hand_15", "right_hand_16"),
        ("right_hand_0", "right_hand_17"), ("right_hand_17", "right_hand_18"),
        ("right_hand_18", "right_hand_19"), ("right_hand_19", "right_hand_20"),
    ]

    components = [
        PoseHeaderComponent(
            name="BODY",
            points=BODY_POINTS,
            limbs= map_limbs(BODY_POINTS, BODY_LIMBS_NAMES),  
            colors=[(0,255,0)],
            point_format="XYC"
        ),

        PoseHeaderComponent(
            name="FACE",
            points=FACE_POINTS,
            limbs=[],   # WholeBody face mesh is huge, usually omitted
            colors=[(255,255,255)],
            point_format="XYC"
        ),

        PoseHeaderComponent(
            name="LEFT_HAND",
            points=LEFT_HAND_POINTS,
            limbs= map_limbs(LEFT_HAND_POINTS, LEFT_HAND_LIMBS_NAMES), 
            colors=[(0,255,0)],
            point_format="XYC"
        ),

        PoseHeaderComponent(
            name="RIGHT_HAND",
            points=RIGHT_HAND_POINTS,
            limbs= map_limbs(RIGHT_HAND_POINTS, RIGHT_HAND_LIMBS_NAMES),
            colors=[(255,128,0)],
            point_format="XYC"
        ),
    ]
    return components

def load_alphapose_json(json_path):
    """
    Load AlphaPose results in either:
    
    FORMAT A (original):
        [
            {"image_id": "0.jpg", "keypoints": [x_0, y_0, c_0, x_1, y_1, c_1, ...], "other keys not used"},
            {"image_id": "1.jpg", "keypoints": [...], ...},
            ...
        ]

    FORMAT B (extended):
        {
            "frames": [... same as above ...],
            "metadata": {
                "fps": float,
                "width": int,
                "height": int
            }
        }

    Returns
    -------
    data : list
        Sorted list of frame detections.
    meta : dict or None
        Metadata if present, else None.
    """
    with open(json_path, "r") as f:
        raw = json.load(f)

    # -----------------------
    # Detect FORMAT B (dict)
    # -----------------------
    if isinstance(raw, dict) and "frames" in raw:
        frames = raw["frames"]

        # Extract metadata safely
        meta = {
            "fps": raw.get("metadata", {}).get("fps", None),
            "width": raw.get("metadata", {}).get("width", None),
            "height": raw.get("metadata", {}).get("height", None),
        }
    else:
        # FORMAT A (list)
        frames = raw
        meta = None

    # -----------------------
    # Sorting function
    # -----------------------
    def extract_frame_number(item):
        """
        Extract numeric part from "image_id".
        Example: "frame_0012.jpg" → 12
        """
        matches = re.findall(r"\d+", item["image_id"])
        return int(matches[0]) if matches else -1  # fallback if no digits

    # Sort frames numerically
    frames = sorted(frames, key=extract_frame_number)

    return frames, meta

def load_alphapose_wholebody_from_json(input_path: str, 
                  version: float = 0.2,
                  fps: float = 24,
                  width=1000,
                  height=1000,
                  depth=0) -> Pose:
    """
    Loads alphapose_wholebody pose data

    Parameters
    ----------
    video_path : string
        Path to input video file.

    Returns
    -------
    Pose
        Loaded pose data with header and body 
    """
    print("Loading pose with alphapose_wholebody...")

    # Load frames + optional metadata
    frames, metadata = load_alphapose_json(input_path)

    # Override fps/width/height ONLY if metadata exists
    if metadata is not None:
        if metadata.get("fps") is not None:
            fps = metadata["fps"]
        if metadata.get("width") is not None:
            width = metadata["width"]
        if metadata.get("height") is not None:
            height = metadata["height"]

    frames_xy = []
    frames_conf = []

    # Parse and reorder all frames
    for item in frames:
        xy, conf = parse_keypoints_and_confidence(item["keypoints"])
        xy_ord, conf_ord = reorder_133_kpts(xy, conf)

        frames_xy.append(xy_ord)
        frames_conf.append(conf_ord)

    # Convert to arrays
    xy_data = np.stack(frames_xy, axis=0)         # (num_frames, num_keypoints, 2)
    conf_data = np.stack(frames_conf, axis=0)     # (num_frames, num_keypoints)

    # Add people dimension:
    xy_data = xy_data[:, None, :, :]            # (num_frames, people, num_keypoints, 2) with people = 1
    conf_data = conf_data[:, None, :]           # (num_frames, people, num_keypoints) with people = 1

    # Build header
    header: PoseHeader = PoseHeader(version=version,
                                    dimensions=PoseHeaderDimensions(width=width, height=height, depth=depth),
                                    components=alphapose_components())

    # Build body
    body: NumPyPoseBody = NumPyPoseBody(fps=fps, data=xy_data, confidence=conf_data)

    return Pose(header, body)

def parse_keypoints_and_confidence(flat):
    """
    AlphaPose stores keypoints as:
        [x0, y0, c0, x1, y1, c1, ...]
    Expected total length for 133 keypoints:
        133 * 3 = 399 values

    Returns:
        xy:   (133, 2)
        conf: (133,)
    """
    assert len(flat) == 133 * 3, \
        f"ERROR: Expected 133 keypoints (399 values), but got {len(flat)} values. " \
        f"This converter only supports AlphaPose WholeBody-133."

    arr = np.array(flat).reshape(-1, 3)
    xy = arr[:, :2]
    conf = arr[:, 2]
    return xy, conf


def reorder_133_kpts(xy, conf):
    """
    Reorder XY and confidence to BODY + FACE + L_HAND + R_HAND.
    AlphaPose 133 indexing:
    - BODY: 0–22
    - FACE: 23–90
    - LH:   91–111
    - RH:   112–132
    """
    body = xy[0:23]
    face = xy[23:23+68]
    lh = xy[91:91+21]
    rh = xy[112:112+21]

    xy_reordered = np.concatenate([body, face, lh, rh], axis=0)

    # Apply same order to confidence
    conf_reordered = np.concatenate([
        conf[0:23],
        conf[23:23+68],
        conf[91:91+21],
        conf[112:112+21],
    ], axis=0)

    return xy_reordered, conf_reordered


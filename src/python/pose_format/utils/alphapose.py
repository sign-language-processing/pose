import re
import json
import numpy as np
from ..numpy.pose_body import NumPyPoseBody
from ..pose import Pose
from ..pose_header import PoseHeader, PoseHeaderComponent, PoseHeaderDimensions
from pose_format.utils.openpose import hand_colors

BODY_POINTS_133 = [
    "nose","left_eye","right_eye","left_ear","right_ear",
    "left_shoulder","right_shoulder","left_elbow","right_elbow",
    "left_wrist","right_wrist","left_hip","right_hip",
    "left_knee","right_knee","left_ankle","right_ankle",
    "left_big_toe","left_small_toe","left_heel",
    "right_big_toe","right_small_toe","right_heel",
]

BODY_LIMBS_NAMES_133 = [
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

LEFT_HAND_LIMBS_NAMES_133 = [
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

RIGHT_HAND_LIMBS_NAMES_133 = [
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

FACE_POINTS_133 = [f"face-{i}" for i in range(68)]
LEFT_HAND_POINTS_133 = [f"left_hand_{i}" for i in range(21)]
RIGHT_HAND_POINTS_133 = [f"right_hand_{i}" for i in range(21)]
GENERAL_HAND_POINTS_133 = [f"hand_{i}" for i in range(21)]


BODY_POINTS_136 = [
    # Head (basic)
    'nose',            # 0
    'left_eye',        # 1
    'right_eye',       # 2
    'left_ear',        # 3
    'right_ear',       # 4

    # Upper body
    'left_shoulder',   # 5
    'right_shoulder',  # 6
    'left_elbow',      # 7
    'right_elbow',     # 8
    'left_wrist',      # 9
    'right_wrist',     # 10

    # Lower body
    'left_hip',        # 11
    'right_hip',       # 12
    'left_knee',       # 13
    'right_knee',      # 14
    'left_ankle',      # 15
    'right_ankle',     # 16

    # Extra body points
    'head_top',        # 17
    'neck',            # 18
    'pelvis',          # 19

    # Feet
    'left_big_toe',    # 20
    'right_big_toe',   # 21
    'left_small_toe',  # 22
    'right_small_toe', # 23
    'left_heel',       # 24
    'right_heel',      # 25
]
# Face (68 points: 26–93)
FACE_POINTS_136 = [f"face-{i}" for i in range(68)]

# Left hand (21 points: 94–114)
LEFT_HAND_POINTS_136 = [f"left_hand_{i}" for i in range(21)]

# Right hand (21 points: 115–135)
RIGHT_HAND_POINTS_136 = [f"right_hand_{i}" for i in range(21)]



GENERAL_HAND_POINTS_136 = [f"hand_{i}" for i in range(21)]
alphapose_136_keypoints = BODY_POINTS_136 + FACE_POINTS_136 + LEFT_HAND_POINTS_136 + RIGHT_HAND_POINTS_136


BODY_LIMBS_NAMES_136 = [
    # Head
    ("nose", "left_eye"),
    ("nose", "right_eye"),
    ("left_eye", "left_ear"),
    ("right_eye", "right_ear"),

    # Head ↔ torso
    ("head_top", "neck"),

    # Shoulders ↔ neck
    ("left_shoulder", "neck"),
    ("right_shoulder", "neck"),

    # Arms
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),

    # Torso
    ("neck", "pelvis"),
    ("pelvis", "left_hip"),
    ("pelvis", "right_hip"),

    # Legs
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),

    # Feet (left)
    ("left_ankle", "left_heel"),
    ("left_heel", "left_big_toe"),
    ("left_heel", "left_small_toe"),

    # Feet (right)
    ("right_ankle", "right_heel"),
    ("right_heel", "right_big_toe"),
    ("right_heel", "right_small_toe"),
]

LEFT_HAND_LIMBS_NAMES_136 = [
    # Thumb
    ("left_hand_0", "left_hand_1"),
    ("left_hand_1", "left_hand_2"),
    ("left_hand_2", "left_hand_3"),
    ("left_hand_3", "left_hand_4"),

    # Index
    ("left_hand_0", "left_hand_5"),
    ("left_hand_5", "left_hand_6"),
    ("left_hand_6", "left_hand_7"),
    ("left_hand_7", "left_hand_8"),

    # Middle
    ("left_hand_0", "left_hand_9"),
    ("left_hand_9", "left_hand_10"),
    ("left_hand_10", "left_hand_11"),
    ("left_hand_11", "left_hand_12"),

    # Ring
    ("left_hand_0", "left_hand_13"),
    ("left_hand_13", "left_hand_14"),
    ("left_hand_14", "left_hand_15"),
    ("left_hand_15", "left_hand_16"),

    # Pinky
    ("left_hand_0", "left_hand_17"),
    ("left_hand_17", "left_hand_18"),
    ("left_hand_18", "left_hand_19"),
    ("left_hand_19", "left_hand_20"),
]
RIGHT_HAND_LIMBS_NAMES_136 = [
    # Thumb
    ("right_hand_0", "right_hand_1"),
    ("right_hand_1", "right_hand_2"),
    ("right_hand_2", "right_hand_3"),
    ("right_hand_3", "right_hand_4"),

    # Index
    ("right_hand_0", "right_hand_5"),
    ("right_hand_5", "right_hand_6"),
    ("right_hand_6", "right_hand_7"),
    ("right_hand_7", "right_hand_8"),

    # Middle
    ("right_hand_0", "right_hand_9"),
    ("right_hand_9", "right_hand_10"),
    ("right_hand_10", "right_hand_11"),
    ("right_hand_11", "right_hand_12"),

    # Ring
    ("right_hand_0", "right_hand_13"),
    ("right_hand_13", "right_hand_14"),
    ("right_hand_14", "right_hand_15"),
    ("right_hand_15", "right_hand_16"),

    # Pinky
    ("right_hand_0", "right_hand_17"),
    ("right_hand_17", "right_hand_18"),
    ("right_hand_18", "right_hand_19"),
    ("right_hand_19", "right_hand_20"),
]

FACE_LIMBS_NAMES_136 = [
    # Jaw / contour
    ("face-0", "face-1"),
    ("face-1", "face-2"),
    ("face-2", "face-3"),
    ("face-3", "face-4"),
    ("face-4", "face-5"),
    ("face-5", "face-6"),
    ("face-6", "face-7"),
    ("face-7", "face-8"),
    ("face-8", "face-9"),
    ("face-9", "face-10"),
    ("face-10", "face-11"),
    ("face-11", "face-12"),

    # Jaw → upper face
    ("face-12", "face-13"),
    ("face-13", "face-14"),
    ("face-14", "face-15"),
    ("face-15", "face-16"),

    # Left eyebrow
    ("face-17", "face-18"),
    ("face-18", "face-19"),
    ("face-19", "face-20"),
    ("face-20", "face-21"),

    # Right eyebrow
    ("face-22", "face-23"),
    ("face-23", "face-24"),
    ("face-24", "face-25"),
    ("face-25", "face-26"),

    # Nose bridge
    ("face-27", "face-28"),
    ("face-28", "face-29"),
    ("face-29", "face-30"),

    # Nose bottom
    ("face-31", "face-32"),
    ("face-32", "face-33"),
    ("face-33", "face-34"),
    ("face-34", "face-35"),

    # Left eye
    ("face-36", "face-37"),
    ("face-37", "face-38"),
    ("face-38", "face-39"),
    ("face-39", "face-40"),
    ("face-40", "face-41"),

    # Right eye
    ("face-42", "face-43"),
    ("face-43", "face-44"),
    ("face-44", "face-45"),
    ("face-45", "face-46"),
    ("face-46", "face-47"),

    # Outer mouth
    ("face-48", "face-49"),
    ("face-49", "face-50"),
    ("face-50", "face-51"),
    ("face-51", "face-52"),
    ("face-52", "face-53"),
    ("face-53", "face-54"),
    ("face-54", "face-55"),
    ("face-55", "face-56"),

    # Inner mouth
    ("face-56", "face-57"),
    ("face-57", "face-58"),
    ("face-58", "face-59"),
    ("face-59", "face-60"),
    ("face-60", "face-61"),
    ("face-61", "face-62"),
    ("face-62", "face-63"),
    ("face-63", "face-64"),
    ("face-64", "face-65"),
    ("face-65", "face-66"),
    ("face-66", "face-67"),
]

def get_alphapose_133_components():
    """
    Creates a list of alphapose components.
    
    Returns
    -------
    list of PoseHeaderComponent
        List of holistic components.
    """

    def map_limbs(points, limbs):
        index_map = {name: idx for idx, name in enumerate(points)}
        return [
            (index_map[a], index_map[b])
            for (a, b) in limbs
        ]

    components = [
        PoseHeaderComponent(
            name="BODY_133",
            points=BODY_POINTS_133,
            limbs= map_limbs(BODY_POINTS_133, BODY_LIMBS_NAMES_133),  
            colors=[(0,255,0)],
            point_format="XYC"
        ),

        PoseHeaderComponent(
            name="FACE_133",
            points=FACE_POINTS_133,
            limbs=[],   # WholeBody face mesh is huge, usually omitted
            colors=[(255,255,255)],
            point_format="XYC"
        ),

        PoseHeaderComponent(
            name="LEFT_HAND_133",
            points=GENERAL_HAND_POINTS_133,
            limbs= map_limbs(LEFT_HAND_POINTS_133, LEFT_HAND_LIMBS_NAMES_133), 
            colors=[(0,255,0)],
            point_format="XYC"
        ),

        PoseHeaderComponent(
            name="RIGHT_HAND_133",
            points=GENERAL_HAND_POINTS_133,
            limbs= map_limbs(RIGHT_HAND_POINTS_133, RIGHT_HAND_LIMBS_NAMES_133),
            colors=[(255,128,0)],
            point_format="XYC"
        ),
    ]
    return components


def get_alphapose_136_components():
    """
    Creates a list of alphapose components.
    
    Returns
    -------
    list of PoseHeaderComponent
        List of holistic components.
    """

    def map_limbs(points, limbs):
        index_map = {name: idx for idx, name in enumerate(points)}
        return [
            (index_map[a], index_map[b])
            for (a, b) in limbs
        ]

    components = [
        PoseHeaderComponent(
            name="BODY_136",
            points=BODY_POINTS_136,
            limbs= map_limbs(BODY_POINTS_136, BODY_LIMBS_NAMES_136),  
            colors=[(0,255,0)],
            point_format="XYC"
        ),

        PoseHeaderComponent(
            name="FACE_136",
            points=FACE_POINTS_136,
            limbs=map_limbs(FACE_POINTS_136, FACE_LIMBS_NAMES_136),   # WholeBody face mesh is huge, usually omitted
            colors=[(255,255,255)],
            point_format="XYC"
        ),

        PoseHeaderComponent(
            name="LEFT_HAND_136",
            points=GENERAL_HAND_POINTS_136,
            limbs= map_limbs(LEFT_HAND_POINTS_136, LEFT_HAND_LIMBS_NAMES_136), 
            colors=[(0,255,0)],
            point_format="XYC"
        ),

        PoseHeaderComponent(
            name="RIGHT_HAND_136",
            points=GENERAL_HAND_POINTS_136,
            limbs= map_limbs(RIGHT_HAND_POINTS_136, RIGHT_HAND_LIMBS_NAMES_136),
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
        xy, conf, n_keypoints = parse_keypoints_and_confidence(item["keypoints"])
        if n_keypoints == 133:
            xy_ord, conf_ord = reorder_133_kpts(xy, conf)
        else: 
            xy_ord, conf_ord = reorder_136_kpts(xy, conf)

        frames_xy.append(xy_ord)
        frames_conf.append(conf_ord)

    # Convert to arrays
    xy_data = np.stack(frames_xy, axis=0)         # (num_frames, num_keypoints, 2)
    conf_data = np.stack(frames_conf, axis=0)     # (num_frames, num_keypoints)

    # Add people dimension:
    xy_data = xy_data[:, None, :, :]            # (num_frames, people, num_keypoints, 2) with people = 1
    conf_data = conf_data[:, None, :]           # (num_frames, people, num_keypoints) with people = 1

    print(f"Detected alphapose with {n_keypoints} keypoints.")
    alphapose_components = get_alphapose_136_components() if n_keypoints==136 else get_alphapose_133_components()
    # Build header
    header: PoseHeader = PoseHeader(version=version,
                                    dimensions=PoseHeaderDimensions(width=width, height=height, depth=depth),
                                    components=alphapose_components)

    # Build body
    body: NumPyPoseBody = NumPyPoseBody(fps=fps, data=xy_data, confidence=conf_data)

    return Pose(header, body)

def parse_keypoints_and_confidence(flat):
    """
    AlphaPose stores keypoints as:
        [x0, y0, c0, x1, y1, c1, ...]
    Expected total length for 133 keypoints:
        133 * 3 = 399 values

    Expected total length for 136 keypoints:
        136 * 3 = 408 values

    Returns:
        xy:   (133, 2)
        conf: (133,)
    """
    # assert len(flat) == 133 * 3, \
    #     f"ERROR: Expected 133 keypoints (399 values), but got {len(flat)} values. " \
    #     f"This converter only supports AlphaPose WholeBody-133."
    if len(flat) == 133 * 3:
        n_keypoints = 133
    elif len(flat) == 136 * 3:
        n_keypoints = 136
    else:
        n_keypoints = None
        assert len(flat) == 133 * 3, \
            f"ERROR: Expected 133 or 136 keypoints, but got {len(flat)} values. " \
            f"This converter only supports AlphaPose WholeBody-133 or WholeBody-136."

    arr = np.array(flat).reshape(-1, 3)
    xy = arr[:, :2]
    conf = arr[:, 2]
    return xy, conf, n_keypoints


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

def reorder_136_kpts(xy, conf):
    """
    Reorder XY and confidence to BODY + FACE + L_HAND + R_HAND.

    AlphaPose 136 indexing:
    - BODY: 0–25   (26)
    - FACE: 26–93  (68)
    - LH:   94–114 (21)
    - RH:   115–135 (21)
    """
    body = xy[0:26]
    face = xy[26:26+68]
    lh = xy[94:94+21]
    rh = xy[115:115+21]

    xy_reordered = np.concatenate([body, face, lh, rh], axis=0)

    # Apply same order to confidence
    conf_reordered = np.concatenate([
        conf[0:26],
        conf[26:26+68],
        conf[94:94+21],
        conf[115:115+21],
    ], axis=0)

    return xy_reordered, conf_reordered
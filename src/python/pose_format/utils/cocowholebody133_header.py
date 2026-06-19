from ..pose_header import PoseHeaderComponent

# --- Canonical COCO Wholebody 133 point lists ---

BODY_POINTS = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
    "left_big_toe", "left_small_toe", "left_heel",
    "right_big_toe", "right_small_toe", "right_heel",
]

# Face uses dash separator ("face-N") matching the COCO Wholebody / MMPose convention.
FACE_POINTS = [f"face-{i}" for i in range(68)]

# Single hand point list shared by both LEFT_HAND and RIGHT_HAND components.
# The component name (LEFT_HAND / RIGHT_HAND) carries the side; point names are generic.
# This mirrors the AlphaPose convention and allows normalize_hands_3d to work correctly.
HAND_POINTS = [f"hand_{i}" for i in range(21)]


# --- Limb connectivity ---

def _map_limbs(points, limb_names):
    index_map = {name: idx for idx, name in enumerate(points)}
    return [(index_map[a], index_map[b]) for (a, b) in limb_names]


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

# 68-point face connectivity (jaw, brows, nose bridge, eyes, lips).
FACE_LIMBS_NAMES = [
    ("face-0", "face-1"), ("face-1", "face-2"), ("face-2", "face-3"), ("face-3", "face-4"),
    ("face-4", "face-5"), ("face-5", "face-6"), ("face-6", "face-7"), ("face-7", "face-8"),
    ("face-8", "face-9"), ("face-9", "face-10"), ("face-10", "face-11"), ("face-11", "face-12"),
    ("face-12", "face-13"), ("face-13", "face-14"), ("face-14", "face-15"), ("face-15", "face-16"),
    ("face-17", "face-18"), ("face-18", "face-19"), ("face-19", "face-20"), ("face-20", "face-21"),
    ("face-22", "face-23"), ("face-23", "face-24"), ("face-24", "face-25"), ("face-25", "face-26"),
    ("face-27", "face-28"), ("face-28", "face-29"), ("face-29", "face-30"),
    ("face-31", "face-32"), ("face-32", "face-33"), ("face-33", "face-34"), ("face-34", "face-35"),
    ("face-36", "face-37"), ("face-37", "face-38"), ("face-38", "face-39"),
    ("face-39", "face-40"), ("face-40", "face-41"),
    ("face-42", "face-43"), ("face-43", "face-44"), ("face-44", "face-45"),
    ("face-45", "face-46"), ("face-46", "face-47"),
    ("face-48", "face-49"), ("face-49", "face-50"), ("face-50", "face-51"), ("face-51", "face-52"),
    ("face-52", "face-53"), ("face-53", "face-54"), ("face-54", "face-55"), ("face-55", "face-56"),
    ("face-56", "face-57"), ("face-57", "face-58"), ("face-58", "face-59"), ("face-59", "face-60"),
    ("face-60", "face-61"), ("face-61", "face-62"), ("face-62", "face-63"), ("face-63", "face-64"),
    ("face-64", "face-65"), ("face-65", "face-66"), ("face-66", "face-67"),
]

HAND_LIMBS_NAMES = [
    ("hand_0", "hand_1"), ("hand_1", "hand_2"), ("hand_2", "hand_3"), ("hand_3", "hand_4"),
    ("hand_0", "hand_5"), ("hand_5", "hand_6"), ("hand_6", "hand_7"), ("hand_7", "hand_8"),
    ("hand_0", "hand_9"), ("hand_9", "hand_10"), ("hand_10", "hand_11"), ("hand_11", "hand_12"),
    ("hand_0", "hand_13"), ("hand_13", "hand_14"), ("hand_14", "hand_15"), ("hand_15", "hand_16"),
    ("hand_0", "hand_17"), ("hand_17", "hand_18"), ("hand_18", "hand_19"), ("hand_19", "hand_20"),
]

# Pre-computed limb index pairs — import these instead of recomputing.
BODY_LIMBS = _map_limbs(BODY_POINTS, BODY_LIMBS_NAMES)
FACE_LIMBS = _map_limbs(FACE_POINTS, FACE_LIMBS_NAMES)
HAND_LIMBS = _map_limbs(HAND_POINTS, HAND_LIMBS_NAMES)


def cocowholebody_components():
    """
    Creates the four PoseHeaderComponent objects for COCO Wholebody 133.

    Returns
    -------
    list of PoseHeaderComponent
        [BODY (23 pts), FACE (68 pts), LEFT_HAND (21 pts), RIGHT_HAND (21 pts)]
    """
    return [
        PoseHeaderComponent(
            name="BODY",
            points=BODY_POINTS,
            limbs=BODY_LIMBS,
            colors=[(0, 255, 0)],
            point_format="XYC"
        ),
        PoseHeaderComponent(
            name="FACE",
            points=FACE_POINTS,
            limbs=FACE_LIMBS,
            colors=[(255, 255, 255)],
            point_format="XYC"
        ),
        PoseHeaderComponent(
            name="LEFT_HAND",
            points=HAND_POINTS,
            limbs=HAND_LIMBS,
            colors=[(0, 255, 0)],
            point_format="XYC"
        ),
        PoseHeaderComponent(
            name="RIGHT_HAND",
            points=HAND_POINTS,
            limbs=HAND_LIMBS,
            colors=[(255, 128, 0)],
            point_format="XYC"
        ),
    ]

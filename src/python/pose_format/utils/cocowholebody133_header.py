from ..pose_header import PoseHeaderComponent


def cocowholebody_components():
    """
    Creates a list of COCO-Wholebody 133 components.
    
    Returns
    -------
    list of PoseHeaderComponent
        List of COCO-Wholebody 133 components.
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
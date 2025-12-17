import openpifpaf
import PIL
import numpy as np

from ..numpy.pose_body import NumPyPoseBody
from ..pose import Pose
from ..pose_header import PoseHeader, PoseHeaderComponent, PoseHeaderDimensions

def openpifpaf_components(): # mmpose and openpifpaf both use COCO WholeBody, so they can use the same header
    """
    Creates a list of mmposewholebody components.
    
    Returns
    -------
    list of PoseHeaderComponent
        List of MMPoseWholeBody components.
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

def process_openpifpaf(frames: list, fps: float) -> NumPyPoseBody:
    """
    Process frames to extract openpifpaf pose data.

    Parameters
    ----------
    input_path : string
        Path to input video file.
    output_path : string
        Path to output pose file.
    fps : float
        Frames per second of the video.
    use_cpu : bool
        Whether to use CPU for processing.

    Returns
    -------
    NumPyPoseBody
        Processed pose body data.
    """

    print("Processing video with OpenPifPaf...")
    predictor = openpifpaf.Predictor(checkpoint='shufflenetv2k30-wholebody')

    frames_data = []
    frames_conf = []

    for frame in frames:

        pil_im = PIL.Image.fromarray(frame).convert('RGB')
        predictions, gt_anns, image_meta = predictor.pil_image(pil_im)

        people_xy = []
        people_conf = []

        for ann in predictions:
            # ann.data shape: (133, 3) -> x, y, confidence
            keypoints = ann.data[:, :2]
            confidence = ann.data[:, 2]

            people_xy.append(keypoints)
            people_conf.append(confidence)

        frames_data.append(people_xy)
        frames_conf.append(people_conf)

    data_array = np.array(frames_data, dtype=np.float32)
    conf_array = np.array(frames_conf, dtype=np.float32)

    return NumPyPoseBody(
        fps=fps,
        data=data_array,
        confidence=conf_array
    )

def estimate_and_load_openpifpaf(frames: list,
                  fps: float = 24,
                  width=1000,
                  height=1000) -> Pose:
    """
    Loads openpifpaf pose data

    Parameters
    ----------
    video_path : string
        Path to input video file.

    Returns
    -------
    Pose
        Loaded pose data with header and body 
    """
    print("Loading pose with OpenPifPaf...")

    dimensions = PoseHeaderDimensions(width=width, height=height)

    header: PoseHeader = PoseHeader(version=0.1,
                                    dimensions=dimensions,
                                    components=openpifpaf_components())
    body: NumPyPoseBody = process_openpifpaf(frames, fps)

    print(f'Pose loaded. {Pose(header, body)}')
    return Pose(header, body)


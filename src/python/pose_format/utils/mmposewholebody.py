import numpy as np
from tqdm import tqdm
from ..numpy.pose_body import NumPyPoseBody
from ..pose import Pose
from ..pose_header import PoseHeader, PoseHeaderComponent, PoseHeaderDimensions
import mmcv, mmengine, mmdet, mmpose
from mmpose.apis import MMPoseInferencer

def mmposewholebody_components():
    """
    Creates a list of mmposewholebody components.
    
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

def load_mmposewholebody(input_path: str, 
                  output_path: str,
                  use_cpu: bool,
                  fps: float = 24,
                  width=1000,
                  height=1000,
                  depth=0) -> Pose:
    """
    Loads mmposewholebody pose data

    Parameters
    ----------
    video_path : string
        Path to input video file.

    Returns
    -------
    Pose
        Loaded pose data with header and body 
    """
    print("Loading pose with mmposewholebody...")

    dimensions = PoseHeaderDimensions(width=width, height=height, depth=depth)

    header: PoseHeader = PoseHeader(version=0.1,
                                    dimensions=dimensions,
                                    components=mmposewholebody_components())
    body: NumPyPoseBody = process_mmposewholebody(input_path, output_path, fps, use_cpu)

    print(f'Pose loaded. {Pose(header, body)}')
    return Pose(header, body)

def process_mmposewholebody(input_path, output_path, fps, use_cpu) -> NumPyPoseBody:
    """
    Process frames to extract mmposewholebody pose data.

    Parameters
    ----------

    Returns
    -------
    NumPyPoseBody
        Processed pose body data.
    """

    # instantiate the inferencer using the model alias
    if use_cpu:
        print("Warning: compiling MMPoseWholeBody on CPU, this may be slow.")
        inferencer = MMPoseInferencer('wholebody', device='cpu')
    else:
        inferencer = MMPoseInferencer('wholebody')

    visualization_path = output_path
    result_generator = inferencer(
        input_path,
        show=False, 
        save_vis=False, 
        return_vis=True,
        save_out_video=True,
        out_dir=visualization_path)  
    print("MMPoseWholeBody processing complete.")

    frames_data = []
    frames_conf = []


    for result in result_generator: 
        predictions_by_frame = result['predictions'] # each result represents one frame
        person_keypoints = []
        person_conf = []

        for predictions_by_person in predictions_by_frame:
            person_keypoints.append(predictions_by_person[0]['keypoints'])  # (num_keypoints, 2) array
            person_conf.append(predictions_by_person[0]['keypoint_scores'])  # (num_keypoints) array
            # TODO use this: https://github.com/open-mmlab/mmpose/blob/main/configs/_base_/datasets/coco_wholebody.py
        
        frames_data.append(person_keypoints)
        frames_conf.append(person_conf)

    data_array = np.array(frames_data)        # (frames, people, points, 2)
    conf_array = np.array(frames_conf)         # (frames, people, points)

    return NumPyPoseBody(fps=fps, data=data_array, confidence=conf_array)


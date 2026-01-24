# based on code from https://github.com/open-mmlab/mmpose/blob/main/configs/_base_/datasets/coco_wholebody.py

import numpy as np
from tqdm import tqdm
from ..numpy.pose_body import NumPyPoseBody
from ..pose import Pose
from ..pose_header import PoseHeader, PoseHeaderComponent, PoseHeaderDimensions
import mmcv, mmengine, mmdet, mmpose
from mmpose.apis import MMPoseInferencer
from pose_format.utils.cocowholebody133_header import cocowholebody_components

def mmposewholebody_components():
    return cocowholebody_components()

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
        return_vis=False,
        save_out_video=False)
            #out_dir=visualization_path  #TODO: make saving of visualizations and json to disk toggleable
    print("MMPoseWholeBody pose estimation complete. Beginning conversion to .pose format...")

    frames_data = []
    frames_conf = []


    # Convert to NumPyPoseBody format
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


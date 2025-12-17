import openpifpaf
import PIL
import numpy as np
import torch 

from ..numpy.pose_body import NumPyPoseBody
from ..pose import Pose
from ..pose_header import PoseHeader, PoseHeaderDimensions
from pose_format.utils.cocowholebody133_header import cocowholebody_components

def openpifpaf_components(): 
    return cocowholebody_components()

def process_openpifpaf(frames: list, fps: float, use_cpu: bool) -> NumPyPoseBody:
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
    
    device = torch.device('cpu' if use_cpu or not torch.cuda.is_available() else 'cuda')
    predictor = openpifpaf.Predictor(checkpoint='shufflenetv2k30-wholebody')
    predictor.model.to(device)
    if device.type == 'cpu':
        print("Warning: using CPU for OpenPifPaf processing may be very slow.")

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
                  use_cpu: bool = False,
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
    body: NumPyPoseBody = process_openpifpaf(frames, fps, use_cpu)

    return Pose(header, body)


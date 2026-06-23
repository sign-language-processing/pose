import numpy as np
import numpy.ma as ma

try:
    from mmpose.apis import MMPoseInferencer
except ImportError:
    raise ImportError(
        "Please install MMPose and its dependencies. For GPU support, mmcv must be installed\n"
        "from the OpenMMLab CUDA-specific index (see https://mmcv.readthedocs.io/en/latest/get_started/installation.html).\n"
        "The remaining packages: pip install 'mmpose>=1.3.2' 'mmengine>=0.10.7' 'mmdet>=3.3.0'"
    )

from ..numpy.pose_body import NumPyPoseBody
from ..pose import Pose
from ..pose_header import PoseHeader, PoseHeaderDimensions
from .cocowholebody133_header import cocowholebody_components

NUM_KEYPOINTS = 133


def load_mmposewholebody(input_path: str,
                         version: float = 0.2,
                         fps: float = 24,
                         width: int = 1000,
                         height: int = 1000,
                         depth: int = 0) -> Pose:
    """
    Run MMPose wholebody inference on a video and return a Pose object.

    Parameters
    ----------
    input_path : str
        Path to the input video file.
    version : float
        Pose format version written to the header.
    fps : float
        Frames per second stored in the pose body.
    width : int
        Frame width in pixels stored in the header dimensions.
    height : int
        Frame height in pixels stored in the header dimensions.
    depth : int
        Depth dimension size (0 for 2D poses).

    Returns
    -------
    Pose
        Loaded pose with header and body.
    """
    header = PoseHeader(
        version=version,
        dimensions=PoseHeaderDimensions(width=width, height=height, depth=depth),
        components=cocowholebody_components(),
    )
    body = _process_video(input_path, fps)
    return Pose(header, body)


def _process_video(input_path: str, fps: float, use_cpu: bool = False) -> NumPyPoseBody:
    """
    Run MMPose wholebody inference and convert frame results to NumPyPoseBody.

    Parameters
    ----------
    input_path : str
        Path to the input video file.
    fps : float
        Frames per second to store in the pose body.
    use_cpu : bool
        If True, run inference on CPU (slow; useful when no GPU is available).

    Returns
    -------
    NumPyPoseBody
    """
    device = 'cpu' if use_cpu else None
    inferencer_kwargs = {'wholebody': True}
    if device is not None:
        inferencer_kwargs['device'] = device

    inferencer = MMPoseInferencer('wholebody', **({'device': device} if device else {}))
    result_generator = inferencer(input_path, show=False, return_vis=False)

    frames_xy = []
    frames_conf = []
    frames_mask = []  # True = valid, False = masked out (no detection)

    for result in result_generator:
        predictions_by_frame = result['predictions']  # list of per-person dicts for this frame

        if len(predictions_by_frame) == 0 or len(predictions_by_frame[0]) == 0:
            # No person detected in this frame. Insert a zeroed, fully-masked row so
            # the frame count stays aligned with the video. Callers can distinguish
            # "no detection" from a real zero-coordinate keypoint via the mask.
            frames_xy.append(np.zeros((1, NUM_KEYPOINTS, 2), dtype=np.float32))
            frames_conf.append(np.zeros((1, NUM_KEYPOINTS), dtype=np.float32))
            frames_mask.append(True)   # True = mask this frame entirely
        else:
            person = predictions_by_frame[0][0]
            frames_xy.append(np.array(person['keypoints'], dtype=np.float32)[None])    # (1, 133, 2)
            frames_conf.append(np.array(person['keypoint_scores'], dtype=np.float32)[None])  # (1, 133)
            frames_mask.append(False)  # False = keep (not masked)

    xy_data = np.concatenate(frames_xy, axis=0)[:, None, :, :]    # (T, 1, 133, 2)
    conf_data = np.concatenate(frames_conf, axis=0)[:, None, :]   # (T, 1, 133)

    # Build the masked array: mask=True on empty frames so downstream code
    # can treat them as missing rather than as detected-at-origin.
    mask = np.array(frames_mask)                                    # (T,)
    xy_mask = mask[:, None, None, None] * np.ones_like(xy_data, dtype=bool)
    masked_xy = ma.array(xy_data, mask=xy_mask)

    return NumPyPoseBody(fps=fps, data=masked_xy, confidence=conf_data)

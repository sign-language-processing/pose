import numpy as np
from tqdm import tqdm

from ..numpy.pose_body import NumPyPoseBody
from ..pose import Pose
from ..pose_header import PoseHeader, PoseHeaderComponent, PoseHeaderDimensions
from .openpose import hand_colors, load_frames_directory_dict

try:
    import mediapipe as mp
    from mediapipe.python.solutions.face_mesh_connections import FACEMESH_IRISES
except ImportError:
    raise ImportError("Please install mediapipe with: pip install mediapipe")

mp_holistic = mp.solutions.holistic

FACEMESH_CONTOURS_POINTS = [
    str(p) for p in sorted(set([p for p_tup in list(mp_holistic.FACEMESH_CONTOURS) for p in p_tup]))
]

BODY_POINTS = mp_holistic.PoseLandmark._member_names_
BODY_LIMBS = [(int(a), int(b)) for a, b in mp_holistic.POSE_CONNECTIONS]

HAND_POINTS = mp_holistic.HandLandmark._member_names_
HAND_LIMBS = [(int(a), int(b)) for a, b in mp_holistic.HAND_CONNECTIONS]

FACE_POINTS_NUM = lambda additional_points=0: additional_points + 468
FACE_POINTS_NUM.__doc__ = """
Gets total number of face points and additional points.

Parameters
----------
additional_points : int, optional
    number of additional points to be added. The defaults is 0.

Returns
-------
int
    total number of face points.
"""
FACE_POINTS = lambda additional_points=0: [str(i) for i in range(FACE_POINTS_NUM(additional_points))]
FACE_POINTS.__doc__ = """
Makes a list of string representations of face points indexes up to total face points number

Parameters
----------
additional_points : int, optional
    number of additional points to be considered. Defaults to 0

Returns
-------
list[str]
    List of strings of face point indices.
"""

FACE_LIMBS = [(int(a), int(b)) for a, b in mp_holistic.FACEMESH_TESSELATION]
FACE_IRISES = [(int(a), int(b)) for a, b in FACEMESH_IRISES]

FLIPPED_BODY_POINTS = [
    'NOSE',
    'RIGHT_EYE_INNER',
    'RIGHT_EYE',
    'RIGHT_EYE_OUTER',
    'LEFT_EYE_INNER',
    'LEFT_EYE',
    'LEFT_EYE_OUTER',
    'RIGHT_EAR',
    'LEFT_EAR',
    'MOUTH_RIGHT',
    'MOUTH_LEFT',
    'RIGHT_SHOULDER',
    'LEFT_SHOULDER',
    'RIGHT_ELBOW',
    'LEFT_ELBOW',
    'RIGHT_WRIST',
    'LEFT_WRIST',
    'RIGHT_PINKY',
    'LEFT_PINKY',
    'RIGHT_INDEX',
    'LEFT_INDEX',
    'RIGHT_THUMB',
    'LEFT_THUMB',
    'RIGHT_HIP',
    'LEFT_HIP',
    'RIGHT_KNEE',
    'LEFT_KNEE',
    'RIGHT_ANKLE',
    'LEFT_ANKLE',
    'RIGHT_HEEL',
    'LEFT_HEEL',
    'RIGHT_FOOT_INDEX',
    'LEFT_FOOT_INDEX',
]


def component_points(component, width: int, height: int, num: int):
    """
    Gets component points

    Parameters
    ----------
    component : object
        Component with landmarks
    width : int
        Width
    height : int
        Height
    num : int
        number of landmarks

    Returns
    -------
    tuple of np.array
        coordinates and confidence for each landmark
    """
    if component is not None:
        lm = component.landmark
        return np.array([[p.x * width, p.y * height, p.z] for p in lm]), np.ones(num)

    return np.zeros((num, 3)), np.zeros(num)


def body_points(component, width: int, height: int, num: int):
    """
    gets body points

    Parameters
    ----------
    component : object
        component containing landmarks
    width : int
        width
    height : int
        Height
    num : int
        number of landmarks

    Returns
    -------
    tuple of np.array
        coordinates and visibility for each landmark.
    """
    if component is not None:
        lm = component.landmark
        return np.array([[p.x * width, p.y * height, p.z] for p in lm]), np.array([p.visibility for p in lm])

    return np.zeros((num, 3)), np.zeros(num)


def process_holistic(frames: list,
                     fps: float,
                     w: int,
                     h: int,
                     kinect=None,
                     progress=False,
                     additional_face_points=0,
                     additional_holistic_config={}) -> NumPyPoseBody:
    """
    process frames using holistic model from mediapipe

    Parameters
    ----------
    frames : list
        List of frames to be processed
    fps : float
        Frames per second
    w : int
        Frame width
    h : int
        Frame height.
    kinect : object, optional
        Kinect depth data.
    progress : bool, optional
        If True, show the progress bar.
    additional_face_points : int, optional
        Additional face landmarks (points)
    additional_holistic_config : dict, optional
        Additional configurations for holistic model

    Returns
    -------
    NumPyPoseBody
        Processed pose data
    """
    if 'static_image_mode' not in additional_holistic_config:
        additional_holistic_config['static_image_mode'] = False
    holistic = mp_holistic.Holistic(**additional_holistic_config)

    try:
        datas = []
        confs = []

        for i, frame in enumerate(tqdm(frames, disable=not progress)):
            results = holistic.process(frame)

            body_data, body_confidence = body_points(results.pose_landmarks, w, h, 33)
            face_data, face_confidence = component_points(results.face_landmarks, w, h,
                                                          FACE_POINTS_NUM(additional_face_points))
            lh_data, lh_confidence = component_points(results.left_hand_landmarks, w, h, 21)
            rh_data, rh_confidence = component_points(results.right_hand_landmarks, w, h, 21)
            body_world_data, body_world_confidence = body_points(results.pose_world_landmarks, w, h, 33)

            data = np.concatenate([body_data, face_data, lh_data, rh_data, body_world_data])
            conf = np.concatenate([body_confidence, face_confidence, lh_confidence, rh_confidence, body_world_confidence])

            if kinect is not None:
                kinect_depth = []
                for x, y, z in np.array(data, dtype="int32"):
                    if 0 < x < w and 0 < y < h:
                        kinect_depth.append(kinect[i, y, x, 0])
                    else:
                        kinect_depth.append(0)

                kinect_vec = np.expand_dims(np.array(kinect_depth), axis=-1)
                data = np.concatenate([data, kinect_vec], axis=-1)

            datas.append(data)
            confs.append(conf)

        pose_body_data = np.expand_dims(np.stack(datas), axis=1)
        pose_body_conf = np.expand_dims(np.stack(confs), axis=1)

        return NumPyPoseBody(data=pose_body_data, confidence=pose_body_conf, fps=fps)
    finally:
        holistic.close()


def holistic_hand_component(name, pf="XYZC") -> PoseHeaderComponent:
    """
    Creates holistic hand component

    Parameters
    ----------
    name : str
        Component name
    pf : str, optional
        Point format

    Returns
    -------
    PoseHeaderComponent
        Hand component
    """
    return PoseHeaderComponent(name=name, points=HAND_POINTS, limbs=HAND_LIMBS, colors=hand_colors, point_format=pf)


def holistic_components(pf="XYZC", additional_face_points=0):
    """
    Creates list of holistic components

    Parameters
    ----------
    pf : str, optional
        Point format
    additional_face_points : int, optional
        Additional face points/landmarks

    Returns
    -------
    list of PoseHeaderComponent
        List of holistic components.
    """
    face_limbs = list(FACE_LIMBS)
    if additional_face_points > 0:
        face_limbs += FACE_IRISES

    return [
        PoseHeaderComponent(name="POSE_LANDMARKS",
                            points=BODY_POINTS,
                            limbs=BODY_LIMBS,
                            colors=[(255, 0, 0)],
                            point_format=pf),
        PoseHeaderComponent(name="FACE_LANDMARKS",
                            points=FACE_POINTS(additional_face_points),
                            limbs=face_limbs,
                            colors=[(128, 0, 0)],
                            point_format=pf),
        holistic_hand_component("LEFT_HAND_LANDMARKS", pf),
        holistic_hand_component("RIGHT_HAND_LANDMARKS", pf),
        PoseHeaderComponent(name="POSE_WORLD_LANDMARKS",
                            points=BODY_POINTS,
                            limbs=BODY_LIMBS,
                            colors=[(255, 0, 0)],
                            point_format=pf),
    ]


def load_holistic(frames: list,
                  fps: float = 24,
                  width=1000,
                  height=1000,
                  depth=0,
                  kinect=None,
                  progress=False,
                  additional_holistic_config={}) -> Pose:
    """
    Loads holistic pose data

    Parameters
    ----------
    frames : list
        List of frames.
    fps : float, optional
        Frames per second.
    width : int, optional
        Frame width.
    height : int, optional
        Frame height.
    depth : int, optional
        Depth data.
    kinect : object, optional
        Kinect depth data.
    progress : bool, optional
        If True, show the progress bar.
    additional_holistic_config : dict, optional
        Additional configurations for the holistic model.

    Returns
    -------
    Pose
        Loaded pose data with header and body 
    """
    pf = "XYZC" if kinect is None else "XYZKC"

    dimensions = PoseHeaderDimensions(width=width, height=height, depth=depth)

    refine_face_landmarks = 'refine_face_landmarks' in additional_holistic_config and additional_holistic_config[
        'refine_face_landmarks']
    additional_face_points = 10 if refine_face_landmarks else 0
    header: PoseHeader = PoseHeader(version=0.2,
                                    dimensions=dimensions,
                                    components=holistic_components(pf, additional_face_points))
    body: NumPyPoseBody = process_holistic(frames, fps, width, height, kinect, progress, additional_face_points,
                                           additional_holistic_config)

    return Pose(header, body)


def formatted_holistic_pose(width: int, height: int, additional_face_points: int = 0):
    """
    Formatted holistic pose

    Parameters
    ----------
    width : int
        Pose width.
    height : int
        Pose height.
    additional_face_points : int, optional
        Additional face points/landmarks.

    Returns
    -------
    object
        Formatted pose components
    """
    dimensions = PoseHeaderDimensions(width=width, height=height, depth=1000)
    header = PoseHeader(version=0.2,
                        dimensions=dimensions,
                        components=holistic_components("XYZC", additional_face_points))
    body = NumPyPoseBody(
        fps=0,  # to be overridden later
        data=np.zeros(shape=(1, 1, header.total_points(), 3)),
        confidence=np.zeros(shape=(1, 1, header.total_points())))
    pose = Pose(header, body)
    return pose.get_components(["POSE_LANDMARKS", "FACE_LANDMARKS", "LEFT_HAND_LANDMARKS", "RIGHT_HAND_LANDMARKS"],
                               {"FACE_LANDMARKS": FACEMESH_CONTOURS_POINTS})


def load_mediapipe_directory(directory: str, fps: int, width: int, height: int, num_face_points: int = 128) -> Pose:
    """
    Load pose data from a directory of MediaPipe

    Parameters
    ----------
    directory : str
        Directory path.
    fps : float
        Frames per second.
    width : int
        Frame width.
    height : int
        Frame height.
    num_face_points : int, optional
        Number of face landmarks. Ideally, we don't want to hard code the 128 for the face, since face points can be 128 (reduced with refinement) or 118 (reduced without refinement) or 478 (full with refinement) or 468 (full without refinement)

    Returns
    -------
    Pose
        Loaded pose data
    """

    frames = load_frames_directory_dict(directory=directory, pattern="(?:^|\D)?(\d+).*?.json")

    if len(frames) > 0:
        first_frame = frames[0]
        num_pose_points = first_frame["pose_landmarks"]["num_landmarks"]
        num_left_hand_points = first_frame["left_hand_landmarks"]["num_landmarks"]
        num_right_hand_points = first_frame["right_hand_landmarks"]["num_landmarks"]
        additional_face_points = 10 if (num_face_points == 478 or num_face_points == 128) else 0
    else:
        raise ValueError("No frames found in directory: {}".format(directory))

    def load_mediapipe_frame(frame):
        """
        Get landmarks data of face landmarks, pose landmarks, and left & right hand landmarks (body_data, face_data, lh_data, rh_data) and confidence values from a given frame.

    Parameters
    ----------
    frame : dict
        Dictionary containing face, pose, left hand, and right hand landmark data.

    Returns
    -------
    tuple of numpy.ndarray
        A tuple containing two arrays:
        The first array is the landmarks data including x, y, z coordinates. 
        The second array is the confidence scores for each landmark.
         """

        def load_landmarks(name, num_points: int):
            points = [[float(p) for p in r.split(",")] for r in frame[name]["landmarks"]]
            points = [(ps + [1.0])[:4] for ps in points]  # Add visibility to all points
            if len(points) == 0:
                points = [[0, 0, 0, 0] for _ in range(num_points)]
            return np.array([[x, y, z] for x, y, z, c in points]), np.array([c for x, y, z, c in points])

        face_data, face_confidence = load_landmarks("face_landmarks", num_face_points)
        body_data, body_confidence = load_landmarks("pose_landmarks", num_pose_points)
        lh_data, lh_confidence = load_landmarks("left_hand_landmarks", num_left_hand_points)
        rh_data, rh_confidence = load_landmarks("right_hand_landmarks", num_right_hand_points)
        data = np.concatenate([body_data, face_data, lh_data, rh_data])
        conf = np.concatenate([body_confidence, face_confidence, lh_confidence, rh_confidence])
        return data, conf

    def load_mediapipe_frames() -> NumPyPoseBody:
        """
        From a list of frames, load  pose data and confidance into a NumPyPoseBody
        
        Processes each frame from `frames` to extract the data and confidence values
        for pose landmarks, face landmarks, and left & right hand landmarks.

        Returns
        -------
        NumPyPoseBody
            PoseBody object with data and confidence for each frame.
        """
        max_frames = int(max(frames.keys())) + 1
        pose_body_data = np.zeros(shape=(max_frames, 1, num_left_hand_points + num_right_hand_points + num_pose_points +
                                         num_face_points, 3),
                                  dtype=float)
        pose_body_conf = np.zeros(shape=(max_frames, 1, num_left_hand_points + num_right_hand_points + num_pose_points +
                                         num_face_points),
                                  dtype=float)
        for frame_id, frame in frames.items():
            data, conf = load_mediapipe_frame(frame)
            pose_body_data[frame_id][0] = data
            pose_body_conf[frame_id][0] = conf
        return NumPyPoseBody(data=pose_body_data, confidence=pose_body_conf, fps=fps)

    pose = formatted_holistic_pose(width=width, height=height, additional_face_points=additional_face_points)

    pose.body = load_mediapipe_frames()

    return pose

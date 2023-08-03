import numpy as np
from tqdm import tqdm

from .openpose import hand_colors
from ..numpy.pose_body import NumPyPoseBody
from ..pose import Pose
from ..pose_header import PoseHeader, PoseHeaderComponent, PoseHeaderDimensions
from ..utils.openpose import load_frames_directory_dict

try:
    import mediapipe as mp
except ImportError:
    raise ImportError("Please install mediapipe with: pip install mediapipe")

mp_holistic = mp.solutions.holistic

BODY_POINTS = mp_holistic.PoseLandmark._member_names_
BODY_LIMBS = [(int(a), int(b)) for a, b in mp_holistic.POSE_CONNECTIONS]

HAND_POINTS = mp_holistic.HandLandmark._member_names_
HAND_LIMBS = [(int(a), int(b)) for a, b in mp_holistic.HAND_CONNECTIONS]

FACE_POINTS_NUM = lambda additional_points=0: additional_points + 468
FACE_POINTS = lambda additional_points=0: [str(i) for i in range(FACE_POINTS_NUM(additional_points))]
FACE_LIMBS = [(int(a), int(b)) for a, b in mp_holistic.FACEMESH_TESSELATION]

FLIPPED_BODY_POINTS = ['NOSE', 'RIGHT_EYE_INNER', 'RIGHT_EYE', 'RIGHT_EYE_OUTER', 'LEFT_EYE_INNER', 'LEFT_EYE',
                       'LEFT_EYE_OUTER', 'RIGHT_EAR', 'LEFT_EAR', 'MOUTH_RIGHT', 'MOUTH_LEFT', 'RIGHT_SHOULDER',
                       'LEFT_SHOULDER', 'RIGHT_ELBOW', 'LEFT_ELBOW', 'RIGHT_WRIST', 'LEFT_WRIST', 'RIGHT_PINKY',
                       'LEFT_PINKY', 'RIGHT_INDEX', 'LEFT_INDEX', 'RIGHT_THUMB', 'LEFT_THUMB', 'RIGHT_HIP', 'LEFT_HIP',
                       'RIGHT_KNEE', 'LEFT_KNEE', 'RIGHT_ANKLE', 'LEFT_ANKLE', 'RIGHT_HEEL', 'LEFT_HEEL',
                       'RIGHT_FOOT_INDEX', 'LEFT_FOOT_INDEX', ]


def component_points(component, width: int, height: int, num: int):
    if component is not None:
        lm = component.landmark
        return np.array([[p.x * width, p.y * height, p.z] for p in lm]), np.ones(num)

    return np.zeros((num, 3)), np.zeros(num)


def body_points(component, width: int, height: int, num: int):
    if component is not None:
        lm = component.landmark
        return np.array([[p.x * width, p.y * height, p.z] for p in lm]), np.array([p.visibility for p in lm])

    return np.zeros((num, 3)), np.zeros(num)


def process_holistic(frames: list, fps: float, w: int, h: int, kinect=None, progress=False, additional_face_points=0,
                     additional_holistic_config={}):
    holistic = mp_holistic.Holistic(static_image_mode=False, **additional_holistic_config)

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

    holistic.close()

    return NumPyPoseBody(data=pose_body_data, confidence=pose_body_conf, fps=fps)


def holistic_hand_component(name, pf="XYZC"):
    return PoseHeaderComponent(name=name, points=HAND_POINTS, limbs=HAND_LIMBS, colors=hand_colors, point_format=pf)


def holistic_components(pf="XYZC", additional_face_points=0):
    return [
        PoseHeaderComponent(name="POSE_LANDMARKS", points=BODY_POINTS, limbs=BODY_LIMBS,
                            colors=[(255, 0, 0)], point_format=pf),
        PoseHeaderComponent(name="FACE_LANDMARKS", points=FACE_POINTS(additional_face_points), limbs=FACE_LIMBS,
                            colors=[(128, 0, 0)], point_format=pf),
        holistic_hand_component("LEFT_HAND_LANDMARKS", pf),
        holistic_hand_component("RIGHT_HAND_LANDMARKS", pf),
        PoseHeaderComponent(name="POSE_WORLD_LANDMARKS", points=BODY_POINTS, limbs=BODY_LIMBS,
                            colors=[(255, 0, 0)], point_format=pf),
    ]


def load_holistic(frames: list, fps: float = 24, width=1000, height=1000, depth=0, kinect=None, progress=False,
                  additional_holistic_config={}):
    pf = "XYZC" if kinect is None else "XYZKC"

    dimensions = PoseHeaderDimensions(width=width, height=height, depth=depth)

    refine_face_landmarks = 'refine_face_landmarks' in additional_holistic_config and additional_holistic_config[
        'refine_face_landmarks']
    additional_face_points = 10 if refine_face_landmarks else 0
    header: PoseHeader = PoseHeader(version=0.1, dimensions=dimensions,
                                    components=holistic_components(pf, additional_face_points))
    body: NumPyPoseBody = process_holistic(frames, fps, width, height, kinect, progress, additional_face_points,
                                           additional_holistic_config)

    return Pose(header, body)


def formatted_holistic_pose(width: int, height: int):
    dimensions = PoseHeaderDimensions(width=width, height=height, depth=1000)
    header = PoseHeader(version=0.1, dimensions=dimensions, components=holistic_components("XYZC", 10))
    body = NumPyPoseBody(fps=10,
                         data=np.zeros(shape=(1, 1, header.total_points(), 3)),
                         confidence=np.zeros(shape=(1, 1, header.total_points())))
    pose = Pose(header, body)
    return pose


def load_mediapipe_directory(directory: str, fps: int, width: int, height: int) -> Pose:
    """
    :param directory:
    :param fps:
    :param width:
    :param height:
    :return:
    """

    frames = load_frames_directory_dict(directory=directory, pattern="(?:^|\D)?(\d+).*?.json")

    def load_mediapipe_frame(frame):
        def load_landmarks(name, num_points: int):
            points = [[float(p) for p in r.split(",")] for r in frame[name]["landmarks"]]
            points = [(ps + [1.0])[:4] for ps in points]  # Add visibility to all points
            if len(points) == 0:
                points = [[0, 0, 0, 0] for _ in range(num_points)]
            return np.array([[x, y, z] for x, y, z, c in points]), np.array([c for x, y, z, c in points])
        face_data, face_confidence = load_landmarks("face_landmarks", 128)
        body_data, body_confidence = load_landmarks("pose_landmarks", 33)
        lh_data, lh_confidence = load_landmarks("left_hand_landmarks", 21)
        rh_data, rh_confidence = load_landmarks("right_hand_landmarks", 21)
        data = np.concatenate([body_data, face_data, lh_data, rh_data])
        conf = np.concatenate([body_confidence, face_confidence, lh_confidence, rh_confidence])
        return data, conf

    def load_mediapipe_frames():
        max_frames = int(max(frames.keys())) + 1
        pose_body_data = np.zeros(shape=(max_frames, 1, 21 + 21 + 33 + 128, 3), dtype=float)
        pose_body_conf = np.zeros(shape=(max_frames, 1, 21 + 21 + 33 + 128), dtype=float)
        for frame_id, frame in frames.items():
            data, conf = load_mediapipe_frame(frame)
            pose_body_data[frame_id][0] = data
            pose_body_conf[frame_id][0] = conf
        return NumPyPoseBody(data=pose_body_data, confidence=pose_body_conf, fps=fps)

    pose = formatted_holistic_pose(width=width, height=height)

    pose.body = load_mediapipe_frames()

    return pose

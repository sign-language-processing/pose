import mediapipe as mp
import numpy as np
from tqdm import tqdm

from .openpose import hand_colors
from ..numpy.pose_body import NumPyPoseBody
from ..pose import Pose
from ..pose_header import PoseHeaderComponent, PoseHeaderDimensions, PoseHeader

mp_holistic = mp.solutions.holistic

BODY_POINTS = mp_holistic.PoseLandmark._member_names_
BODY_LIMBS = [(int(a), int(b)) for a, b in mp_holistic.POSE_CONNECTIONS]

HAND_POINTS = mp_holistic.HandLandmark._member_names_
HAND_LIMBS = [(int(a), int(b)) for a, b in mp_holistic.HAND_CONNECTIONS]

FACE_POINTS = [str(i) for i in range(468)]
FACE_LIMBS = [(int(a), int(b)) for a, b in mp_holistic.FACE_CONNECTIONS]


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


def process_holistic(frames: list, fps: float, w: int, h: int, kinect=None, progress=False):
    holistic = mp_holistic.Holistic(static_image_mode=False)

    datas = []
    confs = []

    for i, frame in enumerate(tqdm(frames, disable=not progress)):
        results = holistic.process(frame)

        body_data, body_confidence = body_points(results.pose_landmarks, w, h, 33)
        face_data, face_confidence = component_points(results.face_landmarks, w, h, 468)
        lh_data, lh_confidence = component_points(results.left_hand_landmarks, w, h, 21)
        rh_data, rh_confidence = component_points(results.right_hand_landmarks, w, h, 21)

        data = np.concatenate([body_data, face_data, lh_data, rh_data])
        conf = np.concatenate([body_confidence, face_confidence, lh_confidence, rh_confidence])

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


def load_holistic(frames: list, fps: float = 24, width=1000, height=1000, depth=0, kinect=None):
    pf = "XYZC" if kinect is None else "XYZKC"

    Holistic_Hand_Component = lambda name: PoseHeaderComponent(name=name, points=HAND_POINTS,
                                                               limbs=HAND_LIMBS, colors=hand_colors, point_format=pf)
    Holistic_Components = [
        PoseHeaderComponent(name="POSE_LANDMARKS", points=BODY_POINTS, limbs=BODY_LIMBS,
                            colors=[(255, 0, 0)], point_format=pf),
        PoseHeaderComponent(name="FACE_LANDMARKS", points=FACE_POINTS, limbs=FACE_LIMBS,
                            colors=[(128, 0, 0)], point_format=pf),
        Holistic_Hand_Component("LEFT_HAND_LANDMARKS"),
        Holistic_Hand_Component("RIGHT_HAND_LANDMARKS"),
    ]

    dimensions = PoseHeaderDimensions(width=width, height=height, depth=depth)

    header: PoseHeader = PoseHeader(version=0.1, dimensions=dimensions, components=Holistic_Components)
    body: NumPyPoseBody = process_holistic(frames, fps, width, height, kinect)

    return Pose(header, body)
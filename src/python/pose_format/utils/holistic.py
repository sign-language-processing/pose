import json
import os
import threading
import warnings
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List

import numpy as np
from tqdm import tqdm

from ..numpy.pose_body import NumPyPoseBody
from ..pose import Pose
from ..pose_header import PoseHeader, PoseHeaderComponent, PoseHeaderDimensions
from .generic import detect_known_pose_format
from .openpose import hand_colors, load_frames_directory_dict

try:
    import mediapipe as mp
    from mediapipe.python.solutions.face_mesh_connections import FACEMESH_IRISES
except ImportError:
    raise ImportError("Please install mediapipe with: pip install \"mediapipe<0.10.30\"")

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

def _swap_left_right(name: str) -> str:
    return name.replace("LEFT", "\0").replace("RIGHT", "LEFT").replace("\0", "RIGHT")


FLIPPED_BODY_POINTS = [_swap_left_right(p) for p in BODY_POINTS]

# Left-right mirror permutation for the 478 face-mesh landmarks: FLIPPED_FACE_POINTS[i] is the
# index whose canonical position is the horizontal reflection of point i. The first 468 entries are
# derived from MediaPipe's symmetric canonical face model (it is not shipped in the pip package):
#   https://raw.githubusercontent.com/google-ai-edge/mediapipe/master/mediapipe/modules/face_geometry/data/canonical_face_model.obj
# Parse its 468 "v x y z" vertices, negate x, and for each take the nearest original vertex; the
# reflection lands exactly on another vertex (max distance 0.0), so the mapping is exact and a clean
# involution. The last 10 entries swap the refined iris landmarks (468-477), which the .obj omits,
# derived from running holistic on an image and its horizontal flip. The first 468 form a
# self-contained permutation for the unrefined (468-point) mesh.
FLIPPED_FACE_POINTS = [
    0, 1, 2, 248, 4, 5, 6, 249, 8, 9, 10, 11, 12, 13, 14, 15,
    16, 17, 18, 19, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261,
    262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277,
    278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293,
    294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 309,
    310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 94, 324,
    325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340,
    341, 342, 343, 344, 345, 346, 347, 348, 349, 350, 351, 352, 353, 354, 355, 356,
    357, 358, 359, 360, 361, 362, 363, 364, 365, 366, 367, 368, 369, 370, 371, 372,
    373, 374, 375, 376, 377, 378, 379, 151, 152, 380, 381, 382, 383, 384, 385, 386,
    387, 388, 389, 390, 164, 391, 392, 393, 168, 394, 395, 396, 397, 398, 399, 175,
    400, 401, 402, 403, 404, 405, 406, 407, 408, 409, 410, 411, 412, 413, 414, 415,
    416, 417, 418, 195, 419, 197, 420, 199, 200, 421, 422, 423, 424, 425, 426, 427,
    428, 429, 430, 431, 432, 433, 434, 435, 436, 437, 438, 439, 440, 441, 442, 443,
    444, 445, 446, 447, 448, 449, 450, 451, 452, 453, 454, 455, 456, 457, 458, 459,
    460, 461, 462, 463, 464, 465, 466, 467, 3, 7, 20, 21, 22, 23, 24, 25,
    26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41,
    42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57,
    58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73,
    74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89,
    90, 91, 92, 93, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106,
    107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122,
    123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138,
    139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 153, 154, 155, 156,
    157, 158, 159, 160, 161, 162, 163, 165, 166, 167, 169, 170, 171, 172, 173, 174,
    176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191,
    192, 193, 194, 196, 198, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211,
    212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227,
    228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243,
    244, 245, 246, 247, 473, 476, 475, 474, 477, 468, 471, 470, 469, 472,
]


class HolisticPool:
    _lock = threading.Lock()
    _instances: Dict[str, List] = {}
    # A small blank frame used to clear tracking state between videos.
    # Faster than reset() (which tears down and restarts the entire MediaPipe graph)
    # while producing identical results to static_image_mode=True on the next frame.
    _BLANK_FRAME = np.ones((256, 256, 3), dtype=np.uint8) * 255

    @staticmethod
    def _config_key(config: dict) -> str:
        return json.dumps(config, sort_keys=True)

    @classmethod
    def acquire(cls, n: int, config: dict) -> list:
        key = cls._config_key(config)
        acquired = []
        with cls._lock:
            pool = cls._instances.setdefault(key, [])
            while pool and len(acquired) < n:
                acquired.append(pool.pop())

        need = n - len(acquired)

        # Clear internal graph state for reused instances by processing a blank frame.
        # Faster than reset() (which tears down and restarts the entire graph)
        # while flushing stale tracking/detection state from a previous video.
        if acquired:
            with ThreadPoolExecutor(max_workers=len(acquired)) as ex:
                list(ex.map(lambda h: h.process(cls._BLANK_FRAME), acquired))

        if need > 0:
            with ThreadPoolExecutor(max_workers=need) as ex:
                new = list(ex.map(lambda _: mp_holistic.Holistic(**config), range(need)))
            acquired.extend(new)

        return acquired

    @classmethod
    def release(cls, instances: list, config: dict):
        key = cls._config_key(config)
        with cls._lock:
            cls._instances.setdefault(key, []).extend(instances)


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


def _extract_frame_data(results, w, h, additional_face_points, kinect=None, frame_idx=0):
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
                kinect_depth.append(kinect[frame_idx, y, x, 0])
            else:
                kinect_depth.append(0)

        kinect_vec = np.expand_dims(np.array(kinect_depth), axis=-1)
        data = np.concatenate([data, kinect_vec], axis=-1)

    return data, conf


def process_holistic(frames: list,
                     fps: float,
                     w: int,
                     h: int,
                     kinect=None,
                     progress=False,
                     additional_face_points=0,
                     additional_holistic_config={},
                     pose_workers=1,
                     reuse=True) -> NumPyPoseBody:
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
    pose_workers : int, optional
        Number of parallel holistic instances for interleaved processing.
    reuse : bool, optional
        If True, holistic instances are pooled and reused across calls.

    Returns
    -------
    NumPyPoseBody
        Processed pose data
    """
    if 'static_image_mode' not in additional_holistic_config:
        additional_holistic_config['static_image_mode'] = False

    if pose_workers <= 0:
        pose_workers = os.cpu_count() or 1

    if pose_workers > 1 and not additional_holistic_config['static_image_mode']:
        warnings.warn("Using multiple workers with static_image_mode=False is not thread-safe; "
                      "results may vary slightly.", stacklevel=2)

    holistics = HolisticPool.acquire(pose_workers, additional_holistic_config)

    try:
        datas = []
        confs = []

        def process_on_worker(worker_idx, frame):
            return holistics[worker_idx].process(frame)

        with ThreadPoolExecutor(max_workers=pose_workers) as executor:
            pending = {}
            worker_futures = {}
            next_to_collect = 0

            for i, frame in enumerate(tqdm(frames, disable=not progress)):
                worker_idx = i % pose_workers
                if worker_idx in worker_futures:
                    worker_futures[worker_idx].result()
                future = executor.submit(process_on_worker, worker_idx, frame)
                pending[i] = future
                worker_futures[worker_idx] = future

                while next_to_collect in pending and pending[next_to_collect].done():
                    results = pending.pop(next_to_collect).result()
                    data, conf = _extract_frame_data(results, w, h, additional_face_points, kinect, next_to_collect)
                    datas.append(data)
                    confs.append(conf)
                    next_to_collect += 1

            while next_to_collect in pending:
                results = pending.pop(next_to_collect).result()
                data, conf = _extract_frame_data(results, w, h, additional_face_points, kinect, next_to_collect)
                datas.append(data)
                confs.append(conf)
                next_to_collect += 1

        if not datas:
            raise ValueError("need at least one array to stack")

        pose_body_data = np.expand_dims(np.stack(datas), axis=1)
        pose_body_conf = np.expand_dims(np.stack(confs), axis=1)
        return NumPyPoseBody(data=pose_body_data, confidence=pose_body_conf, fps=fps)
    finally:
        if reuse:
            HolisticPool.release(holistics, additional_holistic_config)
        else:
            for h in holistics:
                h.close()


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


def mirror_horizontal(pose: Pose) -> Pose:
    """
    Horizontally mirror a holistic pose, as if the source image had been flipped left-to-right.

    Mirroring an image swaps the subject's left and right sides, so a correct mirror must do more
    than negate the x coordinate: image-space points are reflected as ``width - x``, body landmarks
    are relabelled via ``FLIPPED_BODY_POINTS``, the two hand components are swapped, and the face
    mesh is reindexed via ``FLIPPED_FACE_POINTS``.

    ``POSE_WORLD_LANDMARKS`` is left unchanged. MediaPipe reconstructs world landmarks in a canonical
    3D frame and cannot resolve the left/right ambiguity of a single view, so flipping the image does
    not mirror them - re-running holistic on a flipped image returns essentially the same world
    landmarks, and we match that behaviour.

    Parameters
    ----------
    pose : Pose
        A holistic pose (must contain a ``POSE_LANDMARKS`` component).

    Returns
    -------
    Pose
        A new mirrored pose. The input is not modified.
    """
    known_pose_format = detect_known_pose_format(pose)
    if known_pose_format != "holistic":
        raise NotImplementedError(
            f"Unsupported pose header schema {known_pose_format} for {mirror_horizontal.__name__}: {pose.header}"
        )

    components_by_name = {c.name: c for c in pose.header.components}

    component_start = {}
    idx = 0
    for c in pose.header.components:
        component_start[c.name] = idx
        idx += len(c.points)

    body_name_flip = dict(zip(BODY_POINTS, FLIPPED_BODY_POINTS))
    face_name_flip = {str(i): str(flipped) for i, flipped in enumerate(FLIPPED_FACE_POINTS)}
    mirror_component = {
        "LEFT_HAND_LANDMARKS": "RIGHT_HAND_LANDMARKS",
        "RIGHT_HAND_LANDMARKS": "LEFT_HAND_LANDMARKS",
    }

    width = pose.header.dimensions.width

    perm = []
    flip_x = []
    for c in pose.header.components:
        is_world = "WORLD" in c.name
        source = c if is_world else components_by_name.get(mirror_component.get(c.name, c.name), c)
        for point in c.points:
            if is_world:  # world landmarks are not mirrored, see docstring
                flipped_point = point
            elif c.name == "POSE_LANDMARKS":
                flipped_point = body_name_flip.get(point, point)
            elif c.name == "FACE_LANDMARKS":
                flipped_point = face_name_flip.get(point, point)
            else:
                flipped_point = point

            if flipped_point in source.points:
                perm.append(component_start[source.name] + source.points.index(flipped_point))
            else:  # flipped counterpart not present (e.g. a reduced subset): keep the point in place
                perm.append(component_start[c.name] + c.points.index(point))
            flip_x.append(not is_world)

    perm = np.array(perm)
    flip_x = np.array(flip_x)

    data = pose.body.data[:, :, perm, :]
    data[:, :, flip_x, 0] = width - data[:, :, flip_x, 0]
    confidence = pose.body.confidence[:, :, perm]

    mirrored = pose.copy()
    mirrored.body.data = data
    mirrored.body.confidence = confidence
    return mirrored


def load_holistic(frames: list,
                  fps: float = 24,
                  width=1000,
                  height=1000,
                  depth=0,
                  kinect=None,
                  progress=False,
                  additional_holistic_config={},
                  pose_workers=1,
                  reuse=True) -> Pose:
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
                                           additional_holistic_config, pose_workers=pose_workers, reuse=reuse)

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


def load_mediapipe_directory(directory: str, fps: float, width: int, height: int, num_face_points: int = 128) -> Pose:
    """
    Load pose data from a directory of MediaPipe Holistic outputs. This function exists for loading a LEGACY format
    produced by the University of Surrey for one single dataset: Mediapipe Holistic poses for the WMT-SLT 22 data.

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

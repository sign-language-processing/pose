from typing import Tuple, Literal, List, Union
import copy
import numpy as np
import numpy.ma as ma
from pose_format.pose import Pose
from pose_format.numpy import NumPyPoseBody
from pose_format.pose_header import PoseHeader, PoseHeaderDimensions, PoseHeaderComponent, PoseNormalizationInfo
from pose_format.utils.normalization_3d import PoseNormalizer
from pose_format.utils.openpose import OpenPose_Components
from pose_format.utils.openpose import BODY_POINTS as OPENPOSE_BODY_POINTS
from pose_format.utils.openpose_135 import OpenPose_Components as OpenPose135_Components

# from pose_format.utils.holistic import holistic_components
# The import above creates an error: ImportError: Please install mediapipe with: pip install mediapipe

KnownPoseFormat = Literal["holistic", "openpose", "openpose_135"]


def get_component_names(
    pose_or_header_or_components: Union[Pose,PoseHeader]) -> List[str]:
    if isinstance(pose_or_header_or_components, Pose):
        return [c.name for c in pose_or_header_or_components.header.components]
    if isinstance(pose_or_header_or_components, PoseHeader):
        return [c.name for c in pose_or_header_or_components.components]
    raise ValueError(f"Could not get component_names from {pose_or_header_or_components}")


def detect_known_pose_format(pose_or_header: Union[Pose,PoseHeader]) -> KnownPoseFormat:
    component_names= get_component_names(pose_or_header)

    # would be better to import from pose_format.utils.holistic but that creates a dep on mediapipe
    mediapipe_components = [
        "POSE_LANDMARKS",
        "FACE_LANDMARKS",
        "LEFT_HAND_LANDMARKS",
        "RIGHT_HAND_LANDMARKS",
        "POSE_WORLD_LANDMARKS",
    ]

    openpose_components = [c.name for c in OpenPose_Components]

    openpose_135_components = [c.name for c in OpenPose135_Components]

    for component_name in component_names:
        if component_name in mediapipe_components:
            return "holistic"
        if component_name in openpose_components:
            return "openpose"
        if component_name in openpose_135_components:
            return "openpose_135"

    raise ValueError(
        f"Could not detect pose format, unknown pose header schema with component names: {component_names}"
    )


def normalize_pose_size(pose: Pose, target_width: int = 512):
    shift = 1.25
    shoulder_width = (target_width / shift) / 2
    shift_vec = np.full(shape=(pose.body.data.shape[-1]), fill_value=shift, dtype=np.float32)
    pose.body.data = (pose.body.data + shift_vec) * shoulder_width
    pose.header.dimensions.height = pose.header.dimensions.width = target_width


def pose_hide_legs(pose: Pose, remove: bool = False) -> Pose:
    """
    Hide or remove leg components from a pose.
    
    If `remove` is True, the leg components are removed; otherwise, they are hidden (zeroed out).
    """
    known_pose_format = detect_known_pose_format(pose)

    if known_pose_format == "holistic":
        point_names = ["KNEE", "ANKLE", "HEEL", "FOOT_INDEX", "HIP"]
        sides = ["LEFT", "RIGHT"]
        point_names_to_remove = [f"{side}_{name}" for side in sides for name in point_names]
        points_to_remove_dict = {
            "POSE_LANDMARKS": point_names_to_remove,
            "POSE_WORLD_LANDMARKS": point_names_to_remove,
        }

    elif known_pose_format == "openpose":
        words_to_look_for = ["Hip", "Knee", "Ankle", "BigToe", "SmallToe", "Heel"]
        point_names_to_remove = [point for point in OPENPOSE_BODY_POINTS
                                 if any(word in point for word in words_to_look_for)]

            # if any of the items in point_
        points_to_remove_dict = {"pose_keypoints_2d": point_names_to_remove}

    else:
        raise NotImplementedError(
            f"Unsupported pose header schema {known_pose_format} for {pose_hide_legs.__name__}: {pose.header}"
        )

    if remove:
        return pose.remove_components([], points_to_remove_dict)

    # Hide the points instead of removing them
    point_indices = []
    for component, points in points_to_remove_dict.items():
        for point_name in points:
            try:
                point_index = pose.header.get_point_index(component, point_name)
                point_indices.append(point_index)
            except ValueError: # point not found, maybe removed earlier in other preprocessing steps
                pass


    pose.body.data[:, :, point_indices, :] = 0
    pose.body.confidence[:, :, point_indices] = 0

    return pose


def pose_shoulders(pose_header: PoseHeader) -> Tuple[Tuple[str, str], Tuple[str, str]]:
    known_pose_format = detect_known_pose_format(pose_header)

    if known_pose_format == "holistic":
        return ("POSE_LANDMARKS", "RIGHT_SHOULDER"), ("POSE_LANDMARKS", "LEFT_SHOULDER")

    if known_pose_format == "openpose_135":
        return ("BODY_135", "RShoulder"), ("BODY_135", "LShoulder")

    if known_pose_format == "openpose":
        return ("pose_keypoints_2d", "RShoulder"), ("pose_keypoints_2d", "LShoulder")

    raise NotImplementedError(
        f"Unsupported pose header schema {known_pose_format} for {pose_shoulders.__name__}: {pose_header}"
    )


def hands_indexes(pose_header: PoseHeader)-> List[int]:
    known_pose_format = detect_known_pose_format(pose_header)
    if known_pose_format == "holistic":
        return [
            pose_header.get_point_index("LEFT_HAND_LANDMARKS", "MIDDLE_FINGER_MCP"),
            pose_header.get_point_index("RIGHT_HAND_LANDMARKS", "MIDDLE_FINGER_MCP"),
        ]

    if known_pose_format == "openpose":
        return [
            pose_header.get_point_index("hand_left_keypoints_2d", "M_CMC"),
            pose_header.get_point_index("hand_right_keypoints_2d", "M_CMC"),
        ]
    raise NotImplementedError(
        f"Unsupported pose header schema {known_pose_format} for {hands_indexes.__name__}: {pose_header}"
    )


def pose_normalization_info(pose_header: PoseHeader) ->PoseNormalizationInfo:
    (c1, p1), (c2, p2) = pose_shoulders(pose_header)
    return pose_header.normalization_info(p1=(c1, p1), p2=(c2, p2))


def hands_components(pose_header: PoseHeader)-> Tuple[Tuple[str, str], Tuple[str, str, str], Tuple[str, str]]:
    known_pose_format = detect_known_pose_format(pose_header)
    if known_pose_format == "holistic":
        return (
            ("LEFT_HAND_LANDMARKS", "RIGHT_HAND_LANDMARKS"),
            ("WRIST", "PINKY_MCP", "INDEX_FINGER_MCP"),
            ("WRIST", "MIDDLE_FINGER_MCP"),
        )

    if known_pose_format == "openpose":
        return ("hand_left_keypoints_2d", "hand_right_keypoints_2d"), ("BASE", "P_CMC", "I_CMC"), ("BASE", "M_CMC")

    raise NotImplementedError(
        f"Unsupported pose header schema '{known_pose_format}' for {hands_components.__name__}: {pose_header}"
    )


def normalize_component_3d(pose, component_name: str, plane: Tuple[str, str, str], line: Tuple[str, str]):
    hand_pose = pose.get_components([component_name])
    plane_info = hand_pose.header.normalization_info(
        p1=(component_name, plane[0]),
        p2=(component_name, plane[1]),
        p3=(component_name, plane[2])
    )
    line_info = hand_pose.header.normalization_info(
        p1=(component_name, line[0]),
        p2=(component_name, line[1])
        )

    normalizer = PoseNormalizer(plane=plane_info, line=line_info)
    normalized_hand = normalizer(hand_pose.body.data)

    # Add normalized hand to pose
    pose.body.data = ma.concatenate([pose.body.data, normalized_hand], axis=2).astype(np.float32)
    pose.body.confidence = np.concatenate([pose.body.confidence, hand_pose.body.confidence], axis=2)


def normalize_hands_3d(pose: Pose, left_hand=True, right_hand=True):
    (left_hand_component, right_hand_component), plane, line = hands_components(pose.header)
    if left_hand:
        normalize_component_3d(pose, left_hand_component, plane, line)
    if right_hand:
        normalize_component_3d(pose, right_hand_component, plane, line)


def get_standard_components_for_known_format(known_pose_format: KnownPoseFormat) -> List[PoseHeaderComponent]:
    if known_pose_format == "holistic":
        try:
            # pylint: disable=import-outside-toplevel
            import pose_format.utils.holistic as holistic_utils
            return holistic_utils.holistic_components()
        except ImportError as e:
            raise e
    if known_pose_format == "openpose":
        return OpenPose_Components
    if known_pose_format == "openpose_135":
        return OpenPose135_Components

    raise NotImplementedError(f"Unsupported pose header schema {known_pose_format}")


def fake_pose(num_frames: int, fps: int=25, components: Union[List[PoseHeaderComponent],None]=None)->Pose:
    if components is None:
        components = copy.deepcopy(OpenPose_Components) # fixes W0102, dangerous default value

    if components[0].format == "XYZC":
        dimensions = PoseHeaderDimensions(width=1, height=1, depth=1)
    elif components[0].format == "XYC":
        dimensions = PoseHeaderDimensions(width=1, height=1)
    else:
        raise ValueError(f"Unknown point format: {components[0].format}")
    header = PoseHeader(version=0.2, dimensions=dimensions, components=components)

    total_points = header.total_points()
    data = np.random.randn(num_frames, 1, total_points, header.num_dims())
    confidence = np.random.randn(num_frames, 1, total_points)
    masked_data = ma.masked_array(data)

    body = NumPyPoseBody(fps=int(fps), data=masked_data, confidence=confidence)

    return Pose(header, body)


def get_hand_wrist_index(pose: Pose, hand: str)-> int:
    known_pose_format = detect_known_pose_format(pose)
    if known_pose_format == "holistic":
        return pose.header.get_point_index(f"{hand.upper()}_HAND_LANDMARKS", "WRIST")
    if known_pose_format == "openpose":
        return pose.header.get_point_index(f"hand_{hand.lower()}_keypoints_2d", "BASE")
    raise NotImplementedError(
        f"Unsupported pose header schema {known_pose_format} for {get_hand_wrist_index.__name__}: {pose.header}"
    )


def get_body_hand_wrist_index(pose: Pose, hand: str)-> int:
    known_pose_format = detect_known_pose_format(pose)
    if known_pose_format == "holistic":
        return pose.header.get_point_index("POSE_LANDMARKS", f"{hand.upper()}_WRIST")
    if known_pose_format == "openpose":
        return pose.header.get_point_index("pose_keypoints_2d", f"{hand.upper()[0]}Wrist")
    raise NotImplementedError(
        f"Unsupported pose header schema {known_pose_format} for {get_body_hand_wrist_index.__name__}: {pose.header}"
    )


def correct_wrist(pose: Pose, hand: str) -> Pose:
    pose = copy.deepcopy(pose) # was previously modifying the input
    wrist_index = get_hand_wrist_index(pose, hand)
    wrist = pose.body.data[:, :, wrist_index]
    wrist_conf = pose.body.confidence[:, :, wrist_index]

    body_wrist_index = get_body_hand_wrist_index(pose, hand)
    body_wrist = pose.body.data[:, :, body_wrist_index]
    body_wrist_conf = pose.body.confidence[:, :, body_wrist_index]

    point_coordinate_count = wrist.shape[-1]
    stacked_conf = np.stack([wrist_conf] * point_coordinate_count, axis=-1)
    new_wrist_data = ma.where(stacked_conf == 0, body_wrist, wrist)
    new_wrist_conf = ma.where(wrist_conf == 0, body_wrist_conf, wrist_conf)

    pose.body.data[:, :, body_wrist_index] = new_wrist_data
    pose.body.confidence[:, :, body_wrist_index] = new_wrist_conf
    return pose


def correct_wrists(pose: Pose) -> Pose:
    pose = correct_wrist(pose, "LEFT")
    pose = correct_wrist(pose, "RIGHT")
    return pose


def reduce_holistic(pose: Pose) -> Pose:
    known_pose_format = detect_known_pose_format(pose)
    if known_pose_format != "holistic":
        return pose
    # pylint: disable=pointless-string-statement
    """
    # from mediapipe.python.solutions.face_mesh_connections import FACEMESH_CONTOURS
    # points_set = set([p for p_tup in list(FACEMESH_CONTOURS) for p in p_tup])
    # face_contours = [str(p) for p in sorted(points_set)]
    # print(face_contours)
    """
    # To avoid installing mediapipe, we just hardcode the face contours given the above code
    face_contours = [
        '0', '7', '10', '13', '14', '17', '21', '33', '37', '39', '40', '46', '52', '53', '54', '55', '58', '61', '63',
        '65', '66', '67', '70', '78', '80', '81', '82', '84', '87', '88', '91', '93', '95', '103', '105', '107', '109',
        '127', '132', '133', '136', '144', '145', '146', '148', '149', '150', '152', '153', '154', '155', '157', '158',
        '159', '160', '161', '162', '163', '172', '173', '176', '178', '181', '185', '191', '234', '246', '249', '251',
        '263', '267', '269', '270', '276', '282', '283', '284', '285', '288', '291', '293', '295', '296', '297', '300',
        '308', '310', '311', '312', '314', '317', '318', '321', '323', '324', '332', '334', '336', '338', '356', '361',
        '362', '365', '373', '374', '375', '377', '378', '379', '380', '381', '382', '384', '385', '386', '387', '388',
        '389', '390', '397', '398', '400', '402', '405', '409', '415', '454', '466'
    ]

    ignore_names = [
        "EAR", "NOSE", "MOUTH", "EYE",  # Face
        "THUMB", "PINKY", "INDEX",  # Hands
        "KNEE", "ANKLE", "HEEL", "FOOT_INDEX"  # Feet
    ]

    body_component = [c for c in pose.header.components if c.name == "POSE_LANDMARKS"][0]
    body_no_face_no_hands = [p for p in body_component.points if all([i not in p for i in ignore_names])]

    components = [c.name for c in pose.header.components if c.name != "POSE_WORLD_LANDMARKS"]
    return pose.get_components(components, {"FACE_LANDMARKS": face_contours, "POSE_LANDMARKS": body_no_face_no_hands})

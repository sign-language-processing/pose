from pose_format.pose import Pose
from pose_format.pose_header import (PoseHeader, PoseHeaderComponent,
                                     PoseHeaderDimensions)
from pose_format.utils.openpose import limbs_index, load_openpose_directory

BODY_POINTS = [
    "Nose", "LEye", "REye", "LEar", "REar", "LShoulder", "RShoulder", "LElbow", "RElbow", "LWrist", "RWrist", "LHip",
    "RHip", "LKnee", "RKnee", "LAnkle", "RAnkle", "UpperNeck", "HeadTop", "LBigToe", "LSmallToe", "LHeel", "RBigToe",
    "RSmallToe", "RHeel"
]
LEFT_HAND_POINTS = [
    "LThumb1CMC", "LThumb2Knuckles", "LThumb3IP", "LThumb4FingerTip", "LIndex1Knuckles", "LIndex2PIP", "LIndex3DIP",
    "LIndex4FingerTip", "LMiddle1Knuckles", "LMiddle2PIP", "LMiddle3DIP", "LMiddle4FingerTip", "LRing1Knuckles",
    "LRing2PIP", "LRing3DIP", "LRing4FingerTip", "LPinky1Knuckles", "LPinky2PIP", "LPinky3DIP", "LPinky4FingerTip"
]
RIGHT_HAND_POINTS = [
    "RThumb1CMC", "RThumb2Knuckles", "RThumb3IP", "RThumb4FingerTip", "RIndex1Knuckles", "RIndex2PIP", "RIndex3DIP",
    "RIndex4FingerTip", "RMiddle1Knuckles", "RMiddle2PIP", "RMiddle3DIP", "RMiddle4FingerTip", "RRing1Knuckles",
    "RRing2PIP", "RRing3DIP", "RRing4FingerTip", "RPinky1Knuckles", "RPinky2PIP", "RPinky3DIP", "RPinky4FingerTip"
]
FACE_POINTS = [
    "FaceContour0", "FaceContour1", "FaceContour2", "FaceContour3", "FaceContour4", "FaceContour5", "FaceContour6",
    "FaceContour7", "FaceContour8", "FaceContour9", "FaceContour10", "FaceContour11", "FaceContour12", "FaceContour13",
    "FaceContour14", "FaceContour15", "FaceContour16", "REyeBrow0", "REyeBrow1", "REyeBrow2", "REyeBrow3", "REyeBrow4",
    "LEyeBrow4", "LEyeBrow3", "LEyeBrow2", "LEyeBrow1", "LEyeBrow0", "NoseUpper0", "NoseUpper1", "NoseUpper2",
    "NoseUpper3", "NoseLower0", "NoseLower1", "NoseLower2", "NoseLower3", "NoseLower4", "REye0", "REye1", "REye2",
    "REye3", "REye4", "REye5", "LEye0", "LEye1", "LEye2", "LEye3", "LEye4", "LEye5", "OMouth0", "OMouth1", "OMouth2",
    "OMouth3", "OMouth4", "OMouth5", "OMouth6", "OMouth7", "OMouth8", "OMouth9", "OMouth10", "OMouth11", "IMouth0",
    "IMouth1", "IMouth2", "IMouth3", "IMouth4", "IMouth5", "IMouth6", "IMouth7", "RPupil", "LPupil"
]

BODY_135_POINTS = BODY_POINTS + LEFT_HAND_POINTS + RIGHT_HAND_POINTS + FACE_POINTS

BODY_LIMBS = [('RShoulder', 'LShoulder'), ('RShoulder', 'RElbow'), ('RElbow', 'RWrist'), ('LShoulder', 'LElbow'),
              ('LElbow', 'LWrist'), ('Nose', 'LEye'), ('Nose', 'REye'), ('Nose', 'LEar'), ('Nose', 'REar'),
              ('RHip', 'LHip'), ('RHip', 'RShoulder'), ('LHip', 'LShoulder'), ('RHip', 'RKnee'), ('RKnee', 'RAnkle'),
              ('LHip', 'LKnee'), ('LKnee', 'LAnkle'), ('RAnkle', 'RHeel'), ('RAnkle', 'RBigToe'),
              ('RBigToe', 'RSmallToe'), ('LAnkle', 'LHeel'), ('LAnkle', 'LBigToe'), ('LBigToe', 'LSmallToe')]

ABSTRACT_HAND_LIMBS = [
    ("LWrist", "LThumb1CMC"),
    ("LWrist", "LIndex1Knuckles"),
    ("LWrist", "LMiddle1Knuckles"),
    ("LWrist", "LRing1Knuckles"),
    ("LWrist", "LPinky1Knuckles"),  # Base
    ("LThumb1CMC", "LThumb2Knuckles"),
    ("LThumb2Knuckles", "LThumb3IP"),
    ("LThumb3IP", "LThumb4FingerTip"),  # Thumb
    ("LIndex1Knuckles", "LIndex2PIP"),
    ("LIndex2PIP", "LIndex3DIP"),
    ("LIndex3DIP", "LIndex4FingerTip"),  # Index
    ("LMiddle1Knuckles", "LMiddle2PIP"),
    ("LMiddle2PIP", "LMiddle3DIP"),
    ("LMiddle3DIP", "LMiddle4FingerTip"),  # Middle
    ("LRing1Knuckles", "LRing2PIP"),
    ("LRing2PIP", "LRing3DIP"),
    ("LRing3DIP", "LRing4FingerTip"),  # Ring
    ("LPinky1Knuckles", "LPinky2PIP"),
    ("LPinky2PIP", "LPinky3DIP"),
    ("LPinky3DIP", "LPinky4FingerTip"),  # Pinky
]

HAND_LIMBS = [(hand + l1[1:], hand + l2[1:]) for l1, l2 in ABSTRACT_HAND_LIMBS for hand in ["L", "R"]]

FACE_LIMBS = [('FaceContour8', 'FaceContour7'), ('FaceContour7', 'FaceContour6'), ('FaceContour6', 'FaceContour5'),
              ('FaceContour5', 'FaceContour4'), ('FaceContour4', 'FaceContour3'), ('FaceContour3', 'FaceContour2'),
              ('FaceContour2', 'FaceContour1'), ('FaceContour1', 'FaceContour0'), ('FaceContour8', 'FaceContour9'),
              ('FaceContour9', 'FaceContour10'), ('FaceContour10', 'FaceContour11'), ('FaceContour11', 'FaceContour12'),
              ('FaceContour12', 'FaceContour13'), ('FaceContour13', 'FaceContour14'),
              ('FaceContour14', 'FaceContour15'), ('FaceContour15', 'FaceContour16'), ('OMouth0', 'OMouth1'),
              ('OMouth1', 'OMouth2'), ('OMouth2', 'OMouth3'), ('OMouth3', 'OMouth4'), ('OMouth4', 'OMouth5'),
              ('OMouth5', 'OMouth6'), ('OMouth6', 'OMouth7'), ('OMouth7', 'OMouth8'), ('OMouth8', 'OMouth9'),
              ('OMouth9', 'OMouth10'), ('OMouth10', 'OMouth11'), ('OMouth11', 'OMouth0'), ('IMouth0', 'IMouth1'),
              ('IMouth1', 'IMouth2'), ('IMouth2', 'IMouth3'), ('IMouth3', 'IMouth4'), ('IMouth4', 'IMouth5'),
              ('IMouth5', 'IMouth6'), ('IMouth6', 'IMouth7'), ('IMouth7', 'IMouth0'), ('NoseUpper0', 'NoseUpper1'),
              ('NoseUpper1', 'NoseUpper2'), ('NoseUpper2', 'NoseUpper3'), ('NoseUpper3', 'NoseLower0'),
              ('NoseLower0', 'NoseLower1'), ('NoseLower1', 'NoseLower2'), ('NoseLower2', 'NoseLower3'),
              ('NoseLower3', 'NoseLower4'), ('NoseUpper3', 'NoseLower2'), ('REyeBrow0', 'REyeBrow1'),
              ('REyeBrow1', 'REyeBrow2'), ('REyeBrow2', 'REyeBrow3'), ('REyeBrow3', 'REyeBrow4'),
              ('LEyeBrow4', 'LEyeBrow3'), ('LEyeBrow3', 'LEyeBrow2'), ('LEyeBrow2', 'LEyeBrow1'),
              ('LEyeBrow1', 'LEyeBrow0'), ('REye0', 'REye1'), ('REye1', 'REye2'), ('REye2', 'REye3'),
              ('REye3', 'REye4'), ('REye4', 'REye5'), ('REye5', 'REye0'), ('LEye0', 'LEye1'), ('LEye1', 'LEye2'),
              ('LEye2', 'LEye3'), ('LEye3', 'LEye4'), ('LEye4', 'LEye5'), ('LEye5', 'LEye0')]

BODY_135_LIMBS = BODY_LIMBS + HAND_LIMBS + FACE_LIMBS

OpenPose_Components = [
    PoseHeaderComponent(name="BODY_135",
                        points=BODY_135_POINTS,
                        limbs=limbs_index(BODY_135_LIMBS, BODY_135_POINTS),
                        colors=[(255, 0, 0)],
                        point_format="XYC")
]


def load_openpose_135_directory(*args, **kwargs) -> Pose:
    """
    Loads OpnePose data from a directory and returns a Pose object.

    The function reads Openpose data and modifies body data and confidence to contain only first 135 components.
    It then updates header components to OpenPose components.

    Parameters
    ----------
    *args : 
        Variable length argument list.
    **kwargs :
        arbitrary keyword arguments.

    Returns
    -------
    Pose
        modified Pose object with body data and confidence from the first 135 components

    Note
    ----
    The function assumes that the input directory contains "OpenPose data" compatible with the Pose data structure, 
    and body data and confidence matrices must have at least 135 components, no less! 
    """
    pose = load_openpose_directory(*args, **kwargs)

    pose.body.data = pose.body.data[:, :, :135, :]
    pose.body.confidence = pose.body.confidence[:, :, :135]
    pose.header.components = OpenPose_Components

    return pose


if __name__ == "__main__":
    dimensions = PoseHeaderDimensions(width=512, height=512, depth=0)
    header = PoseHeader(version=0.2, dimensions=dimensions, components=OpenPose_Components)

    with open(
            "/home/nlp/amit/sign-language/sign-language-datasets/sign_language_datasets/datasets/autsl/openpose_135.poseheader",
            "wb") as f:
        header.write(f)

from typing import List

import numpy as np

from pose_format import Pose, PoseHeader
from pose_format.numpy import NumPyPoseBody
from pose_format.pose_header import PoseHeaderComponent
from pose_format.pose_visualizer import PoseVisualizer
from pose_format.utils.holistic import HAND_POINTS as HOLISTIC_HAND_POINTS
from pose_format.utils.holistic import holistic_components
from pose_format.utils.openpose import HAND_POINTS as OPENPOSE_HAND_POINTS

LEFT_HAND_MAP = [(("hand_left_keypoints_2d", k1), ("LEFT_HAND_LANDMARKS", k2))
                 for k1, k2 in zip(OPENPOSE_HAND_POINTS, HOLISTIC_HAND_POINTS)]
RIGHT_HAND_MAP = [(("hand_right_keypoints_2d", k1), ("RIGHT_HAND_LANDMARKS", k2))
                  for k1, k2 in zip(OPENPOSE_HAND_POINTS, HOLISTIC_HAND_POINTS)]
BODY_MAP = [
    {("pose_keypoints_2d", "Nose"), ("POSE_LANDMARKS", "NOSE")},
    {("pose_keypoints_2d", "Neck"), ("POSE_LANDMARKS", ("RIGHT_SHOULDER", "LEFT_SHOULDER"))},
    {("pose_keypoints_2d", "RShoulder"), ("POSE_LANDMARKS", "RIGHT_SHOULDER")},
    {("pose_keypoints_2d", "RElbow"), ("POSE_LANDMARKS", "RIGHT_ELBOW")},
    {("pose_keypoints_2d", "RWrist"), ("POSE_LANDMARKS", "RIGHT_WRIST")},
    {("pose_keypoints_2d", "LShoulder"), ("POSE_LANDMARKS", "LEFT_SHOULDER")},
    {("pose_keypoints_2d", "LElbow"), ("POSE_LANDMARKS", "LEFT_ELBOW")},
    {("pose_keypoints_2d", "LWrist"), ("POSE_LANDMARKS", "LEFT_WRIST")},
    {("pose_keypoints_2d", "MidHip"), ("POSE_LANDMARKS", ("RIGHT_HIP", "LEFT_HIP"))},
    {("pose_keypoints_2d", "RHip"), ("POSE_LANDMARKS", "RIGHT_HIP")},
    {("pose_keypoints_2d", "RKnee"), ("POSE_LANDMARKS", "RIGHT_KNEE")},
    {("pose_keypoints_2d", "RAnkle"), ("POSE_LANDMARKS", "RIGHT_ANKLE")},
    {("pose_keypoints_2d", "LHip"), ("POSE_LANDMARKS", "LEFT_HIP")},
    {("pose_keypoints_2d", "LKnee"), ("POSE_LANDMARKS", "LEFT_KNEE")},
    {("pose_keypoints_2d", "LAnkle"), ("POSE_LANDMARKS", "LEFT_ANKLE")},
    {("pose_keypoints_2d", "REye"), ("POSE_LANDMARKS", "RIGHT_EYE")},
    {("pose_keypoints_2d", "LEye"), ("POSE_LANDMARKS", "LEFT_EYE")},
    {("pose_keypoints_2d", "REar"), ("POSE_LANDMARKS", "RIGHT_EAR")},
    {("pose_keypoints_2d", "LEar"), ("POSE_LANDMARKS", "LEFT_EAR")},
    {("pose_keypoints_2d", "LHeel"), ("POSE_LANDMARKS", "LEFT_HEEL")},
    {("pose_keypoints_2d", "RHeel"), ("POSE_LANDMARKS", "RIGHT_HEEL")},
]

FACE_MAP = [
    # face border
    (("face_keypoints_2d", "FB_0"), ("FACE_LANDMARKS", "127")),
    (("face_keypoints_2d", "FB_1"), ("FACE_LANDMARKS", "234")),
    (("face_keypoints_2d", "FB_2"), ("FACE_LANDMARKS", "93")),
    (("face_keypoints_2d", "FB_3"), ("FACE_LANDMARKS", "132")),
    (("face_keypoints_2d", "FB_4"), ("FACE_LANDMARKS", "58")),
    (("face_keypoints_2d", "FB_5"), ("FACE_LANDMARKS", "172")),
    (("face_keypoints_2d", "FB_6"), ("FACE_LANDMARKS", "136")),
    (("face_keypoints_2d", "FB_7"), ("FACE_LANDMARKS", "149")),
    (("face_keypoints_2d", "FB_8"), ("FACE_LANDMARKS", "152")),
    (("face_keypoints_2d", "FB_9"), ("FACE_LANDMARKS", "378")),
    (("face_keypoints_2d", "FB_10"), ("FACE_LANDMARKS", "365")),
    (("face_keypoints_2d", "FB_11"), ("FACE_LANDMARKS", "397")),
    (("face_keypoints_2d", "FB_12"), ("FACE_LANDMARKS", "288")),
    (("face_keypoints_2d", "FB_13"), ("FACE_LANDMARKS", "361")),
    (("face_keypoints_2d", "FB_14"), ("FACE_LANDMARKS", "323")),
    (("face_keypoints_2d", "FB_15"), ("FACE_LANDMARKS", "454")),
    (("face_keypoints_2d", "FB_16"), ("FACE_LANDMARKS", "356")),

    ("interpolate", {"avg_between": (("FACE_LANDMARKS", "136"), ("FACE_LANDMARKS", "149")), "avg_for": ("FACE_LANDMARKS", "150")}),
    ("interpolate", {"avg_between": (("FACE_LANDMARKS", "149"), ("FACE_LANDMARKS", "152")), "avg_for": ("FACE_LANDMARKS", "176")}),
    ("interpolate", {"avg_between": (("FACE_LANDMARKS", "149"), ("FACE_LANDMARKS", "152")), "avg_for": ("FACE_LANDMARKS", "148")}),
    ("interpolate", {"avg_between": (("FACE_LANDMARKS", "152"), ("FACE_LANDMARKS", "378")), "avg_for": ("FACE_LANDMARKS", "377")}),
    ("interpolate", {"avg_between": (("FACE_LANDMARKS", "152"), ("FACE_LANDMARKS", "152")), "avg_for": ("FACE_LANDMARKS", "400")}),
    ("interpolate", {"avg_between": (("FACE_LANDMARKS", "378"), ("FACE_LANDMARKS", "365")), "avg_for": ("FACE_LANDMARKS", "379")}),

    # Right eye
    (("face_keypoints_2d", "FE_42"), ("FACE_LANDMARKS", "362")),
    (("face_keypoints_2d", "FE_43"), ("FACE_LANDMARKS", "385")),
    (("face_keypoints_2d", "FE_44"), ("FACE_LANDMARKS", "387")),
    (("face_keypoints_2d", "FE_45"), ("FACE_LANDMARKS", "263")),
    (("face_keypoints_2d", "FE_46"), ("FACE_LANDMARKS", "373")),
    (("face_keypoints_2d", "FE_47"), ("FACE_LANDMARKS", "380")),

    ("interpolate", {"avg_between": (("FACE_LANDMARKS", "362"), ("FACE_LANDMARKS", "385")), "avg_for": ("FACE_LANDMARKS", "398")}),
    ("interpolate", {"avg_between": (("FACE_LANDMARKS", "362"), ("FACE_LANDMARKS", "385")), "avg_for": ("FACE_LANDMARKS", "384")}),
    ("interpolate", {"avg_between": (("FACE_LANDMARKS", "385"), ("FACE_LANDMARKS", "387")), "avg_for": ("FACE_LANDMARKS", "386")}),
    ("interpolate", {"avg_between": (("FACE_LANDMARKS", "387"), ("FACE_LANDMARKS", "263")), "avg_for": ("FACE_LANDMARKS", "388")}),
    ("interpolate", {"avg_between": (("FACE_LANDMARKS", "387"), ("FACE_LANDMARKS", "263")), "avg_for": ("FACE_LANDMARKS", "466")}),
    ("interpolate", {"avg_between": (("FACE_LANDMARKS", "263"), ("FACE_LANDMARKS", "373")), "avg_for": ("FACE_LANDMARKS", "249")}),
    ("interpolate", {"avg_between": (("FACE_LANDMARKS", "263"), ("FACE_LANDMARKS", "373")), "avg_for": ("FACE_LANDMARKS", "390")}),
    ("interpolate", {"avg_between": (("FACE_LANDMARKS", "373"), ("FACE_LANDMARKS", "380")), "avg_for": ("FACE_LANDMARKS", "374")}),
    ("interpolate", {"avg_between": (("FACE_LANDMARKS", "380"), ("FACE_LANDMARKS", "362")), "avg_for": ("FACE_LANDMARKS", "381")}),
    ("interpolate", {"avg_between": (("FACE_LANDMARKS", "380"), ("FACE_LANDMARKS", "362")), "avg_for": ("FACE_LANDMARKS", "382")}),

    # left eye
    (("face_keypoints_2d", "FE_36"), ("FACE_LANDMARKS", "33")),
    (("face_keypoints_2d", "FE_37"), ("FACE_LANDMARKS", "160")),
    (("face_keypoints_2d", "FE_38"), ("FACE_LANDMARKS", "158")),
    (("face_keypoints_2d", "FE_39"), ("FACE_LANDMARKS", "133")),
    (("face_keypoints_2d", "FE_40"), ("FACE_LANDMARKS", "153")),
    (("face_keypoints_2d", "FE_41"), ("FACE_LANDMARKS", "144")),

    ("interpolate", {"avg_between": (("FACE_LANDMARKS", "33"), ("FACE_LANDMARKS", "160")), "avg_for": ("FACE_LANDMARKS", "246")}),
    ("interpolate", {"avg_between": (("FACE_LANDMARKS", "33"), ("FACE_LANDMARKS", "160")), "avg_for": ("FACE_LANDMARKS", "161")}),
    ("interpolate", {"avg_between": (("FACE_LANDMARKS", "160"), ("FACE_LANDMARKS", "158")), "avg_for": ("FACE_LANDMARKS", "159")}),
    ("interpolate", {"avg_between": (("FACE_LANDMARKS", "158"), ("FACE_LANDMARKS", "133")), "avg_for": ("FACE_LANDMARKS", "157")}),
    ("interpolate", {"avg_between": (("FACE_LANDMARKS", "158"), ("FACE_LANDMARKS", "133")), "avg_for": ("FACE_LANDMARKS", "173")}),
    ("interpolate", {"avg_between": (("FACE_LANDMARKS", "133"), ("FACE_LANDMARKS", "153")), "avg_for": ("FACE_LANDMARKS", "155")}),
    ("interpolate", {"avg_between": (("FACE_LANDMARKS", "133"), ("FACE_LANDMARKS", "153")), "avg_for": ("FACE_LANDMARKS", "154")}),
    ("interpolate", {"avg_between": (("FACE_LANDMARKS", "153"), ("FACE_LANDMARKS", "144")), "avg_for": ("FACE_LANDMARKS", "145")}),
    ("interpolate", {"avg_between": (("FACE_LANDMARKS", "144"), ("FACE_LANDMARKS", "33")), "avg_for": ("FACE_LANDMARKS", "163")}),
    ("interpolate", {"avg_between": (("FACE_LANDMARKS", "144"), ("FACE_LANDMARKS", "33")), "avg_for": ("FACE_LANDMARKS", "7")}),

    # Nose
    (("face_keypoints_2d", "FN_27"), ("FACE_LANDMARKS", "168")),
    (("face_keypoints_2d", "FN_28"), ("FACE_LANDMARKS", "197")),
    (("face_keypoints_2d", "FN_29"), ("FACE_LANDMARKS", "5")),
    (("face_keypoints_2d", "FN_30"), ("FACE_LANDMARKS", "4")),
    ("interpolate", {"avg_between": (("FACE_LANDMARKS", "168"), ("FACE_LANDMARKS", "197")), "avg_for": ("FACE_LANDMARKS", "6")}),
    ("interpolate", {"avg_between": (("FACE_LANDMARKS", "197"), ("FACE_LANDMARKS", "5")), "avg_for": ("FACE_LANDMARKS", "195")}),

    (("face_keypoints_2d", "FN_31"), ("FACE_LANDMARKS", "219")),
    (("face_keypoints_2d", "FN_32"), ("FACE_LANDMARKS", "237")),
    (("face_keypoints_2d", "FN_33"), ("FACE_LANDMARKS", "1")),
    (("face_keypoints_2d", "FN_34"), ("FACE_LANDMARKS", "457")),
    (("face_keypoints_2d", "FN_35"), ("FACE_LANDMARKS", "439")),
    ("interpolate", {"avg_between": (("FACE_LANDMARKS", "219"), ("FACE_LANDMARKS", "237")), "avg_for": ("FACE_LANDMARKS", "218")}),
    ("interpolate", {"avg_between": (("FACE_LANDMARKS", "237"), ("FACE_LANDMARKS", "1")), "avg_for": ("FACE_LANDMARKS", "44")}),
    ("interpolate", {"avg_between": (("FACE_LANDMARKS", "1"), ("FACE_LANDMARKS", "457")), "avg_for": ("FACE_LANDMARKS", "274")}),
    ("interpolate", {"avg_between": (("FACE_LANDMARKS", "457"), ("FACE_LANDMARKS", "439")), "avg_for": ("FACE_LANDMARKS", "438")}),

    # Mouth
    (("face_keypoints_2d", "FLO_48"), ("FACE_LANDMARKS", "61")),
    (("face_keypoints_2d", "FLO_49"), ("FACE_LANDMARKS", "40")),
    (("face_keypoints_2d", "FLO_50"), ("FACE_LANDMARKS", "37")),
    (("face_keypoints_2d", "FLO_51"), ("FACE_LANDMARKS", "0")),
    (("face_keypoints_2d", "FLO_52"), ("FACE_LANDMARKS", "267")),
    (("face_keypoints_2d", "FLO_53"), ("FACE_LANDMARKS", "270")),
    (("face_keypoints_2d", "FLO_54"), ("FACE_LANDMARKS", "291")),
    (("face_keypoints_2d", "FLO_55"), ("FACE_LANDMARKS", "321")),
    (("face_keypoints_2d", "FLO_56"), ("FACE_LANDMARKS", "314")),
    (("face_keypoints_2d", "FLO_57"), ("FACE_LANDMARKS", "17")),
    (("face_keypoints_2d", "FLO_58"), ("FACE_LANDMARKS", "84")),
    (("face_keypoints_2d", "FLO_59"), ("FACE_LANDMARKS", "91")),
    ("interpolate", {"avg_between": (("FACE_LANDMARKS", "61"), ("FACE_LANDMARKS", "40")), "avg_for": ("FACE_LANDMARKS", "185")}),
    ("interpolate", {"avg_between": (("FACE_LANDMARKS", "40"), ("FACE_LANDMARKS", "37")), "avg_for": ("FACE_LANDMARKS", "39")}),
    ("interpolate", {"avg_between": (("FACE_LANDMARKS", "267"), ("FACE_LANDMARKS", "270")), "avg_for": ("FACE_LANDMARKS", "269")}),
    ("interpolate", {"avg_between": (("FACE_LANDMARKS", "270"), ("FACE_LANDMARKS", "291")), "avg_for": ("FACE_LANDMARKS", "409")}),
    ("interpolate", {"avg_between": (("FACE_LANDMARKS", "291"), ("FACE_LANDMARKS", "321")), "avg_for": ("FACE_LANDMARKS", "375")}),
    ("interpolate", {"avg_between": (("FACE_LANDMARKS", "321"), ("FACE_LANDMARKS", "314")), "avg_for": ("FACE_LANDMARKS", "405")}),
    ("interpolate", {"avg_between": (("FACE_LANDMARKS", "84"), ("FACE_LANDMARKS", "91")), "avg_for": ("FACE_LANDMARKS", "181")}),
    ("interpolate", {"avg_between": (("FACE_LANDMARKS", "91"), ("FACE_LANDMARKS", "61")), "avg_for": ("FACE_LANDMARKS", "146")}),

    (("face_keypoints_2d", "FLI_60"), ("FACE_LANDMARKS", "78")),
    (("face_keypoints_2d", "FLI_61"), ("FACE_LANDMARKS", "81")),
    (("face_keypoints_2d", "FLI_62"), ("FACE_LANDMARKS", "13")),
    (("face_keypoints_2d", "FLI_63"), ("FACE_LANDMARKS", "311")),
    (("face_keypoints_2d", "FLI_64"), ("FACE_LANDMARKS", "308")),
    (("face_keypoints_2d", "FLI_65"), ("FACE_LANDMARKS", "402")),
    (("face_keypoints_2d", "FLI_66"), ("FACE_LANDMARKS", "14")),
    (("face_keypoints_2d", "FLI_67"), ("FACE_LANDMARKS", "178")),
    ("interpolate", {"avg_between": (("FACE_LANDMARKS", "78"), ("FACE_LANDMARKS", "81")), "avg_for": ("FACE_LANDMARKS", "191")}),
    ("interpolate", {"avg_between": (("FACE_LANDMARKS", "78"), ("FACE_LANDMARKS", "81")), "avg_for": ("FACE_LANDMARKS", "80")}),
    ("interpolate", {"avg_between": (("FACE_LANDMARKS", "81"), ("FACE_LANDMARKS", "13")), "avg_for": ("FACE_LANDMARKS", "82")}),
    ("interpolate", {"avg_between": (("FACE_LANDMARKS", "13"), ("FACE_LANDMARKS", "311")), "avg_for": ("FACE_LANDMARKS", "312")}),
    ("interpolate", {"avg_between": (("FACE_LANDMARKS", "311"), ("FACE_LANDMARKS", "308")), "avg_for": ("FACE_LANDMARKS", "310")}),
    ("interpolate", {"avg_between": (("FACE_LANDMARKS", "311"), ("FACE_LANDMARKS", "308")), "avg_for": ("FACE_LANDMARKS", "415")}),
    ("interpolate", {"avg_between": (("FACE_LANDMARKS", "308"), ("FACE_LANDMARKS", "402")), "avg_for": ("FACE_LANDMARKS", "318")}),
    ("interpolate", {"avg_between": (("FACE_LANDMARKS", "308"), ("FACE_LANDMARKS", "402")), "avg_for": ("FACE_LANDMARKS", "324")}),
    ("interpolate", {"avg_between": (("FACE_LANDMARKS", "402"), ("FACE_LANDMARKS", "14")), "avg_for": ("FACE_LANDMARKS", "317")}),
    ("interpolate", {"avg_between": (("FACE_LANDMARKS", "14"), ("FACE_LANDMARKS", "178")), "avg_for": ("FACE_LANDMARKS", "87")}),
    ("interpolate", {"avg_between": (("FACE_LANDMARKS", "178"), ("FACE_LANDMARKS", "78")), "avg_for": ("FACE_LANDMARKS", "88")}),
    ("interpolate", {"avg_between": (("FACE_LANDMARKS", "178"), ("FACE_LANDMARKS", "78")), "avg_for": ("FACE_LANDMARKS", "95")}),

    # FEB
    (("face_keypoints_2d", "FEB_17"), ("FACE_LANDMARKS", "70")),
    (("face_keypoints_2d", "FEB_18"), ("FACE_LANDMARKS", "63")),
    (("face_keypoints_2d", "FEB_19"), ("FACE_LANDMARKS", "105")),
    (("face_keypoints_2d", "FEB_20"), ("FACE_LANDMARKS", "66")),
    (("face_keypoints_2d", "FEB_21"), ("FACE_LANDMARKS", "107")),

    (("face_keypoints_2d", "FEB_22"), ("FACE_LANDMARKS", "336")),
    (("face_keypoints_2d", "FEB_23"), ("FACE_LANDMARKS", "296")),
    (("face_keypoints_2d", "FEB_24"), ("FACE_LANDMARKS", "334")),
    (("face_keypoints_2d", "FEB_25"), ("FACE_LANDMARKS", "293")),
    (("face_keypoints_2d", "FEB_26"), ("FACE_LANDMARKS", "300")),
]

POSES_MAP = BODY_MAP + LEFT_HAND_MAP + RIGHT_HAND_MAP + FACE_MAP

def convert_pose(pose: Pose, pose_components: List[PoseHeaderComponent]) -> Pose:
    """
    converts the given pose to a new pose instance based on given pose components.

    Parameters
    ----------
    pose : Pose
        The initial pose object to convert
    pose_components : List[PoseHeaderComponent]
        the new set of pose components to define the pose structure

    Returns
    -------
    Pose
        Converted pose object
    """
    pose_header = PoseHeader(version=pose.header.version, dimensions=pose.header.dimensions, components=pose_components)

    base_shape = (pose.body.data.shape[0], pose.body.data.shape[1], pose_header.total_points())
    data = np.zeros(shape=(*base_shape, len(pose_components[0].format) - 1), dtype=np.float32)
    conf = np.zeros(shape=base_shape, dtype=np.float32)

    original_components = set([c.name for c in pose.header.components])
    new_components = set([c.name for c in pose_components])
    
    # Create a mapping
    mapping = {}
    interpolations = []
    for entry in POSES_MAP:
        if isinstance(entry, tuple) and entry[0] == "interpolate":
            # Handle detailed interpolation
            avg_between = entry[1]["avg_between"]
            avg_for = entry[1]["avg_for"]
            interpolations.append((avg_between, avg_for))
        else:
            original_point = None
            new_point = None
            for component, point in entry:
                if component in original_components:
                    original_point = (component, point)

                if component in new_components and isinstance(point, str):
                    new_point = (component, point)

            if original_point is not None and new_point is not None:
                mapping[new_point] = original_point

    dims = min(len(pose_header.components[0].format), len(pose.header.components[0].format)) - 1
    for (c1, p1), (c2, p2) in mapping.items():
        p2 = tuple([p2]) if isinstance(p2, str) else p2
        p2s = [pose.header._get_point_index(c2, p) for p in list(p2)]
        p1_index = pose_header._get_point_index(c1, p1)
        data[:, :, p1_index, :dims] = pose.body.data[:, :, p2s, :dims].mean(axis=2)
        conf[:, :, p1_index] = pose.body.confidence[:, :, p2s].mean(axis=2)

    for (comp_pair, target) in interpolations:
        (comp1, point1), (comp2, point2) = comp_pair
        target_comp, target_point = target

        index1 = pose_header._get_point_index(comp1, point1)
        index2 = pose_header._get_point_index(comp2, point2)
        target_index = pose_header._get_point_index(target_comp, target_point)

        data[:, :, target_index, :dims] = (data[:, :, index1, :dims] + data[:, :, index2, :dims]) / 2
        conf[:, :, target_index] = (conf[:, :, index1] + conf[:, :, index2]) / 2
    
    pose_body = NumPyPoseBody(fps=pose.body.fps, data=data, confidence=conf)

    return Pose(pose_header, pose_body)


def save_image(pose: Pose, name: str):
    """
    Saves visualized pose as an image with a given name

    Parameters
    ----------
    pose : Pose
        Pose to be visualized and saved
    name : str
        Name to save image to.
    """
    visualizer = PoseVisualizer(pose, thickness=1)
    frame = next(iter(visualizer.draw(background_color=(255, 255, 255))))
    visualizer.save_frame(name, frame)


if __name__ == "__main__":
    with open("sample-data/video/sample.pose", "rb") as f:
        original_pose = Pose.read(f.read(), NumPyPoseBody)
        original_pose.focus()

    conv_holistic = convert_pose(original_pose, holistic_components())
    conv_openpose1 = convert_pose(original_pose, original_pose.header.components)
    conv_openpose2 = convert_pose(conv_holistic, original_pose.header.components)

    save_image(original_pose, "original.png")
    save_image(conv_holistic, "conv_holistic.png")
    save_image(conv_openpose1, "conv_openpose1.png")
    save_image(conv_openpose2, "conv_openpose2.png")

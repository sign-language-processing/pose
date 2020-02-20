BODY_POINTS = ["Nose", "Neck", "RShoulder", "RElbow", "RWrist", "LShoulder", "LElbow", "LWrist", "MidHip",
               "RHip", "RKnee", "RAnkle", "LHip", "LKnee", "LAnkle", "REye", "LEye", "REar", "LEar", "LBigToe",
               "LSmallToe", "LHeel", "RBigToe", "RSmallToe", "RHeel"]

# Based on https://github.com/CMU-Perceptual-Computing-Lab/openpose/raw/master/doc/media/keypoints_pose_25.png
# Everything sprouts out of the neck
BODY_LIMBS = [
    # Body
    ("Neck", "RShoulder"),
    ("RShoulder", "RElbow"),
    ("RElbow", "RWrist"),
    ("Neck", "LShoulder"),
    ("LShoulder", "LElbow"),
    ("LElbow", "LWrist"),
    ("Neck", "MidHip"),
    # Face
    ("Nose", "LEye"),
    ("Nose", "REye"),
    ("Nose", "LEar"),
    ("Nose", "REar"),
    ("Neck", "Nose"),
    # Legs
    ("MidHip", "RHip"),
    ("RHip", "RKnee"),
    ("RKnee", "RAnkle"),
    ("MidHip", "LHip"),
    ("LHip", "LKnee"),
    ("LKnee", "LAnkle"),
    # Feet
    ("RAnkle", "RHeel"),
    ("RAnkle", "RBigToe"),
    ("RBigToe", "RSmallToe"),
    ("LAnkle", "LHeel"),
    ("LAnkle", "LBigToe"),
    ("LBigToe", "LSmallToe"),
]

#        8   12  16  20
#        |   |   |   |
#        7   11  15  19
#    4   |   |   |   |
#    |   6   10  14  18
#    3   |   |   |   |
#    |   5---9---13--17
#    2    \         /
#     \    \       /
#      1    \     /
#       \    \   /
#        ------0-

# Anatomy guide http://blog.handcare.org/blog/2017/10/26/anatomy-101-finger-joints/
HAND_POINTS = [
    "BASE",
    "T_STT", "T_BCMC", "T_MCP", "T_IP",  # Thumb
    "I_CMC", "I_MCP", "I_PIP", "I_DIP",  # Index
    "M_CMC", "M_MCP", "M_PIP", "M_DIP",  # Middle
    "R_CMC", "R_MCP", "R_PIP", "R_DIP",  # Ring
    "P_CMC", "P_MCP", "P_PIP", "P_DIP",  # Pinky
]

# Based on https://github.com/CMU-Perceptual-Computing-Lab/openpose/raw/master/doc/media/keypoints_hand.png
# Everything sprouts out of the base
HAND_LIMBS = [
    ("BASE", "T_STT"), ("BASE", "I_CMC"), ("BASE", "M_CMC"), ("BASE", "R_CMC"), ("BASE", "P_CMC"),  # Base
    ("T_STT", "T_BCMC"), ("T_BCMC", "T_MCP"), ("T_MCP", "T_IP"),  # Thumb
    ("I_CMC", "I_MCP"), ("I_MCP", "I_PIP"), ("I_PIP", "I_DIP"),  # Index
    ("M_CMC", "M_MCP"), ("M_MCP", "M_PIP"), ("M_PIP", "M_DIP"),  # Middle
    ("R_CMC", "R_MCP"), ("R_MCP", "R_PIP"), ("R_PIP", "R_DIP"),  # Ring
    ("P_CMC", "P_MCP"), ("P_MCP", "P_PIP"), ("P_PIP", "P_DIP"),  # Pinky
]

# Based on https://github.com/CMU-Perceptual-Computing-Lab/openpose/raw/master/doc/media/keypoints_face.png
# Border
FACE_BORDER_POINTS = ["FB_" + str(i) for i in range(17)]
FACE_BORDER_LIMBS_LEFT = [("FB_" + str(i), "FB_" + str(i - 1)) for i in reversed(range(1, 9))]
FACE_BORDER_LIMBS_RIGHT = [("FB_" + str(i), "FB_" + str(i + 1)) for i in range(8, 16)]

# Lips
FACE_OUTER_LIPS_POINTS = ["FLO_" + str(i) for i in range(48, 60)]
FACE_OUTER_LIPS_LIMBS = [("FLO_" + str(i), "FLO_" + str(i + 1)) for i in range(48, 59)] + [("FLO_59", "FLO_48")]
FACE_INNER_LIPS_POINTS = ["FLI_" + str(i) for i in range(60, 68)]
FACE_INNER_LIPS_LIMBS = [("FLI_" + str(i), "FLI_" + str(i + 1)) for i in range(60, 67)] + [("FLI_67", "FLI_60")]

# Nose
FACE_NOSE_POINTS = ["FN_" + str(i) for i in range(27, 36)]
FACE_NOSE_BRIDGE_LIMBS = [("FN_" + str(i), "FN_" + str(i + 1)) for i in range(27, 31)]
FACE_NOSE_HORIZONTAL_LIMBS = [("FN_" + str(i), "FN_" + str(i + 1)) for i in range(31, 35)]
FACE_NOSE_LIMBS = FACE_NOSE_BRIDGE_LIMBS + FACE_NOSE_HORIZONTAL_LIMBS + [("FN_30", "FN_33")]

# Eyebrows
FACE_EYE_POINTS = ["FE_" + str(i) for i in range(36, 48)]
FACE_EYE_LEFT_LIMBS = [("FE_" + str(i), "FE_" + str(i + 1)) for i in range(36, 41)] + [("FE_41", "FE_36")]
FACE_EYE_RIGHT_LIMBS = [("FE_" + str(i), "FE_" + str(i + 1)) for i in range(42, 47)] + [("FE_47", "FE_42")]
FACE_PUPILS_POINTS = ["FP_68", "FP_69"]

# Eyes
FACE_EYEBROWS_POINTS = ["FEB_" + str(i) for i in range(17, 27)]
FACE_EYEBROW_LEFT_LIMBS = [("FEB_" + str(i), "FEB_" + str(i + 1)) for i in range(17, 21)]
FACE_EYEBROW_RIGHT_LIMBS = [("FEB_" + str(i), "FEB_" + str(i + 1)) for i in range(22, 26)]

# Face points, in order
FACE_POINTS = FACE_BORDER_POINTS + FACE_EYEBROWS_POINTS + FACE_NOSE_POINTS + FACE_EYE_POINTS + FACE_OUTER_LIPS_POINTS + FACE_INNER_LIPS_POINTS + FACE_PUPILS_POINTS
FACE_LIMBS = FACE_BORDER_LIMBS_LEFT + FACE_BORDER_LIMBS_RIGHT + FACE_OUTER_LIPS_LIMBS + FACE_INNER_LIPS_LIMBS + FACE_NOSE_LIMBS + FACE_EYEBROW_LEFT_LIMBS + FACE_EYEBROW_RIGHT_LIMBS + FACE_EYE_LEFT_LIMBS + FACE_EYE_RIGHT_LIMBS



HAND_POINTS_COLOR = [
    [192, 0, 0],
    [192, 192, 0],
    [0, 192, 0],
    [0, 192, 192],
    [0, 0, 192],
    [127, 127, 127]
]

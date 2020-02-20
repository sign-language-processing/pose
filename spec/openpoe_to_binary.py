import json
from os import path, listdir

import math
import struct
from itertools import chain

from spec.utils.pose import BODY_POINTS, BODY_LIMBS, FACE_POINTS, FACE_LIMBS, HAND_POINTS, HAND_POINTS_COLOR, \
    HAND_LIMBS


def s2b(s: str) -> bytes:
    return bytes(s, 'utf8')

point2d = "fff"  # [float X][float Y][float Confidence]

hand_part = {
    "points": HAND_POINTS,
    "colors": [[math.floor(x + 35 * (i % 4)) for x in HAND_POINTS_COLOR[i // 4]] for i in range(-1, len(HAND_POINTS)-1)],
    "limbs": HAND_LIMBS,
    "point_format": "XYC",
    "format": point2d
}

parts = {
    "pose_keypoints_2d": {
        "points": BODY_POINTS,
        "colors": [[255, 0, 0]], # Red
        "limbs": BODY_LIMBS,
        "point_format": "XYC",
        "format": point2d,
    },
    "face_keypoints_2d": {
        "points": FACE_POINTS,
        "colors": [[128, 0, 0]], # Brown
        "limbs": FACE_LIMBS,
        "point_format": "XYC",
        "format": point2d,
    },
    "hand_left_keypoints_2d": hand_part,
    "hand_right_keypoints_2d": hand_part
}


def write_header(binary):
    binary.write(struct.pack("<f", 0))  # File version

    binary.write(struct.pack("<H", 1200))  # Width
    binary.write(struct.pack("<H", 1200))  # Height
    binary.write(struct.pack("<H", 0))  # Depth

    binary.write(struct.pack("<H", len(parts)))  # Number of parts

    # Add Parts
    for part, features in parts.items():
        binary.write(struct.pack("<%dsx" % len(part), s2b(part)))  # Write part name

        # Write component points format
        point_format = features["point_format"]
        binary.write(struct.pack("<%dsx" % len(point_format), s2b(point_format)))  # Write part name

        # Write component lengths
        lengths = len(features["points"]), len(features["limbs"]), len(features["colors"])
        binary.write(struct.pack("<HHH", *lengths))

        # Names of Points
        point_names = [s2b(n) for n in features["points"]]
        s_format = "".join(["%dsx" % len(p) for p in features["points"]])
        binary.write(struct.pack("<" + s_format, *point_names))

        # Write the indexes of the limbs
        limbs = [features["points"].index(p) for limb in features["limbs"] for p in limb]
        binary.write(struct.pack("<" + "H" * len(limbs), *limbs))

        # Write the colors
        colors = list(chain.from_iterable(features["colors"]))
        binary.write(struct.pack("<" + "H" * len(colors), *colors))

def write_body(binary, frame):
    binary.write(struct.pack("<H", len(frame["people"])))  # Write number of people

    for person in frame["people"]:
        binary.write(struct.pack("<h", person["person_id"][0]))

        for part, features in parts.items():
            p_format = features["format"] * len(features["points"])
            binary.write(struct.pack("<" + p_format, *person[part]))

# Get JSON File
data_dir = path.join(path.pardir, "sample-data")
json_dir = path.join(data_dir, "json")
json_files = list(listdir(json_dir))

frames = [json.load(open(path.join(json_dir, file_name), "r")) for file_name in json_files]

video = open(path.join(data_dir, "video", "sample.pose"), "wb")
write_header(video)
video.write(struct.pack("<H", len(frames)))  # Number of frames


for f_name, data in zip(json_files, frames):
    # Open Binary File
    bin_f_name = ".".join(f_name.split(".")[:-1]) + ".pose"
    img = open(path.join(data_dir, "imgs", bin_f_name), "wb")

    # Create File Header
    write_header(img)

    # Write Body
    img.write(struct.pack("<H", 1))  # Number of frames

    write_body(img, data)
    write_body(video, data)

    img.flush()
    img.close()

video.flush()
video.close()

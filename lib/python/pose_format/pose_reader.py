import struct
from collections import namedtuple
from typing import Tuple

from dataclasses import dataclass
import numpy as np

from .pose import Pose, Points
from .utils.openpose import OpenPose_Components


@dataclass
class Unpack:
    short = struct.Struct("<h")
    ushort = struct.Struct("<H")

    double_ushort = struct.Struct("<HH")
    triple_ushort = struct.Struct("<HHH")

    metadata = struct.Struct("<fHHHH")



class PoseReader:
    def __init__(self, buffer):
        self.buffer = buffer
        self.read_offset = 0

    def read(self) -> Pose:
        header = self.read_header()
        body = self.read_body(header)

        return Pose(header, body, body["fps"])

    def unpack_f(self, s_format: str):
        if not hasattr(Unpack, s_format):
            le_format = "<" + s_format
            setattr(Unpack, s_format, struct.Struct(le_format))

        return self.unpack(getattr(Unpack, s_format))

    def unpack_numpy(self, s_format: str, shape: Tuple):
        le_format = "<" + s_format
        arr = np.ndarray(shape, le_format, self.buffer, self.read_offset)
        self.read_offset += struct.calcsize(le_format) * np.prod(shape)
        return arr

    def unpack(self, s: struct.Struct):
        unpack = s.unpack_from(self.buffer, self.read_offset)
        self.read_offset += s.size
        if len(unpack) == 1:
            return unpack[0]
        return unpack

    def unpack_str(self):
        length = self.unpack(Unpack.ushort)
        bytes_ = self.unpack_f("%ds" % length)
        return bytes_.decode("utf-8")

    def read_header(self):
        # Metadata
        version, width, height, depth, _components = self.unpack(Unpack.metadata)

        components = {}
        for _ in range(_components):
            name = self.unpack_str()
            point_format = self.unpack_str()
            _points, _limbs, _colors = self.unpack(Unpack.triple_ushort)
            points = [self.unpack_str() for _ in range(_points)]
            limb_indexes = [self.unpack(Unpack.double_ushort) for _ in range(_limbs)]
            limbs = [[points[i] for i in idxs] for idxs in limb_indexes]
            colors = self.unpack_numpy("H", (_colors, 3))

            components[name] = {
                "points": points,
                "colors": colors,
                "limb_indexes": limb_indexes,
                "limbs": limbs,
                "point_format": point_format
            }

        header = {
            "version": version,
            "width": width,
            "height": height,
            "depth": depth,
            "components": components
        }

        return header

    def read_body(self, header):
        fps, _frames = self.unpack(Unpack.double_ushort)
        frames = []
        for _ in range(_frames):
            _people = self.unpack(Unpack.ushort)
            people = []
            for _ in range(_people):
                person_id = self.unpack(Unpack.short)
                person = {"id": person_id}
                for name, features in header["components"].items():
                    points = np.array(self.unpack_numpy("f", (len(features["points"]), len(features["point_format"]))))
                    dimensions, confidence = np.split(points, [-1], axis=1) # TODO This split is costly

                    person[name] = Points(dimensions=dimensions, confidence=confidence.flatten().tolist())
                people.append(person)
            frames.append({"people": people})

        return {
            "fps": fps,
            "frames": frames
        }

    @staticmethod
    def from_openpose_json(json, width=1000, height=1000, depth=0):
        header = {
            "version": 0,
            "width": width,
            "height": height,
            "depth": depth,
            "components": OpenPose_Components,
        }

        frame = {"people": []}
        body = {
            "fps": 0,
            "frames": [frame]
        }

        for person in json["people"]:
            p = {"id": person["person_id"][0]}

            for part, features in header["components"].items():
                numbers = person[part]
                point = features["point_format"]
                p[part] = [{k: numbers[i + j] for i, k in enumerate(point)} for j in range(0, len(numbers), len(point))]

            frame["people"].append(p)

        return Pose(header, body)


if __name__ == "__main__":
    from time import time
    from json import load

    # json = load(open("../../../sample-data/json/video_000000000000_keypoints.json", "r"))
    # p = Pose.from_openpose_json(json)
    # p.focus_pose()
    # p.save("test.pose")

    start = time()
    for i in range(10):
        json = load(open("../../../sample-data/json/video_000000000000_keypoints.json", "r"))
        p = PoseReader.from_openpose_json(json)
    print(time() - start)

    start = time()
    for i in range(10):
        buffer = open("test.pose", "rb").read()
        PoseReader(buffer).read()

    print(time() - start)

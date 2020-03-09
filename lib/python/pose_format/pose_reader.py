import struct

from .pose import Pose
from .utils.openpose import OpenPose_Components


class PoseReader:
    def __init__(self, buffer):
        self.buffer = buffer
        self.read_offset = 0

    def read(self) -> Pose:
        header = self.read_header()
        body = self.read_body(header)

        return Pose(header, body, body["fps"])

    def unpack(self, s_format: str):
        le_format = "<" + s_format
        unpack = struct.unpack_from(le_format, self.buffer, self.read_offset)
        self.read_offset += struct.calcsize(le_format)
        if len(s_format) == 1:
            return unpack[0]
        return unpack

    def unpack_str(self):
        length = self.unpack("H")
        return self.unpack("%ds" % length)[0].decode("utf-8")

    def read_header(self):
        # Metadata
        version, width, height, depth, _components = self.unpack("fHHHH")

        components = {}
        for _ in range(_components):
            name = self.unpack_str()
            point_format = self.unpack_str()
            _points, _limbs, _colors = self.unpack("HHH")
            points = [self.unpack_str() for _ in range(_points)]
            limbs = [[points[i] for i in self.unpack("HH")] for _ in range(_limbs)]
            colors = [self.unpack("HHH") for _ in range(_colors)]
            components[name] = {
                "points": points,
                "colors": colors,
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
        fps, _frames = self.unpack("HH")
        frames = []
        for _ in range(_frames):
            _people = self.unpack("H")
            people = []
            for _ in range(_people):
                person_id = self.unpack("h")
                person = {"id": person_id}
                for name, features in header["components"].items():
                    point_format = "f" * len(features["point_format"])
                    points = [self.unpack(point_format) for _ in range(len(features["points"]))]
                    person[name] = [{k: p[i] for i, k in enumerate(features["point_format"])} for p in points]
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
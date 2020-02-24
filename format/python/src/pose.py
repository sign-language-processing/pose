import copy
import struct
import timeit
from collections import defaultdict
from itertools import chain
from json import load
from time import time
from typing import Tuple

import math
import numpy as np
from imgaug import Keypoint, KeypointsOnImage
from imgaug.augmenters import Augmenter
from scipy.interpolate import interp1d


def s2b(s: str) -> bytes:
    return bytes(s, 'utf8')


POINT_DIMENSIONS = ["X", "Y", "Z"]


class Pose:

    def __init__(self, header, body=None, fps=24):
        self.header = header
        self.body = body
        if self.body is None:
            self.body = {"frames": []}
        self.body["fps"] = fps

    def add_frame(self, frame):
        self.body["frames"].append(frame)

    @staticmethod
    def point_to_numpy(point):
        return np.array([point[p] for p in POINT_DIMENSIONS if p in point])

    def iterate_points(self, callback):
        for frame in self.body["frames"]:
            for person in frame["people"]:
                for component in self.header["components"].keys():
                    for point in person[component]:
                        if point["C"] > 0:
                            for k in POINT_DIMENSIONS:
                                if k in point:
                                    callback(point, k)

    def focus_pose(self):
        """
        Gets the pose to start at (0,0) and have dimensions as big as needed
        """
        frames = self.body["frames"]
        if len(frames) == 0:
            return

        mins = {p: math.inf for p in POINT_DIMENSIONS}
        maxs = {p: 0 for p in POINT_DIMENSIONS}

        def set_min(point, k):
            mins[k] = min(mins[k], point[k])

        self.iterate_points(set_min)

        def set_max(point, k):
            maxs[k] = max(maxs[k], point[k])

        self.iterate_points(set_max)

        def translate_point(point, k):
            point[k] -= mins[k]

        self.iterate_points(translate_point)

        self.header["width"] = math.ceil(max(maxs["X"] - mins["X"], 0))
        self.header["height"] = math.ceil(max(maxs["Y"] - mins["Y"], 0))
        self.header["depth"] = math.ceil(max(maxs["Z"] - mins["Z"], 0))

    def normalize(self, dist_p1: Tuple[str, int], dist_p2: Tuple[str, int], scale_factor: float = 1):
        frames = self.body["frames"]
        if len(frames) == 0:
            return

        # 1. Calculate the distance between p1 and p2
        distances = []
        for frame in frames:
            for person in frame["people"]:
                p1 = Pose.point_to_numpy(person[dist_p1[0]][dist_p1[1]])
                p2 = Pose.point_to_numpy(person[dist_p2[0]][dist_p2[1]])
                distances.append(np.linalg.norm(p1 - p2))

        # 2. scale all points to dist/scale
        scale = scale_factor / np.mean(distances)

        def scale_point(point, k):
            point[k] *= scale

        self.iterate_points(scale_point)

        # 3. Shift all points to be positive, starting from 0
        self.focus_pose()

    def augment2d(self, augmenter: Augmenter):
        header = copy.deepcopy(self.header)
        body = copy.deepcopy(self.body)

        keypoints = []
        for frame in body["frames"]:
            for person in frame["people"]:
                for component in self.header["components"].keys():
                    for point in person[component]:
                        if point["C"] > 0:
                            keypoints.append(Keypoint(x=point["X"], y=point["Y"]))

        kps = KeypointsOnImage(keypoints, shape=(header["height"], header["width"]))
        kps_aug = augmenter(keypoints=kps)  # Augment keypoints

        for frame in body["frames"]:
            for person in frame["people"]:
                for component in self.header["components"].keys():
                    for point in person[component]:
                        if point["C"] > 0:
                            keypoint = kps_aug.keypoints.pop(0)
                            point["X"] = keypoint.x
                            point["Y"] = keypoint.y

        return Pose(header, body, fps=body["fps"])

    def interpolate_fps(self, fps, kind='cubic'):
        frames = self.body["frames"]

        new_frames_len = math.ceil((len(frames) - 1) * fps / self.body["fps"]) + 1
        new_x = [i / fps for i in range(new_frames_len)]

        keypoints = defaultdict(lambda: defaultdict(lambda: {"x": [], "y": []}))
        for i, frame in enumerate(frames):
            for person in frame["people"]:
                for name, component in self.header["components"].items():
                    for j, point in enumerate(person[name]):
                        if point["C"] > 0:
                            for d in component["point_format"]:
                                if d in point:
                                    k = (name, j, d)
                                    keypoints[person["id"]][k]["x"].append(i / self.body["fps"])
                                    keypoints[person["id"]][k]["y"].append(point[d])


        intep = {p: {k: interp1d(v["x"], v["y"], kind=kind)(new_x) for k, v in data.items()}
                 for p, data in keypoints.items()}

        new_frames = []
        for i in range(len(new_x)):
            people = []
            for p, new_data in intep.items():
                person = {"id": p}
                for name, component in self.header["components"].items():
                    person[name] = [{d: 0 for d in component["point_format"]} for _ in range(len(component["points"]))]

                for (name, point_i, d), new_y in new_data.items():
                    person[name][point_i][d] = new_y[i]

                people.append(person)
            new_frames.append({"people": people})

        self.body["frames"] = new_frames
        self.body["fps"] = fps

    def save_header(self, f):
        f.write(struct.pack("<f", self.header["version"]))  # File version

        f.write(struct.pack("<H", self.header["width"]))  # Width
        f.write(struct.pack("<H", self.header["height"]))  # Height
        f.write(struct.pack("<H", self.header["depth"]))  # Depth

        f.write(struct.pack("<H", len(self.header["components"])))  # Number of components

        # Add components
        for part, features in self.header["components"].items():
            f.write(struct.pack("<H%ds" % len(part), len(part), s2b(part)))  # Write part name

            # Write component points format
            point_format = features["point_format"]
            f.write(struct.pack("<H%ds" % len(point_format), len(point_format), s2b(point_format)))  # Write part name

            # Write component lengths
            lengths = len(features["points"]), len(features["limbs"]), len(features["colors"])
            f.write(struct.pack("<HHH", *lengths))

            # Names of Points
            point_names = [[len(n), s2b(n)] for n in features["points"]]
            s_format = "".join(["H%ds" % len(p) for p in features["points"]])
            f.write(struct.pack("<" + s_format, *chain.from_iterable(point_names)))

            # Write the indexes of the limbs
            limbs = [features["points"].index(p) for limb in features["limbs"] for p in limb]
            f.write(struct.pack("<" + "H" * len(limbs), *limbs))

            # Write the colors
            colors = list(chain.from_iterable(features["colors"]))
            f.write(struct.pack("<" + "H" * len(colors), *colors))

    def save_body(self, f):
        f.write(struct.pack("<H", self.body["fps"]))  # Write FPS
        f.write(struct.pack("<H", len(self.body["frames"])))  # Write number of frames
        for frame in self.body["frames"]:
            self.save_frame(f, frame)

    def save_frame(self, f, frame):
        f.write(struct.pack("<H", len(frame["people"])))  # Write number of people

        for person in frame["people"]:
            f.write(struct.pack("<h", person["id"]))

            for part, features in self.header["components"].items():
                points = list(chain.from_iterable([p.values() for p in person[part]]))
                p_format = "f" * len(features["point_format"]) * len(features["points"])
                f.write(struct.pack("<" + p_format, *points))

    def save(self, f_name):
        f = open(f_name, "wb")
        self.save_header(f)
        self.save_body(f)
        f.flush()
        f.close()


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
        from format.python.src.utils.openpose import OpenPose_Components

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

import copy
import struct
from collections import defaultdict
from itertools import chain
from typing import Tuple, Iterator, List

import math
import numpy as np
from imgaug import Keypoint, KeypointsOnImage
from imgaug.augmenters import Augmenter
from jsonlines import jsonlines
from scipy.interpolate import interp1d

from lib.python.pose_format.utils.fast_math import distance, angle


def s2b(s: str) -> bytes:
    return bytes(s, 'utf8')


POINT_DIMENSIONS = ["X", "Y", "Z"]

POINT_AGGREGATORS = {
    "distance": distance,
    "angle": angle,
}


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

    def iterate_points_dimensions(self, callback):
        def iterate_dimensions(point):
            for k in POINT_DIMENSIONS:
                if k in point:
                    callback(point, k)

        return self.iterate_points(iterate_dimensions)

    def iterate_points(self, callback):
        for frame in self.body["frames"]:
            for person in frame["people"]:
                for component in self.header["components"].keys():
                    for point in person[component]:
                        if point["C"] > 0:
                            callback(point)

    def focus_pose(self):
        """
        Gets the pose to start at (0,0) and have dimensions as big as needed
        """
        frames = self.body["frames"]
        if len(frames) == 0:
            return

        mins = {p: math.inf for p in POINT_DIMENSIONS}
        maxs = {p: 0 for p in POINT_DIMENSIONS}

        def set_min_max(point, k):
            mins[k] = min(mins[k], point[k])
            maxs[k] = max(maxs[k], point[k])

        self.iterate_points_dimensions(set_min_max)

        def translate_point(point, k):
            point[k] -= mins[k]

        self.iterate_points_dimensions(translate_point)

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
                distances.append(distance(p1, p2))

        # 2. scale all points to dist/scale
        scale = scale_factor / np.mean(distances)

        def scale_point(point, k):
            point[k] *= scale

        self.iterate_points_dimensions(scale_point)

        # 3. Shift all points to be positive, starting from 0
        self.focus_pose()

    def augment2d(self, augmenter: Augmenter):
        keypoints = []
        self.iterate_points(lambda point: keypoints.append(Keypoint(x=point["X"], y=point["Y"])))

        kps = KeypointsOnImage(keypoints, shape=(self.header["height"], self.header["width"]))
        kps_aug = augmenter(keypoints=kps)  # Augment keypoints
        point_idx = 0

        zero_point = {"C": 0, "X": 0, "Y": 0}

        frames = []
        for frame in self.body["frames"]:
            people = []
            for person in frame["people"]:
                person_n = {"id": person["id"]}
                for component in self.header["components"].keys():
                    person_n[component] = []
                    for point in person[component]:
                        if point["C"] > 0:
                            keypoint = kps_aug.keypoints[point_idx]
                            point_n = {"C": point["C"], "X": keypoint.x, "Y": keypoint.y}
                            person_n[component].append(point_n)
                        else:
                            person_n[component].append(zero_point)
                people.append(person_n)

            frames.append({"people": people})

        return Pose(self.header, {"frames": frames}, fps=self.body["fps"])

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

    def to_vectors(self, types: List[str], people=1) -> Iterator:
        aggregators = [POINT_AGGREGATORS[t] for t in types]

        vec_size = sum([len(v["limbs"]) for v in self.header["components"].values()]) * len(aggregators)

        for i, frame in enumerate(self.body["frames"]):
            vector = np.zeros(vec_size)  # its faster to initialize the vector as 0s, than to append to a list
            idx = 0
            for person in frame["people"][:people]:
                for name, component in self.header["components"].items():
                    for (a, b) in component["limbs"]:
                        a_point = person[name][component["points"].index(a)]
                        b_point = person[name][component["points"].index(b)]

                        if a_point["C"] == 0 or b_point["C"] == 0:
                            for _ in aggregators:
                                vector[idx] = 0
                                idx += 1
                        else:
                            p1 = Pose.point_to_numpy(a_point)
                            p2 = Pose.point_to_numpy(b_point)
                            for aggregator in aggregators:
                                vector[idx] = aggregator(p1, p2)
                                idx += 1
            yield vector

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

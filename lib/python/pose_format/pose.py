import copy
import struct
from collections import defaultdict
from itertools import chain
from typing import Tuple, Iterator, List

import math
import numpy as np
from imgaug import Keypoint, KeypointsOnImage
from imgaug.augmenters import Augmenter
from scipy.interpolate import interp1d

from .utils.fast_math import distance, angle


def s2b(s: str) -> bytes:
    return bytes(s, 'utf8')


POINT_AGGREGATORS = {
    "distance": distance,
    "angle": angle,
}


class Points:
    def __init__(self, dimensions, confidence):
        self.dimensions = dimensions
        self.confidence = confidence


class Pose:
    def __init__(self, header, body=None, fps=24):
        self.header = header
        self.body = body
        if self.body is None:
            self.body = {"frames": []}
        self.body["fps"] = fps

    def add_frame(self, frame):
        self.body["frames"].append(frame)

    def iterate_points_dimensions(self, callback):
        def iterate_dimensions(points, i, point):
            for j, v in enumerate(point):
                callback(points, i, j, v)

        return self.iterate_points(iterate_dimensions)

    def iterate_points(self, callback):
        def iterate_component(points):
            dimensions = points.dimensions.tolist()
            for i, v in enumerate(dimensions):
                if points.confidence[i] > 0:
                    callback(points, i, v)

        return self.iterate_components(iterate_component)

    def iterate_components(self, callback):
        for frame in self.body["frames"]:
            for person in frame["people"]:
                for name, component in self.header["components"].items():
                    callback(person[name])

    def focus_pose(self):
        """
        Gets the pose to start at (0,0) and have dimensions as big as needed
        """
        frames = self.body["frames"]
        if len(frames) == 0:
            return

        mins = [math.inf for _ in range(3)]
        maxs = [0 for _ in range(3)]

        def set_min_max(points):
            local_mins = np.amin(points.dimensions, axis=0)
            local_maxs = np.amax(points.dimensions, axis=0)

            for i, v in enumerate(local_mins):
                mins[i] = min(mins[i], v)

            for i, v in enumerate(local_maxs):
                maxs[i] = max(maxs[i], v)

        self.iterate_components(set_min_max)

        if np.count_nonzero(mins) > 0: # Only translate if there is a number to translate by
            def translate_points(points):
                truncated_mins = mins[:points.dimensions.shape[1]]
                points.dimensions = np.subtract(points.dimensions, truncated_mins)

            self.iterate_components(translate_points)

        self.header["width"] = math.ceil(max(maxs[0] - mins[0], 0))
        self.header["height"] = math.ceil(max(maxs[1] - mins[1], 0))
        self.header["depth"] = math.ceil(max(maxs[2] - mins[2], 0))

    def normalize(self, dist_p1: Tuple[str, int], dist_p2: Tuple[str, int], scale_factor: float = 1):
        frames = self.body["frames"]
        if len(frames) == 0:
            return

        # 1. Calculate the distance between p1 and p2
        distances = []
        for frame in frames:
            for person in frame["people"]:
                p1 = person[dist_p1[0]].dimensions[dist_p1[1]]
                p2 = person[dist_p2[0]].dimensions[dist_p2[1]]
                distances.append(distance(p1, p2))

        # 2. scale all points to dist/scale
        scale = scale_factor / np.mean(distances)

        def scale_point(points):
            points.dimensions = points.dimensions * scale

        self.iterate_components(scale_point)

        # 3. Shift all points to be positive, starting from 0
        self.focus_pose()

    def augment2d(self, augmenter: Augmenter):
        keypoints = []

        def add_keypoints(points, i, point):
            keypoints.append(Keypoint(x=point[0], y=point[1]))

        self.iterate_points(add_keypoints)

        kps = KeypointsOnImage(keypoints, shape=(self.header["height"], self.header["width"]))
        kps_aug = augmenter(keypoints=kps)  # Augment keypoints
        point_idx = 0

        frames = []
        for frame in self.body["frames"]:
            people = []
            for person in frame["people"]:
                person_n = {"id": person["id"]}
                for name, component in self.header["components"].items():
                    dimensions = np.zeros((len(component["points"]), len(component["point_format"])-1))
                    for i, c in enumerate(person[name].confidence):
                        if c > 0:
                            keypoint = kps_aug.keypoints[point_idx]
                            point_idx += 1
                            dimensions[i,0] = keypoint.x
                            dimensions[i,1] = keypoint.y
                    person_n[name] = Points(dimensions=dimensions, confidence=person[name].confidence)
                people.append(person_n)

            frames.append({"people": people})

        return Pose(self.header, {"frames": frames}, fps=self.body["fps"])

    def interpolate_fps(self, fps, kind='cubic'):
        raise NotImplementedError("Implementation not good")
        frames = self.body["frames"]

        # New array of the second for every frame
        new_frames_len = math.ceil((len(frames) - 1) * fps / self.body["fps"]) + 1
        new_x = [i / fps for i in range(new_frames_len)]

        keypoints = defaultdict(lambda: defaultdict(lambda: {"x": [], "y": []}))
        for i, frame in enumerate(frames):
            for person in frame["people"]:
                for name, component in self.header["components"].items():
                    dimensions = person[name].dimensions  # TODO need to also interpolate confidence
                    for j, c in enumerate(person[name].confidence):
                        if c > 0:
                            for d, y in enumerate(dimensions[j]):
                                k = (name, j, d)
                                keypoints[person["id"]][k]["x"].append(i / self.body["fps"])
                                keypoints[person["id"]][k]["y"].append(y)

        # Interpolate all dimensions
        interp = {}
        for p, data in keypoints.items():
            interp[p] = {}
            for k, v in data.items():
                if len(v["x"]) == 0:  # Can't interpolate
                    interp[p][k] = np.zeros(len(new_x))
                elif len(v["x"]) == 1:
                    interp[p][k] = np.full(v["y"][0])
                else:
                    min_range = next((i for i, x in enumerate(new_x) if x > v["x"][0]), 0)
                    max_range = next((i for i, x in enumerate(new_x) if x > v["x"][-1]), None)
                    interp[p][k] = [interp1d(v["x"], v["y"], kind=kind)(new_x[min_range:max_range])]

                    # Add padding
                    if min_range > 0:
                        interp[p][k].insert(0, [0] * min_range)
                    if max_range is not None:
                        interp[p][k].append([0] * (len(new_x) - max_range))

                    interp[p][k] = np.concatenate(interp[p][k])

        new_frames = []
        for i in range(len(new_x)):
            people = []
            for p, new_data in interp.items():
                person = {"id": p}
                for name, component in self.header["components"].items():
                    person[name] = Points(
                        dimensions=np.zeros((len(component["points"]), len(component["point_format"]) - 1)),
                        confidence=np.zeros((len(component["points"])))
                    )

                for (name, point_i, d), new_y in new_data.items():
                    print(name, point_i, d)
                    person[name].dimensions[point_i, d] = new_y[i]

                people.append(person)
            new_frames.append({"people": people})

        return Pose(self.header, {"frames": new_frames}, fps)

    def to_vectors(self, types: List[str], people=1) -> Iterator:
        aggregators = [POINT_AGGREGATORS[t] for t in types]

        vec_size = sum([len(v["limbs"]) for v in self.header["components"].values()]) * len(aggregators)

        for i, frame in enumerate(self.body["frames"]):
            limbs = []
            idx = 0
            for name, component in self.header["components"].items():
                for person in frame["people"][:people]:
                    dimensions = person[name].dimensions.tolist()
                    for (a, b) in component["limb_indexes"]:
                        limbs.append([dimensions[a], dimensions[b]])

            # TODO batch this
            vector = np.empty(vec_size)  # its faster to initialize empty vector, than to append to a list
            for aggregator in aggregators:
                for (p1, p2) in limbs:
                    vector[idx] = aggregator(p1, p2)

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

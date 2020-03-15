from io import BufferedReader, BufferedWriter
from typing import Tuple, List

from .vectorizer import Vectorizer
from .pose_body import PoseBody
from .pose_header import PoseHeader, PoseHeaderDimensions, PoseNormalizationInfo
from lib.python.pose_format.utils.reader import BufferReader

import numpy as np
import numpy.ma as ma

from .utils.fast_math import distance_batch


class Pose:
    def __init__(self, header: PoseHeader, body: PoseBody):
        self.header = header
        self.body = body

    @staticmethod
    def read(buffer: bytes):
        reader = BufferReader(buffer)
        header = PoseHeader.read(reader)
        body = PoseBody.read(header, reader)

        return Pose(header, body)

    def write(self, f_name: str):
        f = open(f_name, "wb")
        self.header.write(f)
        self.body.write(f)
        f.flush()
        f.close()

    def focus(self):
        """
        Gets the pose to start at (0,0) and have dimensions as big as needed
        """

        mins = ma.min(self.body.data, axis=(0, 1, 2))
        maxs = ma.max(self.body.data, axis=(0, 1, 2))

        if np.count_nonzero(mins) > 0:  # Only translate if there is a number to translate by
            self.body.data = ma.subtract(self.body.data, mins)

        dimensions = (maxs - mins).tolist()
        self.header.dimensions = PoseHeaderDimensions(*dimensions)

    def normalize(self, info: PoseNormalizationInfo, scale_factor: float = 1):
        """
        Normalize the point to a fixed distance between two points
        """
        transposed = self.body.points_perspective()

        p1s = transposed[info.p1]
        p2s = transposed[info.p2]

        if transposed.shape[1] == 0:
            p1s = p1s[0]
            p2s = p2s[0]
        else:
            p1s = ma.concatenate(p1s)
            p2s = ma.concatenate(p2s)

        mean_distance = np.mean(distance_batch(p1s, p2s))

        scale = scale_factor / mean_distance  # scale all points to dist/scale

        if round(scale, 5) != 1:
            self.body.data = ma.multiply(self.body.data, scale)

    def to_vectors(self, vectorizer: Vectorizer) -> np.ndarray:
        transposed = self.body.points_perspective()

        pt1s = []
        pt2s = []

        idx = 0
        for component in self.header.components:
            for (a, b) in component.limbs:
                pt1s.append(a+idx)
                pt2s.append(b+idx)
            idx += len(component.points)

        people_vectors = vectorizer(transposed[pt1s], transposed[pt2s])
        if people_vectors.shape[1] == 1:
            vectors = np.squeeze(people_vectors)
        else:
            vectors = np.concatenate(people_vectors, axis=0)

        return np.transpose(vectors)


        #
        # vec_size = sum([len(v["limbs"]) for v in self.header["components"].values()]) * len(aggregators)
        #
        # for i, frame in enumerate(self.body["frames"]):
        #     limbs = []
        #     for name, component in self.header["components"].items():
        #         for person in frame["people"][:people]:
        #             dimensions = person[name].dimensions.tolist()
        #             for (a, b) in component["limb_indexes"]:
        #                 limbs.append([dimensions[a], dimensions[b]])
        #
        #     vector = np.zeros(vec_size)  # its faster to initialize empty vector, than to append to a list
        #     idx = 0
        #     # TODO batch this
        #     for aggregator in aggregators:
        #         for (p1, p2) in limbs:
        #             vector[idx] = aggregator(p1, p2)
        #             idx += 1
        #
        #     yield vector

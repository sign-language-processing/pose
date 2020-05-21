from typing import List

import numpy as np
from numpy import ma

from .pose_header import PoseHeader
from .utils.fast_math import distance_batch, angle_batch, slope_batch


class Vectorizer:
    def __call__(self, p1s: np.ndarray, p2s: np.ndarray, header: PoseHeader):
        raise NotImplementedError("Must implement '__call__'")


class ZeroVectorizer(Vectorizer):
    def __call__(self, p1s: np.ndarray, p2s: np.ndarray, header: PoseHeader):
        return np.zeros((len(p1s), 1))

class DistanceVectorizer(Vectorizer):
    def __call__(self, p1s: np.ndarray, p2s: np.ndarray, header: PoseHeader):
        return distance_batch(p1s, p2s).filled(0)


class SlopeVectorizer(Vectorizer):
    def __call__(self, p1s: np.ndarray, p2s: np.ndarray, header: PoseHeader):
        return slope_batch(p1s, p2s).filled(0)


class AngleVectorizer(Vectorizer):
    def __call__(self, p1s: np.ndarray, p2s: np.ndarray, header: PoseHeader):
        return angle_batch(p1s, p2s).filled(0)


class RelativeAngleVectorizer(Vectorizer):
    def __call__(self, p1s: np.ndarray, p2s: np.ndarray, header: PoseHeader):
        angles = angle_batch(p1s, p2s)
        angles = ma.concatenate([angles, np.zeros((1, angles.shape[1], angles.shape[2]))], axis=0)

        none_index = angles.shape[0] - 1

        # Get vector of limb relativity
        l1s = []
        l2s = []

        idx = 0
        for component in header.components:
            for i, to in enumerate(component.relative_limbs):
                l1s.append(i + idx)
                l2s.append(to + idx if to is not None else none_index)
            idx += len(component.limbs)

        # Subtract angles
        relative_angles = angles[l1s] - angles[l2s]

        return relative_angles.filled(0)


class SequenceVectorizer(Vectorizer):
    def __init__(self, vectorizers: List[Vectorizer]):
        self.vectorizers = vectorizers

    def __call__(self, p1s: np.ndarray, p2s: np.ndarray, header: PoseHeader):
        agg = [a(p1s=p1s, p2s=p2s, header=header) for a in self.vectorizers]
        return np.concatenate(agg)

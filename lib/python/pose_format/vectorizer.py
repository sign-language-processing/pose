from typing import List

from numpy import ma

from .utils.fast_math import distance_batch, angle_batch


class Vectorizer:
    def __call__(self, p1s, p2s):
        raise NotImplementedError("Must implement '__call__'")


class DistanceVectorizer(Vectorizer):
    def __call__(self, p1s, p2s):
        return distance_batch(p1s, p2s).filled(0)

class AngleVectorizer(Vectorizer):
    def __call__(self, p1s, p2s):
        return angle_batch(p1s, p2s).filled(0)

class SequenceVectorizer(Vectorizer):
    def __init__(self, vectorizers: List[Vectorizer]):
        self.vectorizers = vectorizers

    def __call__(self, p1s, p2s):
        agg = [a(p1s, p2s) for a in self.vectorizers]
        return ma.concatenate(agg)

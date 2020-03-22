import math
import numpy as np
from numpy import ma


def distance(p1, p2):
    return math.sqrt(sum([(a - b) ** 2 for a, b in zip(p1, p2)]))


def distance_points(points, i, j):
    return distance(points[i], points[j])


def distance_batch(p1s, p2s):
    return np.sqrt(((p1s - p2s) ** 2).sum(axis=-1))


def angle(p1, p2):
    d = p1[0] - p2[0]
    if d == 0:
        return 0
    return (p1[1] - p2[1]) / d


def angle_batch(p1s, p2s):
    d = p2s - p1s
    xs, ys = np.split(d, [-1], axis=3)
    return ma.divide(ys, xs).squeeze(axis=3)

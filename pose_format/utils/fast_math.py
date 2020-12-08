import numpy as np


def distance_batch(p1s, p2s):
    return np.sqrt(((p1s - p2s) ** 2).sum(axis=-1))

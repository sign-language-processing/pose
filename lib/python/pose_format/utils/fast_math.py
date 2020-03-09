from typing import Tuple

import math


def distance(p1, p2):
    return math.sqrt(sum([(a - b) ** 2 for a, b in zip(p1, p2)]))

def angle(p1: Tuple, p2: Tuple):
    d = p1[0] - p2[0]
    if d == 0:
        return 0
    return (p1[1] - p2[1]) / d
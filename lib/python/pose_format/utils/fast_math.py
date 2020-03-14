import math

def distance(p1, p2):
    return math.sqrt(sum([(a - b) ** 2 for a, b in zip(p1, p2)]))

def distance_points(points, i, j):
    return distance(points[i], points[j])


def angle(p1, p2):
    d = p1[0] - p2[0]
    if d == 0:
        return 0
    return (p1[1] - p2[1]) / d


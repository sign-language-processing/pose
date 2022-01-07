def distance_batch(p1s, p2s):
    squared = (p1s - p2s) ** 2
    summed = squared.sum(axis=-1)
    return summed ** 0.5

def distance_batch(p1s, p2s):
    """
    Computes Euclidean distance between two sets of points in batch

    Parameters
    ----------
    p1s : array-like
        array of shape (N, D) where N; number of points & D ; dimensionality of each point
    p2s : array-like
        array of shape (N, D) with N; number of points & D; dimensionality of each point

    Returns
    -------
    array-like
        array of shape (N,) with euclidean distances between points in `p1s` and `p2s`

    Examples
    --------
    >>> distance_batch(np.array([[0, 0], [1, 1]]), np.array([[1, 1], [2, 2]]))
    array([1.41421356, 1.41421356])

    Note
    ----
    Function assumes that the inputs `p1s` and `p2s` have the same shape
    """
    squared = (p1s - p2s)**2
    summed = squared.sum(axis=-1)
    return summed**0.5

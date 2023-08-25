import numpy.ma as ma


class DistanceRepresentation:
    """
    A class to compute the Euclidean distance between two sets of points.
    """

    def distance(self, p1s: ma.MaskedArray, p2s: ma.MaskedArray) -> ma.MaskedArray:
        """
        Compute the Euclidean distance between two sets of points.

        Parameters
        ----------
        p1s : ma.MaskedArray
            First set of points.
        p2s : ma.MaskedArray
            Second set of points.

        Returns
        -------
        ma.MaskedArray
            Euclidean distances between the two sets of points. The returned array has one fewer dimension than the input arrays, as the distance calculation collapses the last dimension.

        Note
        ----
        this method assumes that input arrays `p1s` and `p2s` have same shape.
        """
        diff = p1s - p2s
        square = ma.power(diff, 2)
        sum_squares = square.sum(axis=-1)
        sqrt = ma.sqrt(sum_squares).filled(0)
        return sqrt

    def __call__(self, p1s: ma.MaskedArray, p2s: ma.MaskedArray) -> ma.MaskedArray:
        """
        For `distance` method to compute Euclidean distance between two points.

        Parameters
        ----------
        p1s : ma.MaskedArray, shape (Points, Batch, Len, Dims)
            First set of points.
        p2s : ma.MaskedArray, shape (Points, Batch, Len, Dims)
            Second set of points.

        Returns
        -------
        ma.MaskedArray, shape (Points, Batch, Len)
            Euclidean distances between the two sets of points.
        """
        return self.distance(p1s, p2s)

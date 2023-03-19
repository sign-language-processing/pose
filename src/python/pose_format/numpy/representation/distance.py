import numpy.ma as ma


class DistanceRepresentation:
    def distance(self, p1s: ma.MaskedArray, p2s: ma.MaskedArray) -> ma.MaskedArray:
        diff = p1s - p2s
        square = ma.power(diff, 2)
        sum_squares = square.sum(axis=-1)
        sqrt = ma.sqrt(sum_squares).filled(0)
        return sqrt

    def __call__(self, p1s: ma.MaskedArray, p2s: ma.MaskedArray) -> ma.MaskedArray:
        """
        Euclidean distance between two points
        :param p1s: ma.MaskedArray (Points, Batch, Len, Dims)
        :param p2s: ma.MaskedArray (Points, Batch, Len, Dims)
        :return: ma.MaskedArray (Points, Batch, Len)
        """
        return self.distance(p1s, p2s)

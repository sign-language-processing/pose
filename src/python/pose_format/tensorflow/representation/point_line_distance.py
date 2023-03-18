import tensorflow as tf

from .distance import DistanceRepresentation


class PointLineDistanceRepresentation:
    def __init__(self):
        self.distance = DistanceRepresentation()

    def __call__(self, p1s: tf.Tensor, p2s: tf.Tensor, p3s: tf.Tensor) -> tf.Tensor:
        """
        Distance between the point p1s to the line <p2s, p3s>
        :param p1s: tf.Tensor (Points, Batch, Len, Dims)
        :param p2s: tf.Tensor (Points, Batch, Len, Dims)
        :param p3s: tf.Tensor (Points, Batch, Len, Dims)
        :return: tf.Tensor (Points, Batch, Len)
        """

        # Following Heron's Formula https://en.wikipedia.org/wiki/Heron%27s_formula
        a = self.distance.distance(p1s, p2s)
        b = self.distance.distance(p2s, p3s)
        c = self.distance.distance(p1s, p3s)
        s: tf.Tensor = (a + b + c) / 2
        squared = s * (s - a) * (s - b) * (s - c)
        area = tf.sqrt(squared)

        # Calc "height" of the triangle
        square_area: tf.Tensor = area * 2
        distance = tf.math.divide_no_nan(square_area, b)
        # TODO add .zero_filled()

        return distance

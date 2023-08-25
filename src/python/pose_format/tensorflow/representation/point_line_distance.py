import tensorflow as tf

from .distance import DistanceRepresentation


class PointLineDistanceRepresentation:
    """
    A class to compute the distance between a point and a line segment.
    
    Parameters
    ---------- 
    distance : :class:`~pose_format.tensorflow.representation.distance.DistanceRepresentation`
        Instance of the `DistanceRepresentation` class to compute the Euclidean distance.

    """

    def __init__(self):
        """
        Initializes the PointLineDistanceRepresentation with an instance of DistanceRepresentation.
        """
        self.distance = DistanceRepresentation()

    def __call__(self, p1s: tf.Tensor, p2s: tf.Tensor, p3s: tf.Tensor) -> tf.Tensor:
        """
        Computes the distance between the point `p1s` and the line segment formed by `p2s` and `p3s`.

        Parameters
        ----------
        p1s : tf.Tensor
            The point for which we want to calculate the distance from the line segment 
            with shape (Points, Batch, Len, Dims).
        p2s : tf.Tensor
            One of the endpoints of the line segment with shape (Points, Batch, Len, Dims).
        p3s : tf.Tensor
            The other endpoint of the line segment with shape (Points, Batch, Len, Dims).

        Returns
        -------
        tf.Tensor
            A tensor representing the distance of point `p1s` from the line segment with shape (Points, Batch, Len).

        Note
        ----
        This method computes the distance using Heron's formula to first compute the area of the 
        triangle formed by the three points, and then determines the "height" of this triangle 
        with respect to the base formed by the line segment.
        
        * References: 
            Following Heron's Formula https://en.wikipedia.org/wiki/Heron%27s_formula
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

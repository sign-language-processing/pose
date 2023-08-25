import tensorflow as tf


class DistanceRepresentation:
    """A class to represent the Euclidean distance between two sets of points."""

    def distance(self, p1s: tf.Tensor, p2s: tf.Tensor) -> tf.Tensor:
        """
        Computes the Euclidean distance between two sets of points.

        Parameters
        ----------
        p1s : tf.Tensor
            First set of points with shape (Points, Batch, Len, Dims).
        p2s : tf.Tensor
            Second set of points with shape (Points, Batch, Len, Dims).

        Returns
        -------
        tf.Tensor
            A tensor representing the Euclidean distance between the two points 
            with shape (Points, Batch, Len).
        
        Note
        ----
        The function computes the difference between the two sets of points,
        squares the differences, sums the squared differences along the last axis,
        and then takes the square root to calculate the Euclidean distance.
        """
        diff = p1s - p2s  # (Points, Batch, Len, Dims)
        square = tf.square(diff)
        sum_squares = tf.reduce_sum(square, axis=-1)
        # TODO add .zero_filled()

        return tf.sqrt(sum_squares)

    def __call__(self, p1s: tf.Tensor, p2s: tf.Tensor) -> tf.Tensor:
        """
        Computes the Euclidean distance between two sets of points.

        Parameters
        ----------
        p1s : tf.Tensor
            First set of points with shape (Points, Batch, Len, Dims).
        p2s : tf.Tensor
            Second set of points with shape (Points, Batch, Len, Dims).

        Returns
        -------
        tf.Tensor
            A tensor representing the Euclidean distance between the two points 
            with shape (Points, Batch, Len).
        
        Note
        ----
        This method is essentially an alias for the `distance` method.
        """
        return self.distance(p1s, p2s)

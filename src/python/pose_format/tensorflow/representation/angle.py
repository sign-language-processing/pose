import tensorflow as tf


class AngleRepresentation:
    """
    A class to represent the angle between the X/Y axis and line formed by two points.
    """

    def __call__(self, p1s: tf.Tensor, p2s: tf.Tensor) -> tf.Tensor:
        """
        Computes the angle of the X/Y axis between two points.

        Parameters
        ----------
        p1s : tf.Tensor
            First set of points with shape (Points, Batch, Len, Dims).
        p2s : tf.Tensor
            Second set of points with shape (Points, Batch, Len, Dims).

        Returns
        -------
        tf.Tensor
            A tensor representing the angle of the X/Y axis between two points
            with shape (Points, Batch, Len).
        
        Note
        ----
        The function computes the difference between the two point sets, 
        splits the difference into X and Y components, and then calculates 
        the slope and the angle using the arctan function. 
        If the x difference is zero, the function returns a result that 
        avoids NaN by using `tf.math.divide_no_nan`.
        """
        dims = p1s.shape[-1]

        d = p2s - p1s  # (Points, Batch, Len, Dims)
        xs, ys = tf.split(d, [1] * dims, axis=3)[:2]  # (Points, Batch, Len, 1)
        slopes = tf.math.divide_no_nan(ys, xs)  # Divide, no NaN
        # TODO add .zero_filled()
        slopes = tf.squeeze(slopes, axis=3)

        return tf.math.atan(slopes)

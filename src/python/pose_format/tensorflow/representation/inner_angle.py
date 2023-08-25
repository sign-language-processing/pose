import tensorflow as tf


def get_vectors_norm(vectors):
    """
    Computes the normalized version of the input vectors.

    Parameters
    ----------
    vectors : tf.Tensor
        A tensor containing vectors.

    Returns
    -------
    tf.Tensor
        The normalized vectors.

    Notes
    -----
    This function transposes the input vectors, computes the magnitude (norm) 
    of the vectors, and then returns the normalized version by dividing 
    each vector by its magnitude.
    """
    transposed = tf.transpose(vectors)
    v_mag = tf.sqrt(tf.math.reduce_sum(transposed * transposed, axis=0))
    return tf.transpose(tf.math.divide_no_nan(transposed, v_mag))


class InnerAngleRepresentation:
    """A class to represent the inner angle formed at a point for a given triangle. """

    def __call__(self, p1s: tf.Tensor, p2s: tf.Tensor, p3s: tf.Tensor) -> tf.Tensor:
        """
        Computes the angle at point `p2s` for the triangle formed by `p1s`, `p2s`, and `p3s`.

        Parameters
        ----------
        p1s : tf.Tensor
            First set of points with shape (Points, Batch, Len, Dims).
        p2s : tf.Tensor
            Second set of points, where the angle is formed, 
            with shape (Points, Batch, Len, Dims).
        p3s : tf.Tensor
            Third set of points with shape (Points, Batch, Len, Dims).

        Returns
        -------
        tf.Tensor
            A tensor representing the angle (in radians) at point `p2s` 
            for the triangle with shape (Points, Batch, Len).
        
        Note
        ----
        This method determines the vectors pointing towards `p1s` and `p3s` 
        from the point `p2s`, normalizes these vectors, and then computes 
        the dot product between them. The angle between these vectors is 
        computed using the arccosine function on the dot product.

        Refrences: 
        * https://stackoverflow.com/questions/19729831/angle-between-3-points-in-3d-space
        """

        # Following https://stackoverflow.com/questions/19729831/angle-between-3-points-in-3d-space
        v1 = p1s - p2s  # (Points, Batch, Len, Dims)
        v2 = p3s - p2s  # (Points, Batch, Len, Dims)

        v1_norm = get_vectors_norm(v1)
        v2_norm = get_vectors_norm(v2)

        slopes = tf.reduce_sum(v1_norm * v2_norm, axis=3)
        angles = tf.acos(slopes)

        angles = tf.where(tf.math.is_nan(angles), 0., angles)  # Fix NaN, TODO think of faster way
        return angles

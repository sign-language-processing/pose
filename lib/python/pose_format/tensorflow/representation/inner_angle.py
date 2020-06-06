import tensorflow as tf


def get_vectors_norm(vectors):
    transposed = tf.transpose(vectors)
    v_mag = tf.sqrt(tf.math.reduce_sum(transposed * transposed, axis=0))
    return tf.transpose(tf.math.divide_no_nan(transposed, v_mag))


class InnerAngleRepresentation:
    def __call__(self, a: tf.Tensor, b: tf.Tensor, c: tf.Tensor) -> tf.Tensor:
        """
        Angle in point b for the triangle <a, b, c>
        :param a: tf.Tensor (Points, Batch, Len, Dims)
        :param b: tf.Tensor (Points, Batch, Len, Dims)
        :param c: tf.Tensor (Points, Batch, Len, Dims)
        :return: tf.Tensor (Points, Batch, Len)
        """

        # Following https://stackoverflow.com/questions/19729831/angle-between-3-points-in-3d-space
        v1 = a - b  # (Points, Batch, Len, Dims)
        v2 = c - b  # (Points, Batch, Len, Dims)

        v1_norm = get_vectors_norm(v1)
        v2_norm = get_vectors_norm(v2)

        slopes = tf.reduce_sum(v1_norm * v2_norm, axis=3)
        angles = tf.acos(slopes)

        angles = tf.where(tf.math.is_nan(angles), 0., angles)  # Fix NaN, TODO think of faster way
        return angles

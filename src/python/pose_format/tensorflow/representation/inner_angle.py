import tensorflow as tf


def get_vectors_norm(vectors):
    transposed = tf.transpose(vectors)
    v_mag = tf.sqrt(tf.math.reduce_sum(transposed * transposed, axis=0))
    return tf.transpose(tf.math.divide_no_nan(transposed, v_mag))


class InnerAngleRepresentation:
    def __call__(self, p1s: tf.Tensor, p2s: tf.Tensor, p3s: tf.Tensor) -> tf.Tensor:
        """
        Angle in point p2s for the triangle <p1s, p2s, p3s>
        :param p1s: tf.Tensor (Points, Batch, Len, Dims)
        :param p2s: tf.Tensor (Points, Batch, Len, Dims)
        :param p3s: tf.Tensor (Points, Batch, Len, Dims)
        :return: tf.Tensor (Points, Batch, Len)
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

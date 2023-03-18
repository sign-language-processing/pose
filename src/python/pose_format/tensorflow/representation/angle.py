import tensorflow as tf


class AngleRepresentation:
    def __call__(self, p1s: tf.Tensor, p2s: tf.Tensor) -> tf.Tensor:
        """
        Angle of the X/Y axis between two points
        :param p1s: tf.Tensor (Points, Batch, Len, Dims)
        :param p2s: tf.Tensor (Points, Batch, Len, Dims)
        :return: tf.Tensor (Points, Batch, Len)
        """
        dims = p1s.shape[-1]

        d = p2s - p1s  # (Points, Batch, Len, Dims)
        xs, ys = tf.split(d, [1] * dims, axis=3)[:2]  # (Points, Batch, Len, 1)
        slopes = tf.math.divide_no_nan(ys, xs)  # Divide, no NaN
        # TODO add .zero_filled()
        slopes = tf.squeeze(slopes, axis=3)

        return tf.math.atan(slopes)

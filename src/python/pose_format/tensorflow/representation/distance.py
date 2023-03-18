import tensorflow as tf


class DistanceRepresentation:
    def distance(self, p1s: tf.Tensor, p2s: tf.Tensor) -> tf.Tensor:
        diff = p1s - p2s  # (Points, Batch, Len, Dims)
        square = tf.square(diff)
        sum_squares = tf.reduce_sum(square, axis=-1)
        # TODO add .zero_filled()

        return tf.sqrt(sum_squares)

    def __call__(self, p1s: tf.Tensor, p2s: tf.Tensor) -> tf.Tensor:
        """
        Euclidean distance between two points
        :param p1s: tf.Tensor (Points, Batch, Len, Dims)
        :param p2s: tf.Tensor (Points, Batch, Len, Dims)
        :return: tf.Tensor (Points, Batch, Len)
        """
        return self.distance(p1s, p2s)

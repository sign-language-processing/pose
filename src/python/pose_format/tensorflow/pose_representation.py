from typing import List

import tensorflow as tf

from ..pose_representation import PoseRepresentation


class TensorflowPoseRepresentation(PoseRepresentation):
    def group_embeds(self, embeds: List[tf.Tensor]):
        """
        :param embeds: torch.Tensor List of tensors size (embed_size, Batch, Len)
        :return: Size (Batch, Len, embed_size)
        """
        group = tf.concat(embeds, axis=0)  # (embed_size, Batch, Len)
        return tf.transpose(group, perm=[1, 2, 0])

    def get_points(self, tensor: tf.Tensor, points: List):
        return tf.gather(tensor, points)

    def permute(self, src, shape: tuple):
        return tf.transpose(src, perm=shape)

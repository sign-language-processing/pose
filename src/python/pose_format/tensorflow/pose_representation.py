from typing import List

import tensorflow as tf

from ..pose_representation import PoseRepresentation


class TensorflowPoseRepresentation(PoseRepresentation):
    """
    Class for pose representations using TensorFlow tensors.

        * Inherites from ``PoseRepresentation`` 
        
        This class extends PoseRepresentation and provides methods for manipulating pose representations
        using TensorFlow tensors.
        """

    def group_embeds(self, embeds: List[tf.Tensor]):
        """
        Group embeddings (list of tensors) along the first dimension.

        Parameters
        ----------
        embeds : List[tf.Tensor]
            List of tensors, each with shape (embed_size, Batch, Len).

        Returns
        -------
        tf.Tensor
            Tensor with shape (Batch, Len, embed_size).

        """
        group = tf.concat(embeds, axis=0)  # (embed_size, Batch, Len)
        return tf.transpose(group, perm=[1, 2, 0])

    def get_points(self, tensor: tf.Tensor, points: List):
        """
        Get specific points from a tensor.

        Parameters
        ----------
        tensor : tf.Tensor
            Tensor.
        points : List[int]
            Indices/points needed from Tensor

        Returns
        -------
        tf.Tensor
            Get values from the tensor using the given indices/points

        """
        return tf.gather(tensor, points)

    def permute(self, src, shape: tuple):
        """
        Permute dimensions of a tensor according to a given shape (tuple).

        Parameters
        ----------
        src : tf.Tensor
            tensor to permute
        shape : tuple
            Desired shape to permute to. 

        Returns
        -------
        tf.Tensor
            The permuted tensor.

        """
        return tf.transpose(src, perm=shape)

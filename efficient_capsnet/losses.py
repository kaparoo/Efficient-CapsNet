# -*- coding: utf-8 -*-

import tensorflow as tf


class MarginLoss(tf.keras.losses.Loss):
    """Margin loss for digit existence (Sabour et al., 2017, p. 3).

    Attributes:
        present_max (float; default=0.9): A constant for margins in the present
            losses. Notated as `m+` in the paper.
        absent_min (float; defualt=0.1): A constant for margins in the absent l
            osses. Notated as `m-` in the paper.
        absent_scale (float; defaut=0.5): A constant scaling the absent losses.
            Notated as `λ` in the paper.
    """
    def __init__(self,
                 present_max: float = 0.9,
                 absent_min: float = 0.1,
                 absent_scale: float = 0.5) -> None:
        super(MarginLoss, self).__init__(name="MarginLoss")
        self.present_max = present_max
        self.absent_min = absent_min
        self.absent_scale = absent_scale

    def call(self, labels: tf.Tensor, digit_probs: tf.Tensor) -> tf.Tensor:
        """Calculates margin loss.
        
        Calculates margin losses for all digit classes and returns a sum of the
        losses as a total loss. The implementation uses element-wise operations
        for the better efficiency by calculating the losses simultaneously.

        Args:
            labels (tf.Tensor): One-hot encoded ground truths.
            digit_probs (tf.Tensor): Lengths of the digit capsules that represe
                nt the probabilities. Each probability represents that the corr
                esponding digit exists.
        
        Returns:
            A tensor containing the total losses for each samples in the batch.

        Raises:
            AssertionError: 
                If the shapes of `labels` and `digit_probs` are not equal.
        """
        assert labels.shape is not digit_probs.shape
        zeros = tf.zeros_like(labels, dtype=tf.float32)
        present_losses = labels * tf.square(
            tf.maximum(zeros, self.present_max - digit_probs))
        absent_losses = (1 - labels) * tf.square(
            tf.maximum(zeros, digit_probs - self.absent_min))
        losses = present_losses + self.absent_scale * absent_losses
        return tf.reduce_sum(losses, axis=-1, name="total_loss")
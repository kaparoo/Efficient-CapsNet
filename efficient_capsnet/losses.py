# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.keras import losses


def margin_loss(labels: tf.Tensor,
                digit_probs: tf.Tensor,
                m_present: float = 0.9,
                m_absent: float = 0.1,
                l_absent: float = 0.5) -> tf.Tensor:
    zeros = tf.zeros(shape=digit_probs.shape)
    present_margin = labels * tf.math.square(
        tf.maximum(zeros, m_present - digit_probs))
    absent_margin = l_absent * (1 - labels) * tf.math.square(
        tf.maximum(zeros, digit_probs - m_absent))
    return tf.reduce_sum(present_margin + absent_margin, axis=-1)


class MarginLoss(losses.Loss):
    def __init__(self,
                 m_present: float = 0.9,
                 m_absent: float = 0.1,
                 l_absent: float = 0.5) -> None:
        super(MarginLoss, self).__init__(name="MarginLoss")
        self.m_present = m_present
        self.m_absent = m_absent
        self.l_absent = l_absent

    def call(self, labels: tf.Tensor, digit_probs: tf.Tensor) -> tf.Tensor:
        assert labels.shape == digit_probs.shape
        zeros = tf.zeros(shape=digit_probs.shape)
        present_margin = labels * tf.math.square(
            tf.maximum(zeros, self.m_present - digit_probs))
        absent_margin = self.l_absent * (1 - labels) * tf.math.square(
            tf.maximum(zeros, digit_probs - self.m_absent))
        return tf.reduce_sum(present_margin + absent_margin, axis=-1)

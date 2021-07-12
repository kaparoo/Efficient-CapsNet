# -*- coding: utf-8 -*-

from efficient_capsnet.layers import DigitCaps
from efficient_capsnet.layers import FeatureMap
from efficient_capsnet.layers import PrimaryCaps
from efficient_capsnet.param import CapsNetParam

import tensorflow as tf
from tensorflow.keras import models


class EfficientCapsNet(models.Model):
    def __init__(self, param: CapsNetParam) -> None:
        super(EfficientCapsNet, self).__init__(name="Efficient-CapsNet")
        self.param = param

    def build(self, _) -> None:
        self.feature_map = FeatureMap(self.param)
        self.primary_caps = PrimaryCaps(self.param)
        self.digit_caps = DigitCaps(self.param)

    def call(self, input_image: tf.Tensor) -> tf.Tensor:
        feature_map = self.feature_map(input_image)
        primary_caps = self.primary_caps(feature_map)
        digit_caps = self.digit_caps(primary_caps)
        return tf.norm(digit_caps, axis=-1)
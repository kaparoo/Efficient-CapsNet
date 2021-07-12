# -*- coding: utf-8 -*-

from efficient_capsnet.param import CapsNetParam

import tensorflow as tf
from tensorflow.keras import layers


def _dubbed_squash(input_vector: tf.Tensor, eps: float = 1e-7) -> tf.Tensor:
    _norm = tf.norm(input_vector, axis=-1, keepdims=True)
    _coef = (1 - 1 / (tf.math.exp(_norm) + eps))
    _unit = input_vector / (_norm + eps)
    return _coef * _unit


class FeatureMap(layers.Layer):
    def __init__(self, param: CapsNetParam) -> None:
        super(FeatureMap, self).__init__(name="FeatureMap")
        self.param = param

    def build(self, input_shape: tf.TensorShape) -> None:
        self.conv1 = layers.Conv2D(name="feature_map_conv1",
                                   input_shape=input_shape[1:],
                                   filters=self.param.conv1_filter,
                                   kernel_size=self.param.conv1_kernel,
                                   strides=self.param.conv1_stride,
                                   activation=tf.nn.relu)
        self.norm1 = layers.BatchNormalization()
        self.conv2 = layers.Conv2D(name="feature_map_conv2",
                                   filters=self.param.conv2_filter,
                                   kernel_size=self.param.conv2_kernel,
                                   strides=self.param.conv2_stride,
                                   activation=tf.nn.relu)
        self.norm2 = layers.BatchNormalization()
        self.conv3 = layers.Conv2D(name="feature_map_conv3",
                                   filters=self.param.conv3_filter,
                                   kernel_size=self.param.conv3_kernel,
                                   strides=self.param.conv3_stride,
                                   activation=tf.nn.relu)
        self.norm3 = layers.BatchNormalization()
        self.conv4 = layers.Conv2D(name="feature_map_conv4",
                                   filters=self.param.conv4_filter,
                                   kernel_size=self.param.conv4_kernel,
                                   strides=self.param.conv4_stride,
                                   activation=tf.nn.relu)
        self.norm4 = layers.BatchNormalization()

    def call(self, input_image: tf.Tensor) -> tf.Tensor:
        feature_map = self.norm1(self.conv1(input_image))
        feature_map = self.norm2(self.conv2(feature_map))
        feature_map = self.norm3(self.conv3(feature_map))
        return self.norm4(self.conv4(feature_map))


class PrimaryCaps(layers.Layer):
    def __init__(self, param: CapsNetParam) -> None:
        super(PrimaryCaps, self).__init__(name="PrimaryCaps")
        self.param = param

    def build(self, input_shape: tf.TensorShape) -> None:
        self.dconv = layers.Conv2D(name="primary_caps_dconv",
                                   input_shape=input_shape[1:],
                                   filters=self.param.dconv_filter,
                                   kernel_size=self.param.dconv_kernel,
                                   strides=self.param.dconv_stride,
                                   groups=self.param.dconv_filter)
        self.reshape = layers.Reshape(
            name="primary_caps_reshape",
            target_shape=(-1, self.param.dim_primary_caps))

    def call(self, feature_map: tf.Tensor) -> tf.Tensor:
        return _dubbed_squash(self.reshape(self.dconv(feature_map)))


class DigitCaps(layers.Layer):
    def __init__(self, param: CapsNetParam) -> None:
        super(DigitCaps, self).__init__(name="DigitCaps")
        self.param = param

    def build(self, input_shape: tf.TensorShape) -> None:
        self.num_primary_caps = input_shape[1]
        self.dim_primary_caps = input_shape[2]
        self.num_digit_caps = self.param.num_digit_caps
        self.dim_digit_caps = self.param.dim_digit_caps
        self.W = self.add_weight(
            name="digit_caps_weights",
            shape=(self.num_digit_caps, self.num_primary_caps,
                   self.dim_digit_caps, self.dim_primary_caps),
            dtype=tf.float32,
            initializer="glorot_uniform",
            trainable=True)
        self.B = self.add_weight(num="digit_caps_log_priors",
                                 shape=(self.num_digit_caps, 1,
                                        self.num_primary_caps),
                                 dtype=tf.float32,
                                 initializer="glorot_uniform",
                                 trainable=True)

    def compute_output_shape(self,
                             input_shape: tf.TensorShape) -> tf.TensorShape:
        return tf.TensorShape(shape=(input_shape[0], self.num_digit_caps,
                                     self.dim_digit_caps))

    def call(self, primary_caps: tf.Tensor) -> tf.Tensor:
        # U.shape: [None, num_digit_caps, num_primary_caps, dim_primary_caps, 1]
        U = tf.expand_dims(primary_caps, axis=1)
        U = tf.tile(U, [1, self.num_digit_caps, 1, 1])
        U = tf.expand_dims(U, axis=-1)

        # U_hat.shape: [None, num_digit_caps, num_primary_caps, dim_digit_caps]
        U_hat = tf.squeeze(tf.map_fn(lambda u: tf.matmul(self.W, u), U))

        # A.shape: [None, num_digit_caps, num_primary_caps, num_primary_caps]
        A = tf.matmul(U_hat, tf.transpose(
            U_hat, perm=(0, 1, 3, 2))) / tf.math.sqrt(
                tf.cast(self.dim_primary_caps, dtype=tf.float32))

        # C.shape: [None, num_digit_caps, 1, num_primary_caps]
        A_sum = tf.reduce_sum(A, axis=2, keepdims=True)
        C = tf.nn.softmax(A_sum, axis=1)

        # S.shape: [None, num_digit_caps, dim_digit_caps]
        B = tf.tile(tf.expand_dims(self.B, axis=0),
                    [primary_caps.shape[0], 1, 1, 1])
        S = tf.squeeze(tf.matmul(B + C, U_hat))

        return _dubbed_squash(S)

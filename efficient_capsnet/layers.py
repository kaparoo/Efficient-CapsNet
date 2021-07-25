# -*- coding: utf-8 -*-

from efficient_capsnet.param import CapsNetParam

import tensorflow as tf


class Squash(tf.keras.layers.Layer):
    """Capsule-wise squash function (Mazzia et al., 2021, p. 5).
    
    Attributes:
        eps (float; default=1e-7): A small constant for numerical stability.
    """
    def __init__(self, eps: float = 1e-7, name: str = "squash") -> None:
        super(Squash, self).__init__(name=name)
        self.eps = eps

    def call(self, input_vector: tf.Tensor) -> tf.Tensor:
        """Maps the norm of `input_vector` into [0, 1].
        
        Args:
            input_vector (tf.Tensor): A target vector (or list of vectors).

        Returns:
            A tensor.
        """
        norm = tf.norm(input_vector, axis=-1, keepdims=True)
        coef = 1 - 1 / tf.exp(norm)
        unit = input_vector / (norm + self.eps)
        return coef * unit

    def compute_output_shape(self,
                             input_shape: tf.TensorShape) -> tf.TensorShape:
        return input_shape


class FeatureMap(tf.keras.layers.Layer):
    def __init__(self, param: CapsNetParam, name: str = "FeatureMap") -> None:
        super(FeatureMap, self).__init__(name=name)
        self.param = param

    def build(self, input_shape: tf.TensorShape) -> None:
        self.conv1 = tf.keras.layers.Conv2D(
            input_shape=input_shape[1:],
            filters=self.param.conv1_filter,
            kernel_size=self.param.conv1_kernel,
            strides=self.param.conv1_stride,
            activation=tf.keras.activations.relu,
            name="feature_map_conv1")
        self.norm1 = tf.keras.layers.BatchNormalization(
            name="feature_map_norm1")
        self.conv2 = tf.keras.layers.Conv2D(
            filters=self.param.conv2_filter,
            kernel_size=self.param.conv2_kernel,
            strides=self.param.conv2_stride,
            activation=tf.keras.activations.relu,
            name="feature_map_conv2")
        self.norm2 = tf.keras.layers.BatchNormalization(
            name="feature_map_norm2")
        self.conv3 = tf.keras.layers.Conv2D(
            filters=self.param.conv3_filter,
            kernel_size=self.param.conv3_kernel,
            strides=self.param.conv3_stride,
            activation=tf.keras.activations.relu,
            name="feature_map_conv3")
        self.norm3 = tf.keras.layers.BatchNormalization(
            name="feature_map_norm3")
        self.conv4 = tf.keras.layers.Conv2D(
            filters=self.param.conv4_filter,
            kernel_size=self.param.conv4_kernel,
            strides=self.param.conv4_stride,
            activation=tf.keras.activations.relu,
            name="feature_map_conv4")
        self.norm4 = tf.keras.layers.BatchNormalization(
            name="feature_map_norm4")
        self.built = True

    def call(self, input_images: tf.Tensor) -> tf.Tensor:
        feature_maps = self.norm1(self.conv1(input_images))
        feature_maps = self.norm2(self.conv2(feature_maps))
        feature_maps = self.norm3(self.conv3(feature_maps))
        return self.norm4(self.conv4(feature_maps))

    def compute_output_shape(self,
                             input_shape: tf.TensorShape) -> tf.TensorShape:
        output_shape = self.conv1.compute_output_shape(input_shape)
        output_shape = self.conv2.compute_output_shape(output_shape)
        output_shape = self.conv3.compute_output_shape(output_shape)
        output_shape = self.conv4.compute_output_shape(output_shape)
        return output_shape


class PrimaryCap(tf.keras.layers.Layer):
    def __init__(self, param: CapsNetParam, name: str = "PrimaryCap") -> None:
        super(PrimaryCap, self).__init__(name=name)
        self.param = param

    def build(self, input_shape: tf.TensorShape) -> None:
        self.dconv = tf.keras.layers.Conv2D(
            input_shape=input_shape[1:],
            filters=self.param.dconv_filter,
            kernel_size=self.param.dconv_kernel,
            strides=self.param.dconv_stride,
            groups=self.param.dconv_filter,
            activation=tf.keras.activations.relu,
            name="primary_cap_dconv")
        self.reshape = tf.keras.layers.Reshape(
            target_shape=[-1, self.param.dim_primary_caps],
            name="primary_cap_reshape")
        self.squash = Squash(name="primary_cap_squash")
        self.built = True

    def call(self, feature_maps: tf.Tensor) -> tf.Tensor:
        dconv_outputs = self.dconv(feature_maps)
        return self.squash(self.reshape(dconv_outputs))

    def compute_output_shape(self,
                             input_shape: tf.TensorShape) -> tf.TensorShape:
        output_shape = self.dconv.compute_output_shape(input_shape)
        output_shape = self.reshape.compute_output_shape(output_shape)
        return output_shape


class DigitCap(tf.keras.layers.Layer):
    def __init__(self, param: CapsNetParam, name="DigitCap") -> None:
        super(DigitCap, self).__init__(name=name)
        self.param = param
        self.attention_coef = 1 / tf.sqrt(
            tf.cast(self.param.dim_primary_caps, dtype=tf.float32))

    def build(self, input_shape: tf.TensorShape) -> None:
        self.num_primary_caps = input_shape[1]
        self.dim_primary_caps = input_shape[2]
        self.W = self.add_weight(
            name="digit_caps_transform_tensor",
            shape=(self.param.num_digit_caps, self.num_primary_caps,
                   self.param.dim_digit_caps, self.dim_primary_caps),
            dtype=tf.float32,
            initializer="glorot_uniform",
            trainable=True)
        self.B = self.add_weight(
            name="digit_caps_log_priors",
            shape=[self.param.num_digit_caps, 1, self.num_primary_caps],
            dtype=tf.float32,
            initializer="glorot_uniform",
            trainable=True)
        self.squash = Squash(name="digit_cap_squash")
        self.built = True

    def call(self, primary_caps: tf.Tensor) -> tf.Tensor:
        # U.shape: [None, num_digit_caps, num_primary_caps, dim_primary_caps, 1]
        U = tf.expand_dims(tf.tile(tf.expand_dims(primary_caps, axis=1),
                                   [1, self.param.num_digit_caps, 1, 1]),
                           axis=-1,
                           name="digit_cap_inputs")
        # U_hat.shape: [None, num_digit_caps, num_primary_caps, dim_digit_caps]
        U_hat = tf.squeeze(tf.map_fn(lambda u_i: tf.matmul(self.W, u_i), U),
                           axis=-1,
                           name="digit_cap_predictions")
        # A.shape: [None, num_digit_caps, num_primary_caps, num_primary_caps]
        A = self.attention_coef * tf.matmul(
            U_hat, U_hat, transpose_b=True, name="digit_cap_attentions")
        # C.shape: [None, num_digit_caps, 1, num_primary_caps]
        C = tf.nn.softmax(tf.reduce_sum(A, axis=-2, keepdims=True),
                          axis=-2,
                          name="digit_cap_coupling_coefficients")
        # S.shape: [None, num_digit_caps, dim_digit_caps]
        S = tf.squeeze(tf.matmul(C + self.B, U_hat), axis=-2)
        return self.squash(S)

    def compute_output_shape(self,
                             input_shape: tf.TensorShape) -> tf.TensorShape:
        return tf.TensorShape([
            input_shape[0],
            self.param.num_digit_caps,
            self.param.dim_digit_caps,
        ])
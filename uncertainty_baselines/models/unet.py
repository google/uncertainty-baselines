# coding=utf-8
# Copyright 2024 The Uncertainty Baselines Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unet model."""

from typing import Iterable
import tensorflow as tf


class InstanceNormalization(tf.keras.layers.Layer):
  """Instance Normalization Layer (https://arxiv.org/abs/1607.08022)."""

  def __init__(self, epsilon=1e-5):
    super(InstanceNormalization, self).__init__()
    self.epsilon = epsilon

  def build(self, input_shape):
    self.scale = self.add_weight(
        name='scale',
        shape=input_shape[-1:],
        initializer=tf.random_normal_initializer(1., 0.02),
        trainable=True)

    self.offset = self.add_weight(
        name='offset',
        shape=input_shape[-1:],
        initializer='zeros',
        trainable=True)

  def call(self, x: tf.Tensor) -> tf.Tensor:
    mean, variance = tf.nn.moments(x, axes=[1, 2], keepdims=True)
    inv = tf.math.rsqrt(variance + self.epsilon)
    normalized = (x - mean) * inv
    return self.scale * normalized + self.offset


def upsample(filters, size, norm_type='batchnorm', apply_dropout=False):
  """Upsamples an input.

  Conv2DTranspose => Batchnorm => Dropout => Relu.

  Args:
    filters: number of filters.
    size: Filter size.
    norm_type: Normalization type; either 'batchnorm' or 'instancenorm'.
    apply_dropout: If True, adds the dropout layer.

  Returns:
    Upsample Sequential Model.
  """

  initializer = tf.random_normal_initializer(0., 0.02)

  result = tf.keras.Sequential()
  result.add(
      tf.keras.layers.Conv2DTranspose(
          filters,
          size,
          strides=2,
          padding='same',
          kernel_initializer=initializer,
          use_bias=False))

  if norm_type.lower() == 'batchnorm':
    result.add(tf.keras.layers.BatchNormalization())
  elif norm_type.lower() == 'instancenorm':
    result.add(InstanceNormalization())

  if apply_dropout:
    result.add(tf.keras.layers.Dropout(0.5))

  result.add(tf.keras.layers.ReLU())
  return result


def unet(
    input_shape: Iterable[int],
    filters: Iterable[int],
    num_classes: int = 4,
    trainable_encoder: bool = False,
    seed: int = 0,
) -> tf.keras.models.Model:
  """Build unet model for segmentation.

  https://www.tensorflow.org/tutorials/images/segmentation

  Args:
    input_shape: tf.Tensor.
    filters: filters in unet model.
    num_classes: number of classes.
    trainable_encoder: whether encoder is trainable.
    seed: tf.random seed.

  Returns:
    tf.keras.Model.
  """
  tf.random.set_seed(seed)
  base_model = tf.keras.applications.MobileNetV2(
      input_shape=input_shape, include_top=False)

  # Use the activations of these layers
  layer_names = [
      'block_1_expand_relu',  # 64x64
      'block_3_expand_relu',  # 32x32
      'block_6_expand_relu',  # 16x16
      'block_13_expand_relu',  # 8x8
      'block_16_project',  # 4x4
  ]
  base_model_outputs = [
      base_model.get_layer(name).output for name in layer_names
  ]

  # Create the feature extraction model
  down_stack = tf.keras.Model(
      inputs=base_model.input, outputs=base_model_outputs)

  down_stack.trainable = trainable_encoder

  up_stack = []
  for filter_ in filters:
    up_stack.append(upsample(filter_, 3))

  def unet_model(output_channels):
    inputs = tf.keras.layers.Input(shape=input_shape)

    # Downsampling through the model
    skips = down_stack(inputs)
    x = skips[-1]
    skips = reversed(skips[:-1])

    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
      x = up(x)
      concat = tf.keras.layers.Concatenate()
      x = concat([x, skip])

    # This is the last layer of the model
    last = tf.keras.layers.Conv2DTranspose(
        output_channels, 3, strides=2, padding='same')  # 64x64 -> 128x128

    x = last(x)
    return tf.keras.Model(inputs=inputs, outputs=x)

  return unet_model(num_classes)

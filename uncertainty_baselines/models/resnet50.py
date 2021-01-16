# coding=utf-8
# Copyright 2021 The Uncertainty Baselines Authors.
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

# Lint as: python3
"""ResNet50 Keras model for ImageNet."""

from typing import Any, Dict, Tuple

import tensorflow as tf


def _identity_block(
    input_tensor: tf.Tensor,
    kernel_size: int,
    filters: Tuple[int, int, int],
    stage: int,
    block: str,
    batch_norm_momentum: float,
    batch_norm_epsilon: float) -> tf.Tensor:
  """The identity block is the block that has no conv layer at shortcut.

  Args:
    input_tensor: input tensor
    kernel_size: default 3, the kernel size of middle conv layer at main path.
    filters: list of integers, the filters of 3 conv layer at main path.
    stage: integer, current stage label, used for generating layer names.
    block: 'a','b'..., current block label, used for generating layer names.
    batch_norm_momentum: the batch normalization EMA momentum.
    batch_norm_epsilon: the batch normalization epsilon.

  Returns:
      Output tensor for the block.
  """
  filters1, filters2, filters3 = filters
  # TODO(znado): support NCHW data_format.
  bn_axis = 3
  conv_name_base = 'res' + str(stage) + block + '_branch'
  bn_name_base = 'bn' + str(stage) + block + '_branch'

  x = tf.keras.layers.Conv2D(
      filters1,
      (1, 1),
      use_bias=False,
      kernel_initializer='he_normal',
      name=conv_name_base + '2a')(input_tensor)
  x = tf.keras.layers.BatchNormalization(
      axis=bn_axis,
      name=bn_name_base + '2a',
      momentum=batch_norm_momentum,
      epsilon=batch_norm_epsilon)(x)
  x = tf.keras.layers.Activation('relu')(x)

  x = tf.keras.layers.Conv2D(
      filters2,
      kernel_size,
      use_bias=False,
      padding='same',
      kernel_initializer='he_normal',
      name=conv_name_base + '2b')(x)
  x = tf.keras.layers.BatchNormalization(
      axis=bn_axis,
      name=bn_name_base + '2b',
      momentum=batch_norm_momentum,
      epsilon=batch_norm_epsilon)(x)
  x = tf.keras.layers.Activation('relu')(x)

  x = tf.keras.layers.Conv2D(
      filters3,
      (1, 1),
      use_bias=False,
      kernel_initializer='he_normal',
      name=conv_name_base + '2c')(x)
  x = tf.keras.layers.BatchNormalization(
      axis=bn_axis,
      name=bn_name_base + '2c',
      momentum=batch_norm_momentum,
      epsilon=batch_norm_epsilon)(x)

  x = tf.keras.layers.add([x, input_tensor])
  x = tf.keras.layers.Activation('relu')(x)
  return x


def _conv_block(
    input_tensor: tf.Tensor,
    kernel_size: int,
    filters: Tuple[int, int, int],
    stage: int,
    block: str,
    strides: Tuple[int, int] = (2, 2),
    batch_norm_momentum: float = 0.9,
    batch_norm_epsilon: float = 1e-5) -> tf.Tensor:
  """A block that has a conv layer at shortcut.

  Note that from stage 3, the second conv layer at main path is with
  strides=(2, 2) and the shortcut should have strides=(2, 2) as well.

  Args:
    input_tensor: input tensor
    kernel_size: the kernel size of middle conv layer at main path
    filters: list of integers, the filters of 3 conv layer at main path
    stage: integer, current stage label, used for generating layer names
    block: 'a','b'..., current block label, used for generating layer names
    strides: Strides for the second conv layer in the block.
    batch_norm_momentum: the batch normalization EMA momentum.
    batch_norm_epsilon: the batch normalization epsilon.

  Returns:
    Output tensor for the block.
  """
  filters1, filters2, filters3 = filters
  # TODO(znado): support NCHW data_format.
  bn_axis = 3
  conv_name_base = 'res' + str(stage) + block + '_branch'
  bn_name_base = 'bn' + str(stage) + block + '_branch'

  x = tf.keras.layers.Conv2D(
      filters1,
      (1, 1),
      use_bias=False,
      kernel_initializer='he_normal',
      name=conv_name_base + '2a')(input_tensor)
  x = tf.keras.layers.BatchNormalization(
      axis=bn_axis,
      name=bn_name_base + '2a',
      momentum=batch_norm_momentum,
      epsilon=batch_norm_epsilon)(x)
  x = tf.keras.layers.Activation('relu')(x)

  x = tf.keras.layers.Conv2D(
      filters2,
      kernel_size,
      strides=strides,
      padding='same',
      use_bias=False,
      kernel_initializer='he_normal',
      name=conv_name_base + '2b')(x)
  x = tf.keras.layers.BatchNormalization(
      axis=bn_axis,
      name=bn_name_base + '2b',
      momentum=batch_norm_momentum,
      epsilon=batch_norm_epsilon)(x)
  x = tf.keras.layers.Activation('relu')(x)

  x = tf.keras.layers.Conv2D(
      filters3,
      (1, 1),
      use_bias=False,
      kernel_initializer='he_normal',
      name=conv_name_base + '2c')(x)
  x = tf.keras.layers.BatchNormalization(
      axis=bn_axis,
      name=bn_name_base + '2c',
      momentum=batch_norm_momentum,
      epsilon=batch_norm_epsilon)(x)

  shortcut = tf.keras.layers.Conv2D(
      filters3,
      (1, 1),
      use_bias=False,
      strides=strides,
      kernel_initializer='he_normal',
      name=conv_name_base + '1')(input_tensor)
  shortcut = tf.keras.layers.BatchNormalization(
      axis=bn_axis,
      name=bn_name_base + '1',
      momentum=batch_norm_momentum,
      epsilon=batch_norm_epsilon)(shortcut)

  x = tf.keras.layers.add([x, shortcut])
  x = tf.keras.layers.Activation('relu')(x)
  return x


def create_model(
    batch_size: int,
    batch_norm_momentum: float = 0.9,
    batch_norm_epsilon: float = 1e-5,
    **unused_kwargs: Dict[str, Any]) -> tf.keras.Model:
  """Instantiates the ResNet50 architecture.

  Args:
    batch_size: int value of the static per_replica batch size.
    batch_norm_momentum: the batch normalization EMA momentum.
    batch_norm_epsilon: the batch normalization epsilon.

  Returns:
      A Keras model instance that returns the logits for the model.
  """
  # TODO(znado): support NCHW data format.
  bn_axis = 3
  input_layer = tf.keras.layers.Input(
      shape=(224, 224, 3), batch_size=batch_size)
  x = tf.keras.layers.ZeroPadding2D(
      padding=(3, 3),
      name='conv1_pad')(input_layer)
  x = tf.keras.layers.Conv2D(
      64,
      (7, 7),
      use_bias=False,
      strides=(2, 2),
      padding='valid',
      kernel_initializer='he_normal',
      name='conv1')(x)
  x = tf.keras.layers.BatchNormalization(
      axis=bn_axis,
      name='bn_conv1',
      momentum=batch_norm_momentum,
      epsilon=batch_norm_epsilon)(x)
  x = tf.keras.layers.Activation('relu')(x)
  x = tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

  x = _conv_block(
      x,
      kernel_size=3,
      filters=(64, 64, 256),
      stage=2,
      block='a',
      strides=(1, 1),
      batch_norm_momentum=batch_norm_momentum,
      batch_norm_epsilon=batch_norm_epsilon)
  x = _identity_block(
      x, 3, (64, 64, 256), 2, 'b', batch_norm_momentum, batch_norm_epsilon)
  x = _identity_block(
      x, 3, (64, 64, 256), 2, 'c', batch_norm_momentum, batch_norm_epsilon)

  x = _conv_block(
      x,
      kernel_size=3,
      filters=(128, 128, 512),
      stage=3,
      block='a',
      batch_norm_momentum=batch_norm_momentum,
      batch_norm_epsilon=batch_norm_epsilon)
  x = _identity_block(
      x, 3, (128, 128, 512), 3, 'b', batch_norm_momentum, batch_norm_epsilon)
  x = _identity_block(
      x, 3, (128, 128, 512), 3, 'c', batch_norm_momentum, batch_norm_epsilon)
  x = _identity_block(
      x, 3, (128, 128, 512), 3, 'd', batch_norm_momentum, batch_norm_epsilon)

  x = _conv_block(
      x,
      kernel_size=3,
      filters=(256, 256, 1024),
      stage=4,
      block='a',
      batch_norm_momentum=batch_norm_momentum,
      batch_norm_epsilon=batch_norm_epsilon)
  x = _identity_block(
      x, 3, (256, 256, 1024), 4, 'b', batch_norm_momentum, batch_norm_epsilon)
  x = _identity_block(
      x, 3, (256, 256, 1024), 4, 'c', batch_norm_momentum, batch_norm_epsilon)
  x = _identity_block(
      x, 3, (256, 256, 1024), 4, 'd', batch_norm_momentum, batch_norm_epsilon)
  x = _identity_block(
      x, 3, (256, 256, 1024), 4, 'e', batch_norm_momentum, batch_norm_epsilon)
  x = _identity_block(
      x, 3, (256, 256, 1024), 4, 'f', batch_norm_momentum, batch_norm_epsilon)

  x = _conv_block(
      x,
      kernel_size=3,
      filters=(512, 512, 2048),
      stage=5,
      block='a',
      batch_norm_momentum=batch_norm_momentum,
      batch_norm_epsilon=batch_norm_epsilon)
  x = _identity_block(
      x, 3, (512, 512, 2048), 5, 'b', batch_norm_momentum, batch_norm_epsilon)
  x = _identity_block(
      x, 3, (512, 512, 2048), 5, 'c', batch_norm_momentum, batch_norm_epsilon)

  x = tf.keras.layers.GlobalAveragePooling2D(name='avg_pool')(x)
  x = tf.keras.layers.Dense(
      1000,
      kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01))(x)
  return tf.keras.Model(input_layer, x, name='resnet50')

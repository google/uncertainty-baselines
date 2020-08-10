# coding=utf-8
# Copyright 2020 The Uncertainty Baselines Authors.
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

"""BatchEnsemble for a Wide ResNet architecture."""

import functools
import edward2 as ed
import tensorflow as tf


BatchNormalization = functools.partial(  # pylint: disable=invalid-name
    tf.keras.layers.BatchNormalization,
    epsilon=1e-5,  # using epsilon and momentum defaults from Torch
    momentum=0.9)
Conv2DBatchEnsemble = functools.partial(  # pylint: disable=invalid-name
    ed.layers.Conv2DBatchEnsemble,
    kernel_size=3,
    padding='same',
    use_bias=False,
    kernel_initializer='he_normal')


def make_sign_initializer(random_sign_init):
  if random_sign_init > 0:
    return ed.initializers.RandomSign(random_sign_init)
  else:
    return tf.keras.initializers.RandomNormal(mean=1.0,
                                              stddev=-random_sign_init)


def basic_block(inputs, filters, strides, ensemble_size, random_sign_init, l2):
  """Basic residual block of two 3x3 convs.

  Args:
    inputs: tf.Tensor.
    filters: Number of filters for Conv2D.
    strides: Stride dimensions for Conv2D.
    ensemble_size: Number of ensemble members.
    random_sign_init: Probability of 1 in random sign init.
    l2: L2 regularization coefficient.

  Returns:
    tf.Tensor.
  """
  x = inputs
  y = inputs
  y = BatchNormalization(beta_regularizer=tf.keras.regularizers.l2(l2),
                         gamma_regularizer=tf.keras.regularizers.l2(l2))(y)
  y = tf.keras.layers.Activation('relu')(y)
  y = Conv2DBatchEnsemble(
      filters,
      strides=strides,
      alpha_initializer=make_sign_initializer(random_sign_init),
      gamma_initializer=make_sign_initializer(random_sign_init),
      kernel_regularizer=tf.keras.regularizers.l2(l2),
      ensemble_size=ensemble_size)(y)
  y = BatchNormalization(beta_regularizer=tf.keras.regularizers.l2(l2),
                         gamma_regularizer=tf.keras.regularizers.l2(l2))(y)
  y = tf.keras.layers.Activation('relu')(y)
  y = Conv2DBatchEnsemble(
      filters,
      strides=1,
      alpha_initializer=make_sign_initializer(random_sign_init),
      gamma_initializer=make_sign_initializer(random_sign_init),
      kernel_regularizer=tf.keras.regularizers.l2(l2),
      ensemble_size=ensemble_size)(y)
  if not x.shape.is_compatible_with(y.shape):
    x = Conv2DBatchEnsemble(
        filters,
        kernel_size=1,
        strides=strides,
        alpha_initializer=make_sign_initializer(random_sign_init),
        gamma_initializer=make_sign_initializer(random_sign_init),
        kernel_regularizer=tf.keras.regularizers.l2(l2),
        ensemble_size=ensemble_size)(x)
  x = tf.keras.layers.add([x, y])
  return x


def group(inputs, filters, strides, num_blocks, **kwargs):
  """Group of residual blocks."""
  x = basic_block(inputs, filters=filters, strides=strides, **kwargs)
  for _ in range(num_blocks - 1):
    x = basic_block(x, filters=filters, strides=1, **kwargs)
  return x


def wide_resnet(input_shape, depth, width_multiplier, num_classes,
                ensemble_size, random_sign_init, l2):
  """Builds Wide ResNet.

  Following Zagoruyko and Komodakis (2016), it accepts a width multiplier on the
  number of filters. Using three groups of residual blocks, the network maps
  spatial features of size 32x32 -> 16x16 -> 8x8.

  Args:
    input_shape: tf.Tensor.
    depth: Total number of convolutional layers. "n" in WRN-n-k. It differs from
      He et al. (2015)'s notation which uses the maximum depth of the network
      counting non-conv layers like dense.
    width_multiplier: Integer to multiply the number of typical filters by. "k"
      in WRN-n-k.
    num_classes: Number of output classes.
    ensemble_size: Number of ensemble members.
    random_sign_init: Probability of 1 in random sign init.
    l2: L2 regularization coefficient.

  Returns:
    tf.keras.Model.
  """
  if (depth - 4) % 6 != 0:
    raise ValueError('depth should be 6n+4 (e.g., 16, 22, 28, 40).')
  num_blocks = (depth - 4) // 6
  inputs = tf.keras.layers.Input(shape=input_shape)
  x = Conv2DBatchEnsemble(
      16,
      strides=1,
      alpha_initializer=make_sign_initializer(random_sign_init),
      gamma_initializer=make_sign_initializer(random_sign_init),
      kernel_regularizer=tf.keras.regularizers.l2(l2),
      ensemble_size=ensemble_size)(inputs)
  for strides, filters in zip([1, 2, 2], [16, 32, 64]):
    x = group(x,
              filters=filters * width_multiplier,
              strides=strides,
              num_blocks=num_blocks,
              random_sign_init=random_sign_init,
              ensemble_size=ensemble_size,
              l2=l2)

  x = BatchNormalization(beta_regularizer=tf.keras.regularizers.l2(l2),
                         gamma_regularizer=tf.keras.regularizers.l2(l2))(x)
  x = tf.keras.layers.Activation('relu')(x)
  x = tf.keras.layers.AveragePooling2D(pool_size=8)(x)
  x = tf.keras.layers.Flatten()(x)
  x = ed.layers.DenseBatchEnsemble(
      num_classes,
      alpha_initializer=make_sign_initializer(random_sign_init),
      gamma_initializer=make_sign_initializer(random_sign_init),
      activation=None,
      kernel_initializer='he_normal',
      kernel_regularizer=tf.keras.regularizers.l2(l2),
      bias_regularizer=tf.keras.regularizers.l2(l2),
      ensemble_size=ensemble_size)(x)
  return tf.keras.Model(inputs=inputs, outputs=x)

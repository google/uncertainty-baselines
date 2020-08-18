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

"""Wide ResNet with dropout."""
import functools
import tensorflow as tf

BatchNormalization = functools.partial(  # pylint: disable=invalid-name
    tf.keras.layers.BatchNormalization,
    epsilon=1e-5,  # using epsilon and momentum defaults from Torch
    momentum=0.9)
Conv2D = functools.partial(  # pylint: disable=invalid-name
    tf.keras.layers.Conv2D,
    kernel_size=3,
    padding='same',
    use_bias=False,
    kernel_initializer='he_normal')


def apply_dropout(inputs, dropout_rate, filterwise_dropout):
  """Apply a dropout layer to the inputs."""
  if filterwise_dropout:
    return tf.keras.layers.Dropout(
        dropout_rate, noise_shape=[inputs.shape[0], 1, 1, inputs.shape[3]
                                  ])(inputs, training=True)
  else:
    return tf.keras.layers.Dropout(dropout_rate)(inputs, training=True)


def basic_block(inputs, filters, strides, l2, dropout_rate, residual_dropout,
                filterwise_dropout):
  """Basic residual block of two 3x3 convs.

  Args:
    inputs: tf.Tensor.
    filters: Number of filters for Conv2D.
    strides: Stride dimensions for Conv2D.
    l2: L2 regularization coefficient.
    dropout_rate: Dropout rate.
    residual_dropout: Apply dropout only to the residual connections.
    filterwise_dropout: Dropout whole convolutional filters instead of
      individual values in the feature map.

  Returns:
    tf.Tensor.
  """
  x = inputs
  y = inputs
  y = BatchNormalization(beta_regularizer=tf.keras.regularizers.l2(l2),
                         gamma_regularizer=tf.keras.regularizers.l2(l2))(y)
  y = tf.keras.layers.Activation('relu')(y)

  if not residual_dropout:
    y = apply_dropout(y, dropout_rate, filterwise_dropout)

  y = Conv2D(filters,
             strides=strides,
             kernel_regularizer=tf.keras.regularizers.l2(l2))(y)

  if residual_dropout:
    y = apply_dropout(y, dropout_rate, filterwise_dropout)

  y = BatchNormalization(beta_regularizer=tf.keras.regularizers.l2(l2),
                         gamma_regularizer=tf.keras.regularizers.l2(l2))(y)
  y = tf.keras.layers.Activation('relu')(y)
  if not residual_dropout:
    y = apply_dropout(y, dropout_rate, filterwise_dropout)

  y = Conv2D(filters,
             strides=1,
             kernel_regularizer=tf.keras.regularizers.l2(l2))(y)
  if not x.shape.is_compatible_with(y.shape):
    x = Conv2D(filters,
               kernel_size=1,
               strides=strides,
               kernel_regularizer=tf.keras.regularizers.l2(l2))(x)
    if not residual_dropout:
      y = apply_dropout(y, dropout_rate, filterwise_dropout)
  x = tf.keras.layers.add([x, y])
  return x


def group(inputs, filters, strides, num_blocks, l2, dropout_rate,
          residual_dropout, filterwise_dropout):
  """Group of residual blocks."""
  x = basic_block(inputs,
                  filters=filters,
                  strides=strides,
                  l2=l2,
                  dropout_rate=dropout_rate,
                  residual_dropout=residual_dropout,
                  filterwise_dropout=filterwise_dropout)
  for _ in range(num_blocks - 1):
    x = basic_block(x,
                    filters=filters,
                    strides=1,
                    l2=l2,
                    dropout_rate=dropout_rate,
                    residual_dropout=residual_dropout,
                    filterwise_dropout=filterwise_dropout)
  return x


def wide_resnet_dropout(input_shape, depth, width_multiplier, num_classes, l2,
                        dropout_rate, residual_dropout, filterwise_dropout):
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
    l2: L2 regularization coefficient.
    dropout_rate: Dropout rate.
    residual_dropout: Apply dropout only to the residual connections.
    filterwise_dropout: Dropout whole convolutional filters instead of
      individual values in the feature map.

  Returns:
    tf.keras.Model.
  """
  if (depth - 4) % 6 != 0:
    raise ValueError('depth should be 6n+4 (e.g., 16, 22, 28, 40).')
  num_blocks = (depth - 4) // 6
  inputs = tf.keras.layers.Input(shape=input_shape)
  x = Conv2D(16,
             strides=1,
             kernel_regularizer=tf.keras.regularizers.l2(l2))(inputs)
  if not residual_dropout:
    x = apply_dropout(x, dropout_rate, filterwise_dropout)
  x = group(x,
            filters=16 * width_multiplier,
            strides=1,
            num_blocks=num_blocks,
            l2=l2,
            dropout_rate=dropout_rate,
            residual_dropout=residual_dropout,
            filterwise_dropout=filterwise_dropout)
  x = group(x,
            filters=32 * width_multiplier,
            strides=2,
            num_blocks=num_blocks,
            l2=l2,
            dropout_rate=dropout_rate,
            residual_dropout=residual_dropout,
            filterwise_dropout=filterwise_dropout)
  x = group(x,
            filters=64 * width_multiplier,
            strides=2,
            num_blocks=num_blocks,
            l2=l2,
            dropout_rate=dropout_rate,
            residual_dropout=residual_dropout,
            filterwise_dropout=filterwise_dropout)
  x = BatchNormalization(beta_regularizer=tf.keras.regularizers.l2(l2),
                         gamma_regularizer=tf.keras.regularizers.l2(l2))(x)
  x = tf.keras.layers.Activation('relu')(x)
  x = tf.keras.layers.AveragePooling2D(pool_size=8)(x)
  x = tf.keras.layers.Flatten()(x)
  x = tf.keras.layers.Dense(
      num_classes,
      kernel_initializer='he_normal',
      kernel_regularizer=tf.keras.regularizers.l2(l2),
      bias_regularizer=tf.keras.regularizers.l2(l2))(x)
  return tf.keras.Model(inputs=inputs, outputs=x)

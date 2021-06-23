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

"""Wide ResNet with variational Bayesian layers."""
import functools
from absl import logging
import numpy as np
import tensorflow as tf

try:
  import edward2 as ed  # pylint: disable=g-import-not-at-top
except ImportError:
  logging.warning('Skipped edward2 import due to ImportError.', exc_info=True)

BatchNormalization = functools.partial(  # pylint: disable=invalid-name
    tf.keras.layers.BatchNormalization,
    epsilon=1e-5,  # using epsilon and momentum defaults from Torch
    momentum=0.9)
Conv2DFlipout = functools.partial(  # pylint: disable=invalid-name
    ed.layers.Conv2DFlipout,
    kernel_size=3,
    padding='same',
    use_bias=False)


def basic_block(inputs, filters, strides, prior_stddev, dataset_size,
                stddev_init):
  """Basic residual block of two 3x3 convs.

  Args:
    inputs: tf.Tensor.
    filters: Number of filters for Conv2D.
    strides: Stride dimensions for Conv2D.
    prior_stddev: Fixed standard deviation for weight prior.
    dataset_size: Dataset size to properly scale the KL.
    stddev_init: float to initialize variational posterior stddev parameters.

  Returns:
    tf.Tensor.
  """
  x = inputs
  y = inputs
  y = BatchNormalization()(y)
  y = tf.keras.layers.Activation('relu')(y)
  y = Conv2DFlipout(
      filters,
      strides=strides,
      kernel_initializer=ed.initializers.TrainableHeNormal(
          stddev_initializer=tf.keras.initializers.TruncatedNormal(
              mean=np.log(np.expm1(stddev_init)), stddev=0.1)),
      kernel_regularizer=ed.regularizers.NormalKLDivergenceWithTiedMean(
          stddev=prior_stddev, scale_factor=1./dataset_size))(y)
  y = BatchNormalization()(y)
  y = tf.keras.layers.Activation('relu')(y)
  y = Conv2DFlipout(
      filters,
      strides=1,
      kernel_initializer=ed.initializers.TrainableHeNormal(
          stddev_initializer=tf.keras.initializers.TruncatedNormal(
              mean=np.log(np.expm1(stddev_init)), stddev=0.1)),
      kernel_regularizer=ed.regularizers.NormalKLDivergenceWithTiedMean(
          stddev=prior_stddev, scale_factor=1./dataset_size))(y)
  if not x.shape.is_compatible_with(y.shape):
    x = Conv2DFlipout(
        filters,
        kernel_size=1,
        strides=strides,
        kernel_initializer=ed.initializers.TrainableHeNormal(
            stddev_initializer=tf.keras.initializers.TruncatedNormal(
                mean=np.log(np.expm1(stddev_init)), stddev=0.1)),
        kernel_regularizer=ed.regularizers.NormalKLDivergenceWithTiedMean(
            stddev=prior_stddev, scale_factor=1./dataset_size))(x)
  x = tf.keras.layers.add([x, y])
  return x


def group(inputs, filters, strides, num_blocks, **kwargs):
  """Group of residual blocks."""
  x = basic_block(inputs, filters=filters, strides=strides, **kwargs)
  for _ in range(num_blocks - 1):
    x = basic_block(x, filters=filters, strides=1, **kwargs)
  return x


def wide_resnet_variational(input_shape,
                            depth,
                            width_multiplier,
                            num_classes,
                            prior_stddev,
                            dataset_size,
                            stddev_init):
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
    prior_stddev: Fixed standard deviation for weight prior.
    dataset_size: Dataset size to properly scale the KL.
    stddev_init: float to initialize variational posterior stddev parameters.

  Returns:
    tf.keras.Model.
  """
  if (depth - 4) % 6 != 0:
    raise ValueError('depth should be 6n+4 (e.g., 16, 22, 28, 40).')
  num_blocks = (depth - 4) // 6
  inputs = tf.keras.layers.Input(shape=input_shape)
  x = Conv2DFlipout(
      16,
      strides=1,
      kernel_initializer=ed.initializers.TrainableHeNormal(
          stddev_initializer=tf.keras.initializers.TruncatedNormal(
              mean=np.log(np.expm1(stddev_init)), stddev=0.1)),
      kernel_regularizer=ed.regularizers.NormalKLDivergenceWithTiedMean(
          stddev=prior_stddev, scale_factor=1./dataset_size))(inputs)
  for strides, filters in zip([1, 2, 2], [16, 32, 64]):
    x = group(x,
              filters=filters * width_multiplier,
              strides=strides,
              num_blocks=num_blocks,
              prior_stddev=prior_stddev,
              dataset_size=dataset_size,
              stddev_init=stddev_init)

  x = BatchNormalization()(x)
  x = tf.keras.layers.Activation('relu')(x)
  x = tf.keras.layers.AveragePooling2D(pool_size=8)(x)
  x = tf.keras.layers.Flatten()(x)
  x = ed.layers.DenseFlipout(
      num_classes,
      kernel_initializer=ed.initializers.TrainableHeNormal(
          stddev_initializer=tf.keras.initializers.TruncatedNormal(
              mean=np.log(np.expm1(stddev_init)), stddev=0.1)),
      kernel_regularizer=ed.regularizers.NormalKLDivergenceWithTiedMean(
          stddev=prior_stddev, scale_factor=1./dataset_size))(x)
  return tf.keras.Model(inputs=inputs, outputs=x)

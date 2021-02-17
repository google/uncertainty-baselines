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

"""ResNet50 Keras model with variational Bayesian layers."""

import functools
import string
import warnings

import numpy as np
import tensorflow as tf

try:
  import edward2 as ed  # pylint: disable=g-import-not-at-top
except ImportError as e:
  warnings.warn(f'Skipped due to ImportError: {e}')

BatchNormalization = functools.partial(  # pylint: disable=invalid-name
    tf.keras.layers.BatchNormalization,
    epsilon=1e-5,  # using epsilon and momentum defaults from Torch
    momentum=0.9)
Conv2DFlipout = functools.partial(  # pylint: disable=invalid-name
    ed.layers.Conv2DFlipout, use_bias=False)


def bottleneck_block(inputs, filters, stage, block, strides, prior_stddev,
                     dataset_size, stddev_init):
  """Residual block with 1x1 -> 3x3 -> 1x1 convs in main path.

  Note that strides appear in the second conv (3x3) rather than the first (1x1).
  This is also known as "ResNet v1.5" as it differs from He et al. (2015)
  (http://torch.ch/blog/2016/02/04/resnets.html).

  Args:
    inputs: tf.Tensor.
    filters: list of integers, the filters of 3 conv layer at main path
    stage: integer, current stage label, used for generating layer names
    block: 'a','b'..., current block label, used for generating layer names
    strides: Strides for the second conv layer in the block.
    prior_stddev: Fixed standard deviation for weight prior.
    dataset_size: Dataset size to properly scale the KL.
    stddev_init: float to initialize variational posterior stddev parameters.

  Returns:
    tf.Tensor.
  """
  filters1, filters2, filters3 = filters
  conv_name_base = 'res' + str(stage) + block + '_branch'
  bn_name_base = 'bn' + str(stage) + block + '_branch'

  x = Conv2DFlipout(
      filters1,
      kernel_size=1,
      kernel_initializer=ed.initializers.TrainableHeNormal(
          stddev_initializer=tf.keras.initializers.TruncatedNormal(
              mean=np.log(np.expm1(stddev_init)), stddev=0.1)),
      kernel_regularizer=ed.regularizers.NormalKLDivergenceWithTiedMean(
          stddev=prior_stddev, scale_factor=1. / dataset_size),
      name=conv_name_base + '2a')(
          inputs)
  x = BatchNormalization(name=bn_name_base + '2a')(x)
  x = tf.keras.layers.Activation('relu')(x)

  x = Conv2DFlipout(
      filters2,
      kernel_size=3,
      strides=strides,
      padding='same',
      kernel_initializer=ed.initializers.TrainableHeNormal(
          stddev_initializer=tf.keras.initializers.TruncatedNormal(
              mean=np.log(np.expm1(stddev_init)), stddev=0.1)),
      kernel_regularizer=ed.regularizers.NormalKLDivergenceWithTiedMean(
          stddev=prior_stddev, scale_factor=1. / dataset_size),
      name=conv_name_base + '2b')(
          x)
  x = BatchNormalization(name=bn_name_base + '2b')(x)
  x = tf.keras.layers.Activation('relu')(x)

  x = Conv2DFlipout(
      filters3,
      kernel_size=1,
      kernel_initializer=ed.initializers.TrainableHeNormal(
          stddev_initializer=tf.keras.initializers.TruncatedNormal(
              mean=np.log(np.expm1(stddev_init)), stddev=0.1)),
      kernel_regularizer=ed.regularizers.NormalKLDivergenceWithTiedMean(
          stddev=prior_stddev, scale_factor=1. / dataset_size),
      name=conv_name_base + '2c')(
          x)
  x = BatchNormalization(name=bn_name_base + '2c')(x)

  shortcut = inputs
  if not x.shape.is_compatible_with(shortcut.shape):
    shortcut = Conv2DFlipout(
        filters3,
        kernel_size=1,
        strides=strides,
        kernel_initializer=ed.initializers.TrainableHeNormal(
            stddev_initializer=tf.keras.initializers.TruncatedNormal(
                mean=np.log(np.expm1(stddev_init)), stddev=0.1)),
        kernel_regularizer=ed.regularizers.NormalKLDivergenceWithTiedMean(
            stddev=prior_stddev, scale_factor=1. / dataset_size),
        name=conv_name_base + '1')(
            shortcut)
    shortcut = BatchNormalization(name=bn_name_base + '1')(shortcut)

  x = tf.keras.layers.add([x, shortcut])
  x = tf.keras.layers.Activation('relu')(x)
  return x


def group(inputs, filters, num_blocks, stage, strides, prior_stddev,
          dataset_size, stddev_init):
  """ResNet group."""
  blocks = string.ascii_lowercase
  x = bottleneck_block(
      inputs,
      filters,
      stage,
      block=blocks[0],
      strides=strides,
      prior_stddev=prior_stddev,
      dataset_size=dataset_size,
      stddev_init=stddev_init)
  for i in range(num_blocks - 1):
    x = bottleneck_block(
        x,
        filters,
        stage,
        block=blocks[i + 1],
        strides=1,
        prior_stddev=prior_stddev,
        dataset_size=dataset_size,
        stddev_init=stddev_init)
  return x


def resnet50_variational(input_shape,
                         num_classes,
                         prior_stddev,
                         dataset_size,
                         stddev_init,
                         omit_last_layer=False):
  """Builds variational inference ResNet50.

  Using strided conv, pooling, four groups of residual blocks, and pooling, the
  network maps spatial features of size 224x224 -> 112x112 -> 56x56 -> 28x28 ->
  14x14 -> 7x7 (Table 1 of He et al. (2015)).

  Args:
    input_shape: Shape tuple of input excluding batch dimension.
    num_classes: Number of output classes.
    prior_stddev: Fixed standard deviation for weight prior.
    dataset_size: Dataset size to properly scale the KL.
    stddev_init: float to initialize variational posterior stddev parameters.
    omit_last_layer: Optional. Omits the last pooling layer if it is to True.

  Returns:
    tf.keras.Model.
  """
  inputs = tf.keras.layers.Input(shape=input_shape)
  x = tf.keras.layers.ZeroPadding2D(padding=3, name='conv1_pad')(inputs)
  x = Conv2DFlipout(
      64,
      kernel_size=7,
      strides=2,
      padding='valid',
      kernel_initializer=ed.initializers.TrainableHeNormal(
          stddev_initializer=tf.keras.initializers.TruncatedNormal(
              mean=np.log(np.expm1(stddev_init)), stddev=0.1)),
      kernel_regularizer=ed.regularizers.NormalKLDivergenceWithTiedMean(
          stddev=prior_stddev, scale_factor=1. / dataset_size),
      name='conv1')(
          x)
  x = BatchNormalization(name='bn_conv1')(x)
  x = tf.keras.layers.Activation('relu')(x)
  x = tf.keras.layers.MaxPooling2D(3, strides=2, padding='same')(x)
  x = group(
      x, [64, 64, 256],
      stage=2,
      num_blocks=3,
      strides=1,
      prior_stddev=prior_stddev,
      dataset_size=dataset_size,
      stddev_init=stddev_init)
  x = group(
      x, [128, 128, 512],
      stage=3,
      num_blocks=4,
      strides=2,
      prior_stddev=prior_stddev,
      dataset_size=dataset_size,
      stddev_init=stddev_init)
  x = group(
      x, [256, 256, 1024],
      stage=4,
      num_blocks=6,
      strides=2,
      prior_stddev=prior_stddev,
      dataset_size=dataset_size,
      stddev_init=stddev_init)
  x = group(
      x, [512, 512, 2048],
      stage=5,
      num_blocks=3,
      strides=2,
      prior_stddev=prior_stddev,
      dataset_size=dataset_size,
      stddev_init=stddev_init)

  if omit_last_layer:
    return tf.keras.Model(inputs=inputs, outputs=x, name='resnet50_variational')

  x = tf.keras.layers.GlobalAveragePooling2D(name='avg_pool')(x)
  x = ed.layers.DenseFlipout(
      num_classes,
      activation=None,
      kernel_initializer=ed.initializers.TrainableHeNormal(
          stddev_initializer=tf.keras.initializers.TruncatedNormal(
              mean=np.log(np.expm1(stddev_init)), stddev=0.1)),
      kernel_regularizer=ed.regularizers.NormalKLDivergenceWithTiedMean(
          stddev=prior_stddev, scale_factor=1. / dataset_size),
      name='fc1000')(
          x)

  return tf.keras.Model(inputs=inputs, outputs=x, name='resnet50_variational')

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

"""ResNet50 model."""

import string
import edward2 as ed
import tensorflow as tf

# Use batch normalization defaults from Pytorch.
BATCH_NORM_DECAY = 0.9
BATCH_NORM_EPSILON = 1e-5


def bottleneck_block(inputs,
                     filters,
                     stage,
                     block,
                     strides):
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

  Returns:
    tf.Tensor.
  """
  filters1, filters2, filters3 = filters
  conv_name_base = 'res' + str(stage) + block + '_branch'
  bn_name_base = 'bn' + str(stage) + block + '_branch'

  x = tf.keras.layers.Conv2D(
      filters1,
      kernel_size=1,
      use_bias=False,
      kernel_initializer='he_normal',
      name=conv_name_base + '2a')(inputs)
  x = tf.keras.layers.BatchNormalization(
      momentum=BATCH_NORM_DECAY,
      epsilon=BATCH_NORM_EPSILON,
      name=bn_name_base + '2a')(x)
  x = tf.keras.layers.Activation('relu')(x)

  x = tf.keras.layers.Conv2D(
      filters2,
      kernel_size=3,
      strides=strides,
      padding='same',
      use_bias=False,
      kernel_initializer='he_normal',
      name=conv_name_base + '2b')(x)
  x = tf.keras.layers.BatchNormalization(
      momentum=BATCH_NORM_DECAY,
      epsilon=BATCH_NORM_EPSILON,
      name=bn_name_base + '2b')(x)
  x = tf.keras.layers.Activation('relu')(x)

  x = tf.keras.layers.Conv2D(
      filters3,
      kernel_size=1,
      use_bias=False,
      kernel_initializer='he_normal',
      name=conv_name_base + '2c')(x)
  x = tf.keras.layers.BatchNormalization(
      momentum=BATCH_NORM_DECAY,
      epsilon=BATCH_NORM_EPSILON,
      name=bn_name_base + '2c')(x)

  shortcut = inputs
  if not x.shape.is_compatible_with(shortcut.shape):
    shortcut = tf.keras.layers.Conv2D(
        filters3,
        kernel_size=1,
        use_bias=False,
        strides=strides,
        kernel_initializer='he_normal',
        name=conv_name_base + '1')(shortcut)
    shortcut = tf.keras.layers.BatchNormalization(
        momentum=BATCH_NORM_DECAY,
        epsilon=BATCH_NORM_EPSILON,
        name=bn_name_base + '1')(shortcut)

  x = tf.keras.layers.add([x, shortcut])
  x = tf.keras.layers.Activation('relu')(x)
  return x


def group(inputs, filters, num_blocks, stage, strides):
  blocks = string.ascii_lowercase
  x = bottleneck_block(inputs, filters, stage, block=blocks[0], strides=strides)
  for i in range(num_blocks - 1):
    x = bottleneck_block(x, filters, stage, block=blocks[i + 1], strides=1)
  return x


def resnet50_mimo(input_shape, num_classes, ensemble_size, width_multiplier=1):
  """Builds a multiheaded ResNet50.

  Using strided conv, pooling, four groups of residual blocks, and pooling, the
  network maps spatial features of size 224x224 -> 112x112 -> 56x56 -> 28x28 ->
  14x14 -> 7x7 (Table 1 of He et al. (2015)).

  Args:
    input_shape: Shape tuple of input excluding batch dimension.
    num_classes: Number of output classes.
    ensemble_size: Number of ensembles i.e. number of heads and inputs.
    width_multiplier: Multiply the number of filters for wide ResNet.

  Returns:
    tf.keras.Model.
  """
  input_shape = list(input_shape)
  inputs = tf.keras.layers.Input(shape=input_shape)
  x = tf.keras.layers.Permute([2, 3, 4, 1])(inputs)
  assert ensemble_size == input_shape[0]
  x = tf.keras.layers.Reshape(list(input_shape[1:-1]) +
                              [input_shape[-1] * ensemble_size])(
                                  x)
  x = tf.keras.layers.ZeroPadding2D(padding=3, name='conv1_pad')(x)
  x = tf.keras.layers.Conv2D(
      width_multiplier * 64,
      kernel_size=7,
      strides=2,
      padding='valid',
      use_bias=False,
      kernel_initializer='he_normal',
      name='conv1')(x)
  x = tf.keras.layers.BatchNormalization(
      momentum=BATCH_NORM_DECAY,
      epsilon=BATCH_NORM_EPSILON,
      name='bn_conv1')(x)
  x = tf.keras.layers.Activation('relu')(x)
  x = tf.keras.layers.MaxPooling2D(3, strides=2, padding='same')(x)
  x = group(x, [width_multiplier * 64,
                width_multiplier * 64,
                width_multiplier * 256], stage=2, num_blocks=3, strides=1)
  x = group(x, [width_multiplier * 128,
                width_multiplier * 128,
                width_multiplier * 512], stage=3, num_blocks=4, strides=2)
  x = group(x, [width_multiplier * 256,
                width_multiplier * 256,
                width_multiplier * 1024], stage=4, num_blocks=6, strides=2)
  x = group(x, [width_multiplier * 512,
                width_multiplier * 512,
                width_multiplier * 2048], stage=5, num_blocks=3, strides=2)
  x = tf.keras.layers.GlobalAveragePooling2D(name='avg_pool')(x)
  x = ed.layers.DenseMultihead(
      num_classes,
      activation=None,
      kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01),
      ensemble_size=ensemble_size,
      name='fc1000')(
          x)
  return tf.keras.Model(inputs=inputs, outputs=x, name='resnet50')

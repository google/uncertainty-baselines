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

"""ResNet50 model."""
import functools
import string

import tensorflow as tf

# Use batch normalization defaults from Pytorch.
BATCH_NORM_DECAY = 0.9
BATCH_NORM_EPSILON = 1e-5

# Default layers.
DEFAULT_CONV_LAYER = functools.partial(tf.keras.layers.Conv2D, padding='same')

DEFAULT_OUTPUT_LAYER = functools.partial(
    tf.keras.layers.Dense,
    activation=None,
    kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01),
    name='fc1000')


def bottleneck_block(inputs, filters, stage, block, strides, conv_layer):
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
    conv_layer: tf.keras.layers.Layer.

  Returns:
    tf.Tensor.
  """
  filters1, filters2, filters3 = filters
  conv_name_base = 'res' + str(stage) + block + '_branch'
  bn_name_base = 'bn' + str(stage) + block + '_branch'

  x = conv_layer(
      filters1,
      kernel_size=1,
      use_bias=False,
      kernel_initializer='he_normal',
      name=conv_name_base + '2a')(
          inputs)
  x = tf.keras.layers.BatchNormalization(
      momentum=BATCH_NORM_DECAY,
      epsilon=BATCH_NORM_EPSILON,
      name=bn_name_base + '2a')(x)
  x = tf.keras.layers.Activation('relu')(x)

  x = conv_layer(
      filters2,
      kernel_size=3,
      strides=strides,
      padding='same',
      use_bias=False,
      kernel_initializer='he_normal',
      name=conv_name_base + '2b')(
          x)
  x = tf.keras.layers.BatchNormalization(
      momentum=BATCH_NORM_DECAY,
      epsilon=BATCH_NORM_EPSILON,
      name=bn_name_base + '2b')(x)
  x = tf.keras.layers.Activation('relu')(x)

  x = conv_layer(
      filters3,
      kernel_size=1,
      use_bias=False,
      kernel_initializer='he_normal',
      name=conv_name_base + '2c')(
          x)
  x = tf.keras.layers.BatchNormalization(
      momentum=BATCH_NORM_DECAY,
      epsilon=BATCH_NORM_EPSILON,
      name=bn_name_base + '2c')(x)

  shortcut = inputs
  if not x.shape.is_compatible_with(shortcut.shape):
    shortcut = conv_layer(
        filters3,
        kernel_size=1,
        use_bias=False,
        strides=strides,
        kernel_initializer='he_normal',
        name=conv_name_base + '1')(
            shortcut)
    shortcut = tf.keras.layers.BatchNormalization(
        momentum=BATCH_NORM_DECAY,
        epsilon=BATCH_NORM_EPSILON,
        name=bn_name_base + '1')(shortcut)

  x = tf.keras.layers.add([x, shortcut])
  x = tf.keras.layers.Activation('relu')(x)
  return x


def group(inputs, filters, num_blocks, stage, strides, conv_layer):
  """Group of residual blocks."""
  blocks = string.ascii_lowercase
  x = bottleneck_block(
      inputs,
      filters,
      stage,
      block=blocks[0],
      strides=strides,
      conv_layer=conv_layer)
  for i in range(num_blocks - 1):
    x = bottleneck_block(
        x,
        filters,
        stage,
        block=blocks[i + 1],
        strides=1,
        conv_layer=conv_layer)
  return x


def resnet50(input_shape,
             batch_size,
             num_classes,
             conv_layer=DEFAULT_CONV_LAYER,
             output_layer=DEFAULT_OUTPUT_LAYER):
  """Builds ResNet50.

  Using strided conv, pooling, four groups of residual blocks, and pooling, the
  network maps spatial features of size 224x224 -> 112x112 -> 56x56 -> 28x28 ->
  14x14 -> 7x7 (Table 1 of He et al. (2015)).

  Args:
    input_shape: Shape tuple of input excluding batch dimension.
    batch_size: The batch size of the input layer. Required by the spectral
      normalization.
    num_classes: Number of output classes.
    conv_layer: tf.keras.layers.Layer.
    output_layer: tf.keras.layers.Layer.

  Returns:
    tf.keras.Model.
  """
  inputs = tf.keras.layers.Input(shape=input_shape, batch_size=batch_size)
  x = tf.keras.layers.ZeroPadding2D(padding=3, name='conv1_pad')(inputs)
  # TODO(jereliu): apply SpectralNormalization to input layer as well.
  x = tf.keras.layers.Conv2D(
      64,
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
  x = group(
      x, [64, 64, 256], stage=2, num_blocks=3, strides=1, conv_layer=conv_layer)
  x = group(
      x, [128, 128, 512],
      stage=3,
      num_blocks=4,
      strides=2,
      conv_layer=conv_layer)
  x = group(
      x, [256, 256, 1024],
      stage=4,
      num_blocks=6,
      strides=2,
      conv_layer=conv_layer)
  x = group(
      x, [512, 512, 2048],
      stage=5,
      num_blocks=3,
      strides=2,
      conv_layer=conv_layer)
  x = tf.keras.layers.GlobalAveragePooling2D(name='avg_pool')(x)

  outputs = output_layer(num_classes)(x)

  return tf.keras.Model(inputs=inputs, outputs=outputs, name='resnet50')

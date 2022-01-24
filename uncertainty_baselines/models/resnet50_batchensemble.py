# coding=utf-8
# Copyright 2022 The Uncertainty Baselines Authors.
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

"""BatchEnsemble ResNet50."""

import functools
import string
from absl import logging
import tensorflow as tf

try:
  import edward2 as ed  # pylint: disable=g-import-not-at-top
except ImportError:
  logging.warning('Skipped edward2 import due to ImportError.', exc_info=True)

# Use batch normalization defaults from Pytorch.
BATCH_NORM_DECAY = 0.9
BATCH_NORM_EPSILON = 1e-5

EnsembleBatchNormalization = functools.partial(  # pylint: disable=invalid-name
    ed.layers.EnsembleSyncBatchNorm,
    epsilon=BATCH_NORM_EPSILON,
    momentum=BATCH_NORM_DECAY)
BatchNormalization = functools.partial(  # pylint: disable=invalid-name
    tf.keras.layers.BatchNormalization,
    epsilon=BATCH_NORM_EPSILON,
    momentum=BATCH_NORM_DECAY)


def make_random_sign_initializer(random_sign_init):
  if random_sign_init > 0:
    initializer = ed.initializers.RandomSign(random_sign_init)
  else:
    initializer = tf.keras.initializers.RandomNormal(mean=1.0,
                                                     stddev=-random_sign_init)
  return initializer


def bottleneck_block(inputs,
                     filters,
                     stage,
                     block,
                     strides,
                     ensemble_size,
                     random_sign_init,
                     use_ensemble_bn):
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
    ensemble_size: the ensemble size, when it is one, it goes back to the
        single model case.
    random_sign_init: whether uses random sign initializer to initializer
        the fast weights.
    use_ensemble_bn: whether to use ensemble batch norm.

  Returns:
    tf.Tensor.
  """
  filters1, filters2, filters3 = filters
  conv_name_base = 'res' + str(stage) + block + '_branch'
  bn_name_base = 'bn' + str(stage) + block + '_branch'

  x = ed.layers.Conv2DBatchEnsemble(
      filters1,
      kernel_size=1,
      use_bias=False,
      kernel_initializer='he_normal',
      alpha_initializer=make_random_sign_initializer(random_sign_init),
      gamma_initializer=make_random_sign_initializer(random_sign_init),
      name=conv_name_base + '2a',
      ensemble_size=ensemble_size)(inputs)
  if use_ensemble_bn:
    x = EnsembleBatchNormalization(
        ensemble_size=ensemble_size,
        name=bn_name_base+'2a')(x)
  else:
    x = BatchNormalization(name=bn_name_base+'2a')(x)
  x = tf.keras.layers.Activation('relu')(x)

  x = ed.layers.Conv2DBatchEnsemble(
      filters2,
      kernel_size=3,
      strides=strides,
      padding='same',
      use_bias=False,
      kernel_initializer='he_normal',
      alpha_initializer=make_random_sign_initializer(random_sign_init),
      gamma_initializer=make_random_sign_initializer(random_sign_init),
      name=conv_name_base + '2b',
      ensemble_size=ensemble_size)(x)
  if use_ensemble_bn:
    x = EnsembleBatchNormalization(
        ensemble_size=ensemble_size,
        name=bn_name_base+'2b')(x)
  else:
    x = BatchNormalization(name=bn_name_base+'2b')(x)
  x = tf.keras.layers.Activation('relu')(x)

  x = ed.layers.Conv2DBatchEnsemble(
      filters3,
      kernel_size=1,
      use_bias=False,
      kernel_initializer='he_normal',
      alpha_initializer=make_random_sign_initializer(random_sign_init),
      gamma_initializer=make_random_sign_initializer(random_sign_init),
      name=conv_name_base + '2c',
      ensemble_size=ensemble_size)(x)
  if use_ensemble_bn:
    x = EnsembleBatchNormalization(
        ensemble_size=ensemble_size,
        name=bn_name_base+'2c')(x)
  else:
    x = BatchNormalization(name=bn_name_base+'2c')(x)

  shortcut = inputs
  if not x.shape.is_compatible_with(shortcut.shape):
    shortcut = ed.layers.Conv2DBatchEnsemble(
        filters3,
        kernel_size=1,
        strides=strides,
        use_bias=False,
        kernel_initializer='he_normal',
        alpha_initializer=make_random_sign_initializer(random_sign_init),
        gamma_initializer=make_random_sign_initializer(random_sign_init),
        name=conv_name_base + '1',
        ensemble_size=ensemble_size)(inputs)
    if use_ensemble_bn:
      shortcut = EnsembleBatchNormalization(
          ensemble_size=ensemble_size,
          name=bn_name_base+'1')(shortcut)
    else:
      shortcut = BatchNormalization(name=bn_name_base+'1')(shortcut)

  x = tf.keras.layers.add([x, shortcut])
  x = tf.keras.layers.Activation('relu')(x)
  return x


def group(inputs, filters, num_blocks, stage, strides,
          ensemble_size, random_sign_init, use_ensemble_bn):
  """Group of residual blocks."""
  bottleneck_block_ = functools.partial(bottleneck_block,
                                        filters=filters,
                                        stage=stage,
                                        ensemble_size=ensemble_size,
                                        random_sign_init=random_sign_init,
                                        use_ensemble_bn=use_ensemble_bn)
  blocks = string.ascii_lowercase
  x = bottleneck_block_(inputs, block=blocks[0], strides=strides)
  for i in range(num_blocks - 1):
    x = bottleneck_block_(x, block=blocks[i + 1], strides=1)
  return x


def resnet50_batchensemble(input_shape,
                           num_classes,
                           ensemble_size,
                           random_sign_init,
                           use_ensemble_bn):
  """Builds BatchEnsemble ResNet50.

  Using strided conv, pooling, four groups of residual blocks, and pooling, the
  network maps spatial features of size 224x224 -> 112x112 -> 56x56 -> 28x28 ->
  14x14 -> 7x7 (Table 1 of He et al. (2015)).

  Args:
    input_shape: Shape tuple of input excluding batch dimension.
    num_classes: Number of output classes.
    ensemble_size: Ensemble size.
    random_sign_init: float, probability of RandomSign initializer.
    use_ensemble_bn: whether to use ensemble batch norm.

  Returns:
    tf.keras.Model.
  """
  group_ = functools.partial(group,
                             ensemble_size=ensemble_size,
                             use_ensemble_bn=use_ensemble_bn,
                             random_sign_init=random_sign_init)
  inputs = tf.keras.layers.Input(shape=input_shape)
  x = tf.keras.layers.ZeroPadding2D(padding=3, name='conv1_pad')(inputs)
  x = ed.layers.Conv2DBatchEnsemble(
      64,
      kernel_size=7,
      strides=2,
      padding='valid',
      use_bias=False,
      kernel_initializer='he_normal',
      alpha_initializer=make_random_sign_initializer(random_sign_init),
      gamma_initializer=make_random_sign_initializer(random_sign_init),
      name='conv1',
      ensemble_size=ensemble_size)(x)
  if use_ensemble_bn:
    x = EnsembleBatchNormalization(
        ensemble_size=ensemble_size,
        name='bn_conv1')(x)
  else:
    x = BatchNormalization(name='bn_conv1')(x)
  x = tf.keras.layers.Activation('relu')(x)
  x = tf.keras.layers.MaxPooling2D(3, strides=(2, 2), padding='same')(x)
  x = group_(x, [64, 64, 256], stage=2, num_blocks=3, strides=1)
  x = group_(x, [128, 128, 512], stage=3, num_blocks=4, strides=2)
  x = group_(x, [256, 256, 1024], stage=4, num_blocks=6, strides=2)
  x = group_(x, [512, 512, 2048], stage=5, num_blocks=3, strides=2)
  x = tf.keras.layers.GlobalAveragePooling2D(name='avg_pool')(x)
  x = ed.layers.DenseBatchEnsemble(
      num_classes,
      ensemble_size=ensemble_size,
      kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01),
      alpha_initializer=make_random_sign_initializer(random_sign_init),
      gamma_initializer=make_random_sign_initializer(random_sign_init),
      activation=None,
      name='fc1000')(x)
  return tf.keras.Model(inputs=inputs, outputs=x, name='resnet50')


def resnet101_batchensemble(input_shape,
                            num_classes,
                            ensemble_size,
                            random_sign_init,
                            use_ensemble_bn):
  """Builds BatchEnsemble ResNet101.

  Using strided conv, pooling, four groups of residual blocks, and pooling, the
  network maps spatial features of size 224x224 -> 112x112 -> 56x56 -> 28x28 ->
  14x14 -> 7x7 (Table 1 of He et al. (2015)).

  Args:
    input_shape: Shape tuple of input excluding batch dimension.
    num_classes: Number of output classes.
    ensemble_size: Ensemble size.
    random_sign_init: float, probability of RandomSign initializer.
    use_ensemble_bn: whether to use ensemble batch norm.

  Returns:
    tf.keras.Model.
  """
  group_ = functools.partial(group,
                             ensemble_size=ensemble_size,
                             use_ensemble_bn=use_ensemble_bn,
                             random_sign_init=random_sign_init)
  inputs = tf.keras.layers.Input(shape=input_shape)
  x = tf.keras.layers.ZeroPadding2D(padding=3, name='conv1_pad')(inputs)
  x = ed.layers.Conv2DBatchEnsemble(
      64,
      kernel_size=7,
      strides=2,
      padding='valid',
      use_bias=False,
      kernel_initializer='he_normal',
      alpha_initializer=make_random_sign_initializer(random_sign_init),
      gamma_initializer=make_random_sign_initializer(random_sign_init),
      name='conv1',
      ensemble_size=ensemble_size)(x)
  if use_ensemble_bn:
    x = EnsembleBatchNormalization(
        ensemble_size=ensemble_size,
        name='bn_conv1')(x)
  else:
    x = BatchNormalization(name='bn_conv1')(x)
  x = tf.keras.layers.Activation('relu')(x)
  x = tf.keras.layers.MaxPooling2D(3, strides=(2, 2), padding='same')(x)
  x = group_(x, [64, 64, 256], stage=2, num_blocks=3, strides=1)
  x = group_(x, [128, 128, 512], stage=3, num_blocks=4, strides=2)
  x = group_(x, [256, 256, 1024], stage=4, num_blocks=23, strides=2)
  x = group_(x, [512, 512, 2048], stage=5, num_blocks=3, strides=2)
  x = tf.keras.layers.GlobalAveragePooling2D(name='avg_pool')(x)
  x = ed.layers.DenseBatchEnsemble(
      num_classes,
      ensemble_size=ensemble_size,
      kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01),
      alpha_initializer=make_random_sign_initializer(random_sign_init),
      gamma_initializer=make_random_sign_initializer(random_sign_init),
      activation=None,
      name='fc1000')(x)
  return tf.keras.Model(inputs=inputs, outputs=x, name='resnet101')


def resnet_batchensemble(input_shape,
                         num_classes,
                         ensemble_size,
                         random_sign_init,
                         use_ensemble_bn,
                         depth=50):
  """Builds BatchEnsemble ResNet50 or ResNet 101.

  Args:
    input_shape: Shape tuple of input excluding batch dimension.
    num_classes: Number of output classes.
    ensemble_size: Ensemble size.
    random_sign_init: float, probability of RandomSign initializer.
    use_ensemble_bn: whether to use ensemble batch norm.
    depth: ResNet depth, default to 50.

  Returns:
    tf.keras.Model.
  """
  if depth == 50:
    return resnet50_batchensemble(
        input_shape,
        num_classes,
        ensemble_size,
        random_sign_init,
        use_ensemble_bn)
  elif depth == 101:
    return resnet101_batchensemble(
        input_shape,
        num_classes,
        ensemble_size,
        random_sign_init,
        use_ensemble_bn)
  else:
    raise ValueError('Only support ResNet with depth 50 or 101 for now.')

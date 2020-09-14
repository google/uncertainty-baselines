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

"""Rank-1 BNN ResNet-50.

A Rank-1 Bayesian neural net (Rank-1 BNN) [1] is an efficient and scalable
approach to variational BNNs that posits prior distributions on rank-1 factors
of the weights and optimizes global mixture variational posterior distributions.

References:

  [1]: Michael W. Dusenberry*, Ghassen Jerfel*, Yeming Wen, Yian Ma, Jasper
       Snoek, Katherine Heller, Balaji Lakshminarayanan, Dustin Tran. Efficient
       and Scalable Bayesian Neural Nets with Rank-1 Factors. In Proc. of
       International Conference on Machine Learning (ICML) 2020.
       https://arxiv.org/abs/2005.07186
"""
import functools
import string
import edward2 as ed
import tensorflow as tf
from uncertainty_baselines.models import rank1_bnn_utils

# Use batch normalization defaults from Pytorch.
BATCH_NORM_DECAY = 0.9
BATCH_NORM_EPSILON = 1e-5

EnsembleSyncBatchNormalization = functools.partial(  # pylint: disable=invalid-name
    ed.layers.EnsembleSyncBatchNorm,
    epsilon=BATCH_NORM_EPSILON,
    momentum=BATCH_NORM_DECAY)
BatchNormalization = functools.partial(  # pylint: disable=invalid-name
    tf.keras.layers.BatchNormalization,
    epsilon=BATCH_NORM_EPSILON,
    momentum=BATCH_NORM_DECAY)


def bottleneck_block(inputs,
                     filters,
                     stage,
                     block,
                     strides,
                     alpha_initializer,
                     gamma_initializer,
                     alpha_regularizer,
                     gamma_regularizer,
                     use_additive_perturbation,
                     ensemble_size,
                     random_sign_init,
                     dropout_rate,
                     prior_stddev,
                     use_tpu,
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
    alpha_initializer: The initializer for the alpha parameters.
    gamma_initializer: The initializer for the gamma parameters.
    alpha_regularizer: The regularizer for the alpha parameters.
    gamma_regularizer: The regularizer for the gamma parameters.
    use_additive_perturbation: Whether or not to use additive perturbations
      instead of multiplicative perturbations.
    ensemble_size: Number of ensemble members.
    random_sign_init: Value used to initialize trainable deterministic
      initializers, as applicable. Values greater than zero result in
      initialization to a random sign vector, where random_sign_init is the
      probability of a 1 value. Values less than zero result in initialization
      from a Gaussian with mean 1 and standard deviation equal to
      -random_sign_init.
    dropout_rate: Dropout rate.
    prior_stddev: Standard deviation of the prior.
    use_tpu: whether the model runs on TPU.
    use_ensemble_bn: Whether to use ensemble sync BN.

  Returns:
    tf.Tensor.
  """
  filters1, filters2, filters3 = filters
  conv_name_base = 'res' + str(stage) + block + '_branch'
  bn_name_base = 'bn' + str(stage) + block + '_branch'

  x = ed.layers.Conv2DRank1(
      filters1,
      kernel_size=1,
      use_bias=False,
      alpha_initializer=rank1_bnn_utils.make_initializer(
          alpha_initializer, random_sign_init, dropout_rate),
      gamma_initializer=rank1_bnn_utils.make_initializer(
          gamma_initializer, random_sign_init, dropout_rate),
      kernel_initializer='he_normal',
      alpha_regularizer=rank1_bnn_utils.make_regularizer(
          alpha_regularizer, 1., prior_stddev),
      gamma_regularizer=rank1_bnn_utils.make_regularizer(
          gamma_regularizer, 1., prior_stddev),
      use_additive_perturbation=use_additive_perturbation,
      name=conv_name_base + '2a',
      ensemble_size=ensemble_size)(inputs)

  if use_ensemble_bn:
    x = EnsembleSyncBatchNormalization(
        ensemble_size=ensemble_size, name=bn_name_base + '2a')(x)
  else:
    x = ed.layers.ensemble_batchnorm(
        x,
        ensemble_size=ensemble_size,
        use_tpu=use_tpu,
        momentum=BATCH_NORM_DECAY,
        epsilon=BATCH_NORM_EPSILON,
        name=bn_name_base+'2a')

  x = tf.keras.layers.Activation('relu')(x)

  x = ed.layers.Conv2DRank1(
      filters2,
      kernel_size=3,
      strides=strides,
      padding='same',
      use_bias=False,
      alpha_initializer=rank1_bnn_utils.make_initializer(
          alpha_initializer, random_sign_init, dropout_rate),
      gamma_initializer=rank1_bnn_utils.make_initializer(
          gamma_initializer, random_sign_init, dropout_rate),
      kernel_initializer='he_normal',
      alpha_regularizer=rank1_bnn_utils.make_regularizer(
          alpha_regularizer, 1., prior_stddev),
      gamma_regularizer=rank1_bnn_utils.make_regularizer(
          gamma_regularizer, 1., prior_stddev),
      use_additive_perturbation=use_additive_perturbation,
      name=conv_name_base + '2b',
      ensemble_size=ensemble_size)(x)

  if use_ensemble_bn:
    x = EnsembleSyncBatchNormalization(
        ensemble_size=ensemble_size, name=bn_name_base + '2b')(x)
  else:
    x = ed.layers.ensemble_batchnorm(
        x,
        ensemble_size=ensemble_size,
        use_tpu=use_tpu,
        momentum=BATCH_NORM_DECAY,
        epsilon=BATCH_NORM_EPSILON,
        name=bn_name_base+'2b')

  x = tf.keras.layers.Activation('relu')(x)

  x = ed.layers.Conv2DRank1(
      filters3,
      kernel_size=1,
      use_bias=False,
      alpha_initializer=rank1_bnn_utils.make_initializer(
          alpha_initializer, random_sign_init, dropout_rate),
      gamma_initializer=rank1_bnn_utils.make_initializer(
          gamma_initializer, random_sign_init, dropout_rate),
      kernel_initializer='he_normal',
      alpha_regularizer=rank1_bnn_utils.make_regularizer(
          alpha_regularizer, 1., prior_stddev),
      gamma_regularizer=rank1_bnn_utils.make_regularizer(
          gamma_regularizer, 1., prior_stddev),
      use_additive_perturbation=use_additive_perturbation,
      name=conv_name_base + '2c',
      ensemble_size=ensemble_size)(x)

  if use_ensemble_bn:
    x = EnsembleSyncBatchNormalization(
        ensemble_size=ensemble_size, name=bn_name_base + '2c')(x)
  else:
    x = ed.layers.ensemble_batchnorm(
        x,
        ensemble_size=ensemble_size,
        use_tpu=use_tpu,
        momentum=BATCH_NORM_DECAY,
        epsilon=BATCH_NORM_EPSILON,
        name=bn_name_base + '2c')

  shortcut = inputs
  if not x.shape.is_compatible_with(shortcut.shape):
    shortcut = ed.layers.Conv2DRank1(
        filters3,
        kernel_size=1,
        strides=strides,
        use_bias=False,
        alpha_initializer=rank1_bnn_utils.make_initializer(
            alpha_initializer, random_sign_init, dropout_rate),
        gamma_initializer=rank1_bnn_utils.make_initializer(
            gamma_initializer, random_sign_init, dropout_rate),
        kernel_initializer='he_normal',
        alpha_regularizer=rank1_bnn_utils.make_regularizer(
            alpha_regularizer, 1., prior_stddev),
        gamma_regularizer=rank1_bnn_utils.make_regularizer(
            gamma_regularizer, 1., prior_stddev),
        use_additive_perturbation=use_additive_perturbation,
        name=conv_name_base + '1',
        ensemble_size=ensemble_size)(inputs)
    if use_ensemble_bn:
      shortcut = EnsembleSyncBatchNormalization(
          ensemble_size=ensemble_size, name=bn_name_base + '1')(shortcut)
    else:
      shortcut = ed.layers.ensemble_batchnorm(
          shortcut,
          ensemble_size=ensemble_size,
          use_tpu=use_tpu,
          momentum=BATCH_NORM_DECAY,
          epsilon=BATCH_NORM_EPSILON,
          name=bn_name_base + '1')

  x = tf.keras.layers.add([x, shortcut])
  x = tf.keras.layers.Activation('relu')(x)
  return x


def group(inputs,
          filters,
          num_blocks,
          stage,
          strides,
          alpha_initializer,
          gamma_initializer,
          alpha_regularizer,
          gamma_regularizer,
          use_additive_perturbation,
          ensemble_size,
          random_sign_init,
          dropout_rate,
          prior_stddev,
          use_tpu,
          use_ensemble_bn):
  """Group of residual blocks."""
  bottleneck_block_ = functools.partial(
      bottleneck_block,
      filters=filters,
      stage=stage,
      alpha_initializer=alpha_initializer,
      gamma_initializer=gamma_initializer,
      alpha_regularizer=alpha_regularizer,
      gamma_regularizer=gamma_regularizer,
      use_additive_perturbation=use_additive_perturbation,
      ensemble_size=ensemble_size,
      random_sign_init=random_sign_init,
      dropout_rate=dropout_rate,
      prior_stddev=prior_stddev,
      use_tpu=use_tpu,
      use_ensemble_bn=use_ensemble_bn)
  blocks = string.ascii_lowercase
  x = bottleneck_block_(inputs, block=blocks[0], strides=strides)
  for i in range(num_blocks - 1):
    x = bottleneck_block_(x, block=blocks[i + 1], strides=1)
  return x


def resnet50_rank1(input_shape,
                   num_classes,
                   alpha_initializer,
                   gamma_initializer,
                   alpha_regularizer,
                   gamma_regularizer,
                   use_additive_perturbation,
                   ensemble_size,
                   random_sign_init,
                   dropout_rate,
                   prior_stddev,
                   use_tpu,
                   use_ensemble_bn):
  """Builds ResNet50 with rank 1 priors.

  Using strided conv, pooling, four groups of residual blocks, and pooling, the
  network maps spatial features of size 224x224 -> 112x112 -> 56x56 -> 28x28 ->
  14x14 -> 7x7 (Table 1 of He et al. (2015)).

  Args:
    input_shape: Shape tuple of input excluding batch dimension.
    num_classes: Number of output classes.
    alpha_initializer: The initializer for the alpha parameters.
    gamma_initializer: The initializer for the gamma parameters.
    alpha_regularizer: The regularizer for the alpha parameters.
    gamma_regularizer: The regularizer for the gamma parameters.
    use_additive_perturbation: Whether or not to use additive perturbations
      instead of multiplicative perturbations.
    ensemble_size: Number of ensemble members.
    random_sign_init: Value used to initialize trainable deterministic
      initializers, as applicable. Values greater than zero result in
      initialization to a random sign vector, where random_sign_init is the
      probability of a 1 value. Values less than zero result in initialization
      from a Gaussian with mean 1 and standard deviation equal to
      -random_sign_init.
    dropout_rate: Dropout rate.
    prior_stddev: Standard deviation of the prior.
    use_tpu: whether the model runs on TPU.
    use_ensemble_bn: Whether to use ensemble batch norm.

  Returns:
    tf.keras.Model.
  """
  group_ = functools.partial(
      group,
      alpha_initializer=alpha_initializer,
      gamma_initializer=gamma_initializer,
      alpha_regularizer=alpha_regularizer,
      gamma_regularizer=gamma_regularizer,
      use_additive_perturbation=use_additive_perturbation,
      ensemble_size=ensemble_size,
      random_sign_init=random_sign_init,
      dropout_rate=dropout_rate,
      prior_stddev=prior_stddev,
      use_tpu=use_tpu,
      use_ensemble_bn=use_ensemble_bn)
  inputs = tf.keras.layers.Input(shape=input_shape)
  x = tf.keras.layers.ZeroPadding2D(padding=3, name='conv1_pad')(inputs)
  x = ed.layers.Conv2DRank1(
      64,
      kernel_size=7,
      strides=2,
      padding='valid',
      use_bias=False,
      alpha_initializer=rank1_bnn_utils.make_initializer(
          alpha_initializer, random_sign_init, dropout_rate),
      gamma_initializer=rank1_bnn_utils.make_initializer(
          gamma_initializer, random_sign_init, dropout_rate),
      kernel_initializer='he_normal',
      alpha_regularizer=rank1_bnn_utils.make_regularizer(
          alpha_regularizer, 1., prior_stddev),
      gamma_regularizer=rank1_bnn_utils.make_regularizer(
          gamma_regularizer, 1., prior_stddev),
      use_additive_perturbation=use_additive_perturbation,
      name='conv1',
      ensemble_size=ensemble_size)(x)
  if use_ensemble_bn:
    x = EnsembleSyncBatchNormalization(
        ensemble_size=ensemble_size, name='bn_conv1')(x)
  else:
    x = ed.layers.ensemble_batchnorm(
        x,
        ensemble_size=ensemble_size,
        use_tpu=use_tpu,
        momentum=BATCH_NORM_DECAY,
        epsilon=BATCH_NORM_EPSILON,
        name='bn_conv1')

  x = tf.keras.layers.Activation('relu')(x)
  x = tf.keras.layers.MaxPooling2D(3, strides=(2, 2), padding='same')(x)
  x = group_(x, [64, 64, 256], stage=2, num_blocks=3, strides=1)
  x = group_(x, [128, 128, 512], stage=3, num_blocks=4, strides=2)
  x = group_(x, [256, 256, 1024], stage=4, num_blocks=6, strides=2)
  x = group_(x, [512, 512, 2048], stage=5, num_blocks=3, strides=2)
  x = tf.keras.layers.GlobalAveragePooling2D(name='avg_pool')(x)
  x = ed.layers.DenseRank1(
      num_classes,
      alpha_initializer=rank1_bnn_utils.make_initializer(
          alpha_initializer, random_sign_init, dropout_rate),
      gamma_initializer=rank1_bnn_utils.make_initializer(
          gamma_initializer, random_sign_init, dropout_rate),
      kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01),
      alpha_regularizer=rank1_bnn_utils.make_regularizer(
          alpha_regularizer, 1., prior_stddev),
      gamma_regularizer=rank1_bnn_utils.make_regularizer(
          gamma_regularizer, 1., prior_stddev),
      use_additive_perturbation=use_additive_perturbation,
      ensemble_size=ensemble_size,
      activation=None,
      name='fc1000')(x)
  return tf.keras.Model(inputs=inputs, outputs=x, name='resnet50')

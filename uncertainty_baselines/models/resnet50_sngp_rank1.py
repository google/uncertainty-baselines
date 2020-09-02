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

"""BatchEnsemble ResNet50."""

import functools
import string
import warnings
import tensorflow as tf
from uncertainty_baselines.models import rank1_bnn_utils

try:
  import edward2 as ed  # pylint: disable=g-import-not-at-top
except ImportError as e:
  warnings.warn(f'Skipped due to ImportError: {e}')

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


def make_random_sign_initializer(random_sign_init):
  if random_sign_init > 0:
    initializer = ed.initializers.RandomSign(random_sign_init)
  else:
    initializer = tf.keras.initializers.RandomNormal(
        mean=1.0, stddev=-random_sign_init)
  return initializer


def make_conv2d_rank1_layer(use_spec_norm,
                            spec_norm_iteration,
                            spec_norm_bound):
  """Defines type of Conv2D layer to use based on spectral normalization."""
  Conv2DRank1Base = functools.partial(  # pylint: disable=invalid-name
      ed.layers.Conv2DRank1, padding='same')

  def Conv2DRank1Normed(*conv_args, **conv_kwargs):  # pylint: disable=invalid-name
    return ed.experimental.sngp.SpectralNormalizationConv2D(
        Conv2DRank1Base(*conv_args, **conv_kwargs),
        iteration=spec_norm_iteration,
        norm_multiplier=spec_norm_bound)

  return Conv2DRank1Normed if use_spec_norm else Conv2DRank1Base


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
                     use_ensemble_bn,
                     conv_layer):
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
    use_ensemble_bn: whether to use ensemble batch norm.
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
    x = BatchNormalization(name=bn_name_base + '2a')(x)

  x = tf.keras.layers.Activation('relu')(x)

  x = conv_layer(
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
    x = BatchNormalization(name=bn_name_base + '2b')(x)

  x = tf.keras.layers.Activation('relu')(x)

  x = conv_layer(
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
    x = BatchNormalization(name=bn_name_base + '2c')(x)

  shortcut = inputs
  if not x.shape.is_compatible_with(shortcut.shape):
    shortcut = conv_layer(
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
      shortcut = BatchNormalization(name=bn_name_base + '1')(shortcut)

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
          use_ensemble_bn,
          conv_layer):
  """Group of residual blocks."""
  blocks = string.ascii_lowercase
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
      use_ensemble_bn=use_ensemble_bn,
      conv_layer=conv_layer)

  x = bottleneck_block_(inputs, block=blocks[0], strides=strides)
  for i in range(num_blocks - 1):
    x = bottleneck_block_(x, block=blocks[i + 1], strides=1)
  return x


def resnet50_sngp_rank1(input_shape,
                        batch_size,
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
                        use_ensemble_bn,
                        use_gp_layer,
                        gp_hidden_dim,
                        gp_scale,
                        gp_bias,
                        gp_input_normalization,
                        gp_cov_discount_factor,
                        gp_cov_ridge_penalty,
                        gp_output_imagenet_initializer,
                        use_spec_norm,
                        spec_norm_iteration,
                        spec_norm_bound,
                        input_spec_norm,
                        initializer):
  """Builds SNGP+BatchEnsemble ResNet50.

  Using strided conv, pooling, four groups of residual blocks, and pooling, the
  network maps spatial features of size 224x224 -> 112x112 -> 56x56 -> 28x28 ->
  14x14 -> 7x7 (Table 1 of He et al. (2015)).

  Args:
    input_shape: Shape tuple of input excluding batch dimension.
    batch_size: The batch size of the input layer. Required by the spectral
      normalization.
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
    use_ensemble_bn: Whether to use ensemble batch norm.
    use_gp_layer: Whether to use Gaussian process layer as the output layer.
    gp_hidden_dim: The hidden dimension of the GP layer, which corresponds to
      the number of random features used for the approximation.
    gp_scale: The length-scale parameter for the RBF kernel of the GP layer.
    gp_bias: The bias term for GP layer.
    gp_input_normalization: Whether to normalize the input using LayerNorm for
      GP layer. This is similar to automatic relevance determination (ARD) in
      the classic GP learning.
    gp_cov_discount_factor: The discount factor to compute the moving average of
      precision matrix.
    gp_cov_ridge_penalty: Ridge penalty parameter for GP posterior covariance.
    gp_output_imagenet_initializer: Whether to initialize GP output layer using
      Gaussian with small standard deviation (sd=0.01).
    use_spec_norm: Whether to apply spectral normalization.
    spec_norm_iteration: Number of power iterations to perform for estimating
      the spectral norm of weight matrices.
    spec_norm_bound: Upper bound to spectral norm of weight matrices.
    input_spec_norm: Whether to apply spectral normalization to the input layer.

  Returns:
    tf.keras.Model.
  """
  Conv2DRank1 = make_conv2d_rank1_layer(  # pylint: disable=invalid-name
      use_spec_norm, spec_norm_iteration, spec_norm_bound)

  group_ = functools.partial(
      group,
      use_ensemble_bn=use_ensemble_bn,
      alpha_initializer=alpha_initializer,
      gamma_initializer=gamma_initializer,
      alpha_regularizer=alpha_regularizer,
      gamma_regularizer=gamma_regularizer,
      use_additive_perturbation=use_additive_perturbation,
      ensemble_size=ensemble_size,
      random_sign_init=random_sign_init,
      dropout_rate=dropout_rate,
      prior_stddev=prior_stddev,
      conv_layer=Conv2DRank1)

  inputs = tf.keras.layers.Input(shape=input_shape, batch_size=batch_size)
  x = tf.keras.layers.ZeroPadding2D(padding=3, name='conv1_pad')(inputs)

  InputConv2DRank1 = make_conv2d_rank1_layer(  # pylint: disable=invalid-name
      (input_spec_norm and use_spec_norm), spec_norm_iteration, spec_norm_bound)
  print('Finna build InputConv2DRank1')
  x = InputConv2DRank1(
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
      name='conv1',
      ensemble_size=ensemble_size)(x)

  if use_ensemble_bn:
    x = EnsembleSyncBatchNormalization(
        ensemble_size=ensemble_size, name='bn_conv1')(x)
  else:
    x = BatchNormalization(name='bn_conv1')(x)

  x = tf.keras.layers.Activation('relu')(x)
  x = tf.keras.layers.MaxPooling2D(3, strides=(2, 2), padding='same')(x)
  print('Finna build group 1')
  x = group_(x, [64, 64, 256], stage=2, num_blocks=3, strides=1)
  print('Finna build group 2')
  x = group_(x, [128, 128, 512], stage=3, num_blocks=4, strides=2)
  print('Finna build group 3')
  x = group_(x, [256, 256, 1024], stage=4, num_blocks=6, strides=2)
  print('Finna build group 4')
  x = group_(x, [512, 512, 2048], stage=5, num_blocks=3, strides=2)
  x = tf.keras.layers.GlobalAveragePooling2D(name='avg_pool')(x)
  if use_gp_layer:
    gp_output_initializer = None
    if gp_output_imagenet_initializer:
      # Use the same initializer as dense
      if initializer == 'he_normal':
        gp_output_initializer = 'he_normal'
      elif initializer == 'keras_normal':
        gp_output_initializer = tf.keras.initializers.RandomNormal(stddev=0.01)
      elif initializer == 'random_normal':
        gp_output_initializer = tf.random_normal_initializer(mean=0.0, stddev=0.01)
    gp_layer = functools.partial(
        ed.layers.RandomFeatureGaussianProcess,
        num_inducing=gp_hidden_dim,
        gp_kernel_scale=gp_scale,
        gp_output_bias=gp_bias,
        normalize_input=gp_input_normalization,
        gp_cov_momentum=gp_cov_discount_factor,
        gp_cov_ridge_penalty=gp_cov_ridge_penalty,
        scale_random_features=False,
        use_custom_random_features=True,
        kernel_initializer=gp_output_initializer)
    outputs = gp_layer(num_classes)(x)
  else:
    outputs = ed.layers.DenseRank1(
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
  return tf.keras.Model(
      inputs=inputs, outputs=outputs, name='resnet50')

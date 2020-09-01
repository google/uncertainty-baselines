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

"""Wide ResNet with SNGP and Rank1 BNNsß."""
import functools
import edward2 as ed
import tensorflow as tf

from uncertainty_baselines.models import rank1_bnn_utils

BatchNormalization = functools.partial(  # pylint: disable=invalid-name
    tf.keras.layers.BatchNormalization,
    epsilon=1e-5,  # using epsilon and momentum defaults from Torch
    momentum=0.9)


def make_conv2d_rank1_layer(use_spec_norm,
                            spec_norm_iteration,
                            spec_norm_bound):
  """Defines type of Conv2D layer to use based on spectral normalization."""
  Conv2DRank1 = functools.partial(  # pylint: disable=invalid-name
      ed.layers.Conv2DRank1,
      kernel_size=3,
      padding='same',
      use_bias=False,
      kernel_initializer='he_normal')

  def Conv2DRank1Normed(*conv_args, **conv_kwargs):  # pylint: disable=invalid-name
    return ed.experimental.sngp.normalization.SpectralNormalizationConv2D(
        Conv2DRank1(*conv_args, **conv_kwargs),
        iteration=spec_norm_iteration,
        norm_multiplier=spec_norm_bound)

  return Conv2DRank1Normed if use_spec_norm else Conv2DRank1


def basic_block(inputs,
                filters,
                strides,
                alpha_initializer,
                gamma_initializer,
                alpha_regularizer,
                gamma_regularizer,
                use_additive_perturbation,
                ensemble_size,
                random_sign_init,
                dropout_rate,
                prior_mean,
                prior_stddev,
                use_spec_norm,
                spec_norm_iteration,
                spec_norm_bound):
  """Basic residual block of two 3x3 convs.

  Args:
    inputs: tf.Tensor.
    filters: Number of filters for Conv2D.
    strides: Stride dimensions for Conv2D.
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
    prior_mean: Mean of the prior.
    prior_stddev: Standard deviation of the prior.
    use_spec_norm: Whether to apply spectral normalization.
    spec_norm_iteration: Number of power iterations to perform for estimating
      the spectral norm of weight matrices.
    spec_norm_bound: Upper bound to spectral norm of weight matrices.

  Returns:
    tf.Tensor.
  """
  Conv2DRank1 = make_conv2d_rank1_layer(  # pylint: disable=invalid-name
      use_spec_norm,
      spec_norm_iteration,
      spec_norm_bound)
  x = inputs
  y = inputs
  y = BatchNormalization()(y)
  y = tf.keras.layers.Activation('relu')(y)
  y = Conv2DRank1(
      filters,
      strides=strides,
      alpha_initializer=rank1_bnn_utils.make_initializer(
          alpha_initializer, random_sign_init, dropout_rate),
      gamma_initializer=rank1_bnn_utils.make_initializer(
          gamma_initializer, random_sign_init, dropout_rate),
      alpha_regularizer=rank1_bnn_utils.make_regularizer(
          alpha_regularizer, prior_mean, prior_stddev),
      gamma_regularizer=rank1_bnn_utils.make_regularizer(
          gamma_regularizer, prior_mean, prior_stddev),
      use_additive_perturbation=use_additive_perturbation,
      ensemble_size=ensemble_size)(y)
  y = BatchNormalization()(y)
  y = tf.keras.layers.Activation('relu')(y)
  y = Conv2DRank1(
      filters,
      strides=1,
      alpha_initializer=rank1_bnn_utils.make_initializer(
          alpha_initializer, random_sign_init, dropout_rate),
      gamma_initializer=rank1_bnn_utils.make_initializer(
          gamma_initializer, random_sign_init, dropout_rate),
      alpha_regularizer=rank1_bnn_utils.make_regularizer(
          alpha_regularizer, prior_mean, prior_stddev),
      gamma_regularizer=rank1_bnn_utils.make_regularizer(
          gamma_regularizer, prior_mean, prior_stddev),
      use_additive_perturbation=use_additive_perturbation,
      ensemble_size=ensemble_size)(y)
  if not x.shape.is_compatible_with(y.shape):
    x = Conv2DRank1(
        filters,
        kernel_size=1,
        strides=strides,
        alpha_initializer=rank1_bnn_utils.make_initializer(
            alpha_initializer, random_sign_init, dropout_rate),
        gamma_initializer=rank1_bnn_utils.make_initializer(
            gamma_initializer, random_sign_init, dropout_rate),
        alpha_regularizer=rank1_bnn_utils.make_regularizer(
            alpha_regularizer, prior_mean, prior_stddev),
        gamma_regularizer=rank1_bnn_utils.make_regularizer(
            gamma_regularizer, prior_mean, prior_stddev),
        use_additive_perturbation=use_additive_perturbation,
        ensemble_size=ensemble_size)(x)
  x = tf.keras.layers.add([x, y])
  return x


def group(inputs, filters, strides, num_blocks, **kwargs):
  """Group of residual blocks."""
  x = basic_block(inputs, filters=filters, strides=strides, **kwargs)
  for _ in range(num_blocks - 1):
    x = basic_block(x, filters=filters, strides=1, **kwargs)
  return x


def wide_resnet_sngp_rank1(input_shape,
                           batch_size,
                           depth,
                           width_multiplier,
                           num_classes,
                           alpha_initializer,
                           gamma_initializer,
                           alpha_regularizer,
                           gamma_regularizer,
                           use_additive_perturbation,
                           ensemble_size,
                           random_sign_init,
                           dropout_rate,
                           prior_mean,
                           prior_stddev,
                           use_gp_layer,
                           gp_input_dim,
                           gp_hidden_dim,
                           gp_scale,
                           gp_bias,
                           gp_input_normalization,
                           gp_cov_discount_factor,
                           gp_cov_ridge_penalty,
                           use_spec_norm,
                           spec_norm_iteration,
                           spec_norm_bound):
  """Builds Wide ResNet.

  Following Zagoruyko and Komodakis (2016), it accepts a width multiplier on the
  number of filters. Using three groups of residual blocks, the network maps
  spatial features of size 32x32 -> 16x16 -> 8x8.

  Args:
    input_shape: tf.Tensor.
    batch_size: The batch size of the input layer. Required by the spectral
      normalization.
    depth: Total number of convolutional layers. "n" in WRN-n-k. It differs from
      He et al. (2015)'s notation which uses the maximum depth of the network
      counting non-conv layers like dense.
    width_multiplier: Integer to multiply the number of typical filters by. "k"
      in WRN-n-k.
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
    prior_mean: Mean of the prior.
    prior_stddev: Standard deviation of the prior.
    use_gp_layer: Whether to use Gaussian process layer as the output layer.
    gp_input_dim: The input dimension to GP layer.
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
    use_spec_norm: Whether to apply spectral normalization.
    spec_norm_iteration: Number of power iterations to perform for estimating
      the spectral norm of weight matrices.
    spec_norm_bound: Upper bound to spectral norm of weight matrices.

  Returns:
    tf.keras.Model.
  """
  Conv2DRank1 = make_conv2d_rank1_layer(  # pylint: disable=invalid-name
      use_spec_norm,
      spec_norm_iteration,
      spec_norm_bound)
  GaussianProcess = functools.partial(  # pylint: disable=invalid-name
      ed.experimental.sngp.RandomFeatureGaussianProcess,
      num_inducing=gp_hidden_dim,
      gp_kernel_scale=gp_scale,
      gp_output_bias=gp_bias,
      normalize_input=gp_input_normalization,
      gp_cov_momentum=gp_cov_discount_factor,
      gp_cov_ridge_penalty=gp_cov_ridge_penalty)

  if (depth - 4) % 6 != 0:
    raise ValueError('depth should be 6n+4 (e.g., 16, 22, 28, 40).')
  num_blocks = (depth - 4) // 6
  inputs = tf.keras.layers.Input(shape=input_shape, batch_size=batch_size)

  x = Conv2DRank1(
      16,
      strides=1,
      alpha_initializer=rank1_bnn_utils.make_initializer(
          alpha_initializer, random_sign_init, dropout_rate),
      gamma_initializer=rank1_bnn_utils.make_initializer(
          gamma_initializer, random_sign_init, dropout_rate),
      alpha_regularizer=rank1_bnn_utils.make_regularizer(
          alpha_regularizer, prior_mean, prior_stddev),
      gamma_regularizer=rank1_bnn_utils.make_regularizer(
          gamma_regularizer, prior_mean, prior_stddev),
      use_additive_perturbation=use_additive_perturbation,
      ensemble_size=ensemble_size)(inputs)
  for strides, filters in zip([1, 2, 2], [16, 32, 64]):
    x = group(x,
              filters=filters * width_multiplier,
              strides=strides,
              num_blocks=num_blocks,
              alpha_initializer=alpha_initializer,
              gamma_initializer=gamma_initializer,
              alpha_regularizer=alpha_regularizer,
              gamma_regularizer=gamma_regularizer,
              use_additive_perturbation=use_additive_perturbation,
              ensemble_size=ensemble_size,
              random_sign_init=random_sign_init,
              dropout_rate=dropout_rate,
              prior_mean=prior_mean,
              prior_stddev=prior_stddev,
              use_spec_norm=use_spec_norm,
              spec_norm_iteration=spec_norm_iteration,
              spec_norm_bound=spec_norm_bound)

  x = BatchNormalization()(x)
  x = tf.keras.layers.Activation('relu')(x)
  x = tf.keras.layers.AveragePooling2D(pool_size=8)(x)
  x = tf.keras.layers.Flatten()(x)

  if use_gp_layer:
    # Uses random projection to reduce the input dimension of the GP layer.
    if gp_input_dim > 0:
      # TODO(ghassen): should the final layer be deterministic ? Add flag.
      x = tf.keras.layers.Dense(
          gp_input_dim,
          kernel_initializer='random_normal',
          use_bias=False,
          trainable=False)(x)
    logits, covmat = GaussianProcess(num_classes)(x)
  else:
    logits = ed.layers.DenseRank1(
        num_classes,
        alpha_initializer=rank1_bnn_utils.make_initializer(
            alpha_initializer, random_sign_init, dropout_rate),
        gamma_initializer=rank1_bnn_utils.make_initializer(
            gamma_initializer, random_sign_init, dropout_rate),
        kernel_initializer='he_normal',
        activation=None,
        alpha_regularizer=rank1_bnn_utils.make_regularizer(
            alpha_regularizer, prior_mean, prior_stddev),
        gamma_regularizer=rank1_bnn_utils.make_regularizer(
            gamma_regularizer, prior_mean, prior_stddev),
        use_additive_perturbation=use_additive_perturbation,
        ensemble_size=ensemble_size)(x)
    covmat = tf.eye(batch_size)

  return tf.keras.Model(inputs=inputs, outputs=[logits, covmat])

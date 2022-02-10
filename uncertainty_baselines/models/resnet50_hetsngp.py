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

"""ResNet50 model with HetSNGP."""
import functools
import string

import edward2 as ed
import tensorflow as tf


# Use batch normalization defaults from Pytorch.
BATCH_NORM_DECAY = 0.9
BATCH_NORM_EPSILON = 1e-5


def MonteCarloDropout(  # pylint:disable=invalid-name
    inputs,
    dropout_rate,
    use_mc_dropout,
    filterwise_dropout):
  """Defines the Monte Carlo dropout layer."""
  training = None
  noise_shape = None

  if use_mc_dropout:
    training = True

  if filterwise_dropout:
    noise_shape = [inputs.shape[0], 1, 1, inputs.shape[3]]

  return tf.keras.layers.Dropout(
      dropout_rate, noise_shape=noise_shape)(
          inputs, training=training)


def make_random_feature_initializer(random_feature_type):
  # Use stddev=0.05 to replicate the default behavior of
  # tf.keras.initializer.RandomNormal.
  if random_feature_type == 'orf':
    return ed.initializers.OrthogonalRandomFeatures(stddev=0.05)
  elif random_feature_type == 'rff':
    return tf.keras.initializers.RandomNormal(stddev=0.05)
  else:
    return random_feature_type


def make_conv2d_layer(use_spec_norm,
                      spec_norm_iteration,
                      spec_norm_bound):
  """Defines type of Conv2D layer to use based on spectral normalization."""
  Conv2DBase = functools.partial(tf.keras.layers.Conv2D, padding='same')  # pylint: disable=invalid-name
  def Conv2DNormed(*conv_args, **conv_kwargs):  # pylint: disable=invalid-name
    return ed.layers.SpectralNormalizationConv2D(
        Conv2DBase(*conv_args, **conv_kwargs),
        iteration=spec_norm_iteration,
        norm_multiplier=spec_norm_bound)

  return Conv2DNormed if use_spec_norm else Conv2DBase


def bottleneck_block(inputs, filters, stage, block, strides, conv_layer,
                     dropout_layer):
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
    dropout_layer: Callable for dropout layer.

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
  x = dropout_layer(x)

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
  x = dropout_layer(x)

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
    shortcut = dropout_layer(shortcut)

  x = tf.keras.layers.add([x, shortcut])
  x = tf.keras.layers.Activation('relu')(x)
  return x


def group(inputs, filters, num_blocks, stage, strides, conv_layer,
          dropout_layer):
  """Group of residual blocks."""
  blocks = string.ascii_lowercase
  x = bottleneck_block(
      inputs,
      filters,
      stage,
      block=blocks[0],
      strides=strides,
      conv_layer=conv_layer,
      dropout_layer=dropout_layer)
  for i in range(num_blocks - 1):
    x = bottleneck_block(
        x,
        filters,
        stage,
        block=blocks[i + 1],
        strides=1,
        conv_layer=conv_layer,
        dropout_layer=dropout_layer)
  return x


def resnet50_hetsngp_add_last_layer(
    inputs, x, num_classes, num_factors, use_gp_layer, gp_hidden_dim, gp_scale,
    gp_bias, gp_input_normalization, gp_random_feature_type,
    gp_cov_discount_factor, gp_cov_ridge_penalty,
    gp_output_imagenet_initializer, temperature, num_mc_samples, eps,
    sngp_var_weight, het_var_weight):
  """Builds ResNet50.

  Using strided conv, pooling, four groups of residual blocks, and pooling, the
  network maps spatial features of size 224x224 -> 112x112 -> 56x56 -> 28x28 ->
  14x14 -> 7x7 (Table 1 of He et al. (2015)).

  Args:
    inputs: inputs
    x: x
    num_classes: Number of output classes.
    num_factors: Number of factors for the heteroscedastic variance.
    use_gp_layer: Whether to use Gaussian process layer as the output layer.
    gp_hidden_dim: The hidden dimension of the GP layer, which corresponds to
      the number of random features used for the approximation.
    gp_scale: The length-scale parameter for the RBF kernel of the GP layer.
    gp_bias: The bias term for GP layer.
    gp_input_normalization: Whether to normalize the input using LayerNorm for
      GP layer. This is similar to automatic relevance determination (ARD) in
      the classic GP learning.
    gp_random_feature_type: The type of random feature to use for
      `RandomFeatureGaussianProcess`.
    gp_cov_discount_factor: The discount factor to compute the moving average of
      precision matrix.
    gp_cov_ridge_penalty: Ridge penalty parameter for GP posterior covariance.
    gp_output_imagenet_initializer: Whether to initialize GP output layer using
      Gaussian with small standard deviation (sd=0.01).
    temperature: Float or scalar `Tensor` representing the softmax
      temperature.
    num_mc_samples: The number of Monte-Carlo samples used to estimate the
        predictive distribution.
    eps: Float. Clip probabilities into [eps, 1.0] softmax or
        [eps, 1.0 - eps] sigmoid before applying log (softmax), or inverse
        sigmoid.
    sngp_var_weight: Weight in [0,1] for the SNGP variance in the output.
    het_var_weight: Weight in [0,1] for the het. variance in the output.

  Returns:
    tf.keras.Model.
  """
  x = tf.keras.layers.GlobalAveragePooling2D(name='avg_pool')(x)

  if use_gp_layer:
    gp_output_initializer = None
    if gp_output_imagenet_initializer:
      # Use the same initializer as dense
      gp_output_initializer = tf.keras.initializers.RandomNormal(stddev=0.01)
    output_layer = functools.partial(
        ed.layers.HeteroscedasticSNGPLayer,
        num_factors=num_factors,
        num_inducing=gp_hidden_dim,
        gp_kernel_scale=gp_scale,
        gp_output_bias=gp_bias,
        normalize_input=gp_input_normalization,
        gp_cov_momentum=gp_cov_discount_factor,
        gp_cov_ridge_penalty=gp_cov_ridge_penalty,
        scale_random_features=False,
        use_custom_random_features=True,
        custom_random_features_initializer=make_random_feature_initializer(
            gp_random_feature_type),
        kernel_initializer=gp_output_initializer,
        temperature=temperature,
        train_mc_samples=num_mc_samples,
        test_mc_samples=num_mc_samples,
        share_samples_across_batch=True,
        logits_only=True,
        eps=eps,
        dtype=tf.float32,
        sngp_var_weight=sngp_var_weight,
        het_var_weight=het_var_weight)
  else:
    output_layer = functools.partial(
        tf.keras.layers.Dense,
        activation=None,
        kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01),
        name='fc1000')

  outputs = output_layer(num_classes)(x)
  return tf.keras.Model(inputs=inputs, outputs=outputs, name='resnet50')


def resnet50_hetsngp(input_shape,
                     batch_size,
                     num_classes,
                     num_factors,
                     use_mc_dropout,
                     dropout_rate,
                     filterwise_dropout,
                     use_gp_layer,
                     gp_hidden_dim,
                     gp_scale,
                     gp_bias,
                     gp_input_normalization,
                     gp_random_feature_type,
                     gp_cov_discount_factor,
                     gp_cov_ridge_penalty,
                     gp_output_imagenet_initializer,
                     use_spec_norm,
                     spec_norm_iteration,
                     spec_norm_bound,
                     temperature,
                     num_mc_samples=100,
                     eps=1e-5,
                     sngp_var_weight=1.,
                     het_var_weight=1.,
                     omit_last_layer=False):
  """Builds ResNet50.

  Using strided conv, pooling, four groups of residual blocks, and pooling, the
  network maps spatial features of size 224x224 -> 112x112 -> 56x56 -> 28x28 ->
  14x14 -> 7x7 (Table 1 of He et al. (2015)).

  Args:
    input_shape: Shape tuple of input excluding batch dimension.
    batch_size: The batch size of the input layer. Required by the spectral
      normalization.
    num_classes: Number of output classes.
    num_factors: Number of factors for the heteroscedastic variance.
    use_mc_dropout: Whether to apply Monte Carlo dropout.
    dropout_rate: Dropout rate.
    filterwise_dropout:  Dropout whole convolutional filters instead of
      individual values in the feature map.
    use_gp_layer: Whether to use Gaussian process layer as the output layer.
    gp_hidden_dim: The hidden dimension of the GP layer, which corresponds to
      the number of random features used for the approximation.
    gp_scale: The length-scale parameter for the RBF kernel of the GP layer.
    gp_bias: The bias term for GP layer.
    gp_input_normalization: Whether to normalize the input using LayerNorm for
      GP layer. This is similar to automatic relevance determination (ARD) in
      the classic GP learning.
    gp_random_feature_type: The type of random feature to use for
      `RandomFeatureGaussianProcess`.
    gp_cov_discount_factor: The discount factor to compute the moving average of
      precision matrix.
    gp_cov_ridge_penalty: Ridge penalty parameter for GP posterior covariance.
    gp_output_imagenet_initializer: Whether to initialize GP output layer using
      Gaussian with small standard deviation (sd=0.01).
    use_spec_norm: Whether to apply spectral normalization.
    spec_norm_iteration: Number of power iterations to perform for estimating
      the spectral norm of weight matrices.
    spec_norm_bound: Upper bound to spectral norm of weight matrices.
    temperature: Float or scalar `Tensor` representing the softmax
      temperature.
    num_mc_samples: The number of Monte-Carlo samples used to estimate the
        predictive distribution.
    eps: Float. Clip probabilities into [eps, 1.0] softmax or
        [eps, 1.0 - eps] sigmoid before applying log (softmax), or inverse
        sigmoid.
    sngp_var_weight: Weight in [0,1] for the SNGP variance in the output.
    het_var_weight: Weight in [0,1] for the het. variance in the output.
    omit_last_layer: Optional. Omits the last pooling layer if it is set to
      True.

  Returns:
    tf.keras.Model.
  """
  dropout_layer = functools.partial(
      MonteCarloDropout,
      dropout_rate=dropout_rate,
      use_mc_dropout=use_mc_dropout,
      filterwise_dropout=filterwise_dropout)
  conv_layer = make_conv2d_layer(use_spec_norm=use_spec_norm,
                                 spec_norm_iteration=spec_norm_iteration,
                                 spec_norm_bound=spec_norm_bound)

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
  x = dropout_layer(x)
  x = tf.keras.layers.MaxPooling2D(3, strides=2, padding='same')(x)

  x = group(
      x, [64, 64, 256],
      stage=2,
      num_blocks=3,
      strides=1,
      conv_layer=conv_layer,
      dropout_layer=dropout_layer)
  x = group(
      x, [128, 128, 512],
      stage=3,
      num_blocks=4,
      strides=2,
      conv_layer=conv_layer,
      dropout_layer=dropout_layer)
  x = group(
      x, [256, 256, 1024],
      stage=4,
      num_blocks=6,
      strides=2,
      conv_layer=conv_layer,
      dropout_layer=dropout_layer)
  x = group(
      x, [512, 512, 2048],
      stage=5,
      num_blocks=3,
      strides=2,
      conv_layer=conv_layer,
      dropout_layer=dropout_layer)

  if omit_last_layer:
    return tf.keras.Model(inputs=inputs, outputs=x, name='resnet50')

  return resnet50_hetsngp_add_last_layer(
      inputs, x, num_classes, num_factors, use_gp_layer, gp_hidden_dim,
      gp_scale, gp_bias, gp_input_normalization, gp_random_feature_type,
      gp_cov_discount_factor, gp_cov_ridge_penalty,
      gp_output_imagenet_initializer, temperature, num_mc_samples, eps,
      sngp_var_weight, het_var_weight)

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

# Lint as: python3
"""Wide ResNet 28-10 with SNGP on CIFAR-10.

Spectral-normalized neural GP (SNGP) [1] is a simple method to improve
a deterministic neural network's uncertainty by applying spectral
normalization to hidden weights, and then replace the dense output layer with
a Gaussian process.
As a single-model method, SNGP can be combined with other classic
uncertainty techniques (e.g., Monte Carlo dropout, deep ensemble) to further
improve performance. This script supports adding Monte Carlo dropout to
SNGP by setting `use_mc_dropout=True` and `num_dropout_samples` to an integer
greater than 1.

## References:

[1]: Jeremiah Liu et al. Simple and Principled Uncertainty Estimation with
     Deterministic Deep Learning via Distance Awareness.
     _arXiv preprint arXiv:2006.10108_, 2020.
     https://arxiv.org/abs/2006.10108
[2]: Zhiyun Lu, Eugene Ie, Fei Sha. Uncertainty Estimation with Infinitesimal
     Jackknife.  _arXiv preprint arXiv:2006.07584_, 2020.
     https://arxiv.org/abs/2006.07584.
"""

import functools
from typing import Any, Dict, Iterable, Optional
import tensorflow as tf

import util as models_util  # local file import
# pylint: disable=invalid-name

BatchNormalization = functools.partial(
    tf.keras.layers.BatchNormalization,
    epsilon=1e-5,  # using epsilon and momentum defaults from Torch
    momentum=0.9)


def basic_block(
    inputs: tf.Tensor,
    filters: int,
    strides: int,
    l2: float,
    dropout_rate: float,
    use_mc_dropout: bool,
    conv_layer: tf.keras.layers.Layer):
  """Basic residual block of two 3x3 convs.

  Args:
    inputs: tf.Tensor.
    filters: Number of filters for Conv2D.
    strides: Stride dimensions for Conv2D.
    l2: L2 regularization coefficient.
    dropout_rate: Dropout rate.
    use_mc_dropout: Whether to apply Monte Carlo dropout.
    conv_layer: (tf.keras.layers.Layer) The conv layer used.

  Returns:
    tf.Tensor.
  """
  x = inputs
  y = inputs
  y = BatchNormalization(beta_regularizer=tf.keras.regularizers.l2(l2),
                         gamma_regularizer=tf.keras.regularizers.l2(l2))(y)
  y = tf.keras.layers.Activation('relu')(y)
  y = models_util.apply_dropout(y, dropout_rate, use_mc_dropout)

  y = conv_layer(filters=filters,
                 strides=strides,
                 kernel_regularizer=tf.keras.regularizers.l2(l2))(y)
  y = BatchNormalization(beta_regularizer=tf.keras.regularizers.l2(l2),
                         gamma_regularizer=tf.keras.regularizers.l2(l2))(y)
  y = tf.keras.layers.Activation('relu')(y)
  y = models_util.apply_dropout(y, dropout_rate, use_mc_dropout)

  y = conv_layer(filters=filters,
                 strides=1,
                 kernel_regularizer=tf.keras.regularizers.l2(l2))(y)
  if not x.shape.is_compatible_with(y.shape):
    x = conv_layer(filters=filters,
                   kernel_size=1,
                   strides=strides,
                   kernel_regularizer=tf.keras.regularizers.l2(l2))(x)
    y = models_util.apply_dropout(y, dropout_rate, use_mc_dropout)

  x = tf.keras.layers.add([x, y])
  return x


def group(inputs, filters, strides, num_blocks, l2, dropout_rate,
          use_mc_dropout, conv_layer):
  """Group of residual blocks."""
  x = basic_block(inputs,
                  filters=filters,
                  strides=strides,
                  l2=l2,
                  dropout_rate=dropout_rate,
                  use_mc_dropout=use_mc_dropout,
                  conv_layer=conv_layer)
  for _ in range(num_blocks - 1):
    x = basic_block(x,
                    filters=filters,
                    strides=1,
                    l2=l2,
                    dropout_rate=dropout_rate,
                    use_mc_dropout=use_mc_dropout,
                    conv_layer=conv_layer)
  return x


def wide_resnet(
    batch_size: Optional[int],
    input_shape: Iterable[int],
    depth: int,
    width_multiplier: int,
    num_classes: int,
    l2: float,
    dropout_rate: float,
    use_mc_dropout: bool,
    spec_norm_hparams: Dict[str, Any] = None,
    gp_layer_hparams: Dict[str, Any] = None):
  """Builds Wide ResNet.

  Following Zagoruyko and Komodakis (2016), it accepts a width multiplier on the
  number of filters. Using three groups of residual blocks, the network maps
  spatial features of size 32x32 -> 16x16 -> 8x8.

  Args:
    batch_size: (int) Value of the static per_replica batch size.
    input_shape: (tf.Tensor) shape of input to the model.
    depth: Total number of convolutional layers. "n" in WRN-n-k. It differs from
      He et al. (2015)'s notation which uses the maximum depth of the network
      counting non-conv layers like dense.
    width_multiplier: Integer to multiply the number of typical filters by. "k"
      in WRN-n-k.
    num_classes: Number of output classes.
    l2: L2 regularization coefficient.
    dropout_rate: Dropout rate.
    use_mc_dropout: Whether to apply Monte Carlo dropout.
    spec_norm_hparams: (dict) Hyperparameters for spectral normalization.
    gp_layer_hparams: (dict) Hyperparameters for Gaussian Process output layer.

  Returns:
    tf.keras.Model.
  """
  if (depth - 4) % 6 != 0:
    raise ValueError('depth should be 6n+4 (e.g., 16, 22, 28, 40).')
  num_blocks = (depth - 4) // 6
  inputs = tf.keras.layers.Input(shape=input_shape, batch_size=batch_size)

  # pylint: disable=invalid-name
  if spec_norm_hparams:
    spec_norm_bound = spec_norm_hparams['spec_norm_bound']
    spec_norm_iteration = spec_norm_hparams['spec_norm_iteration']
  else:
    spec_norm_bound = None
    spec_norm_iteration = None
  conv2d = models_util.make_conv2d_layer(
      kernel_size=3,
      use_bias=False,
      kernel_initializer='he_normal',
      activation=None,
      use_spec_norm=(spec_norm_hparams is not None),
      spec_norm_bound=spec_norm_bound,
      spec_norm_iteration=spec_norm_iteration)

  x = conv2d(filters=16,
             strides=1,
             kernel_regularizer=tf.keras.regularizers.l2(l2))(inputs)
  x = models_util.apply_dropout(x, dropout_rate, use_mc_dropout,
                                filter_wise_dropout=True)

  x = group(x,
            filters=16 * width_multiplier,
            strides=1,
            num_blocks=num_blocks,
            l2=l2,
            dropout_rate=dropout_rate,
            use_mc_dropout=use_mc_dropout,
            conv_layer=conv2d)
  x = group(x,
            filters=32 * width_multiplier,
            strides=2,
            num_blocks=num_blocks,
            l2=l2,
            dropout_rate=dropout_rate,
            use_mc_dropout=use_mc_dropout,
            conv_layer=conv2d)
  x = group(x,
            filters=64 * width_multiplier,
            strides=2,
            num_blocks=num_blocks,
            l2=l2,
            dropout_rate=dropout_rate,
            use_mc_dropout=use_mc_dropout,
            conv_layer=conv2d)
  x = BatchNormalization(beta_regularizer=tf.keras.regularizers.l2(l2),
                         gamma_regularizer=tf.keras.regularizers.l2(l2))(x)
  x = tf.keras.layers.Activation('relu')(x)
  x = tf.keras.layers.AveragePooling2D(pool_size=8)(x)
  x = tf.keras.layers.Flatten()(x)

  output_layer = models_util.make_output_layer(
      gp_layer_hparams=gp_layer_hparams)
  if gp_layer_hparams and gp_layer_hparams['gp_input_dim'] > 0:
    # Uses random projection to reduce the input dimension of the GP layer.
    x = tf.keras.layers.Dense(
        gp_layer_hparams['gp_input_dim'],
        kernel_initializer='random_normal',
        use_bias=False,
        trainable=False,
        name='gp_random_projection')(
            x)
  outputs = output_layer(num_classes, name='logits')(x)

  return tf.keras.Model(inputs=inputs, outputs=outputs)


def create_model(
    batch_size: Optional[int],
    depth: int = 28,
    width_multiplier: int = 10,
    input_shape: Iterable[int] = (32, 32, 3),
    num_classes: int = 10,
    l2_weight: float = 0.0,
    dropout_rate: float = 0.0,
    use_mc_dropout: bool = False,
    spec_norm_hparams: Dict[str, Any] = None,
    gp_layer_hparams: Dict[str, Any] = None,
    **unused_kwargs: Dict[str, Any]) -> tf.keras.models.Model:
  """Return wide resnet model."""
  return wide_resnet(
      batch_size=batch_size,
      input_shape=input_shape,
      depth=depth,
      width_multiplier=width_multiplier,
      num_classes=num_classes,
      l2=l2_weight,
      dropout_rate=dropout_rate,
      use_mc_dropout=use_mc_dropout,
      spec_norm_hparams=spec_norm_hparams,
      gp_layer_hparams=gp_layer_hparams)

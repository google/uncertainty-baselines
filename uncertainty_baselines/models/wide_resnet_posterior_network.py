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

# Lint as: python3
"""Wide Residual Posterior Network (see https://arxiv.org/abs/2006.09239)."""

import functools
from typing import Dict, Iterable, Optional

import edward2 as ed
import tensorflow as tf

HP_KEYS = ('bn_l2', 'input_conv_l2', 'group_1_conv_l2', 'group_2_conv_l2',
           'group_3_conv_l2', 'dense_kernel_l2', 'dense_bias_l2')

BatchNormalization = functools.partial(  # pylint: disable=invalid-name
    tf.keras.layers.BatchNormalization,
    epsilon=1e-5,  # using epsilon and momentum defaults from Torch
    momentum=0.9)


def Conv2D(filters, seed=None, **kwargs):  # pylint: disable=invalid-name
  """Conv2D layer that is deterministically initialized."""
  default_kwargs = {
      'kernel_size': 3,
      'padding': 'same',
      'use_bias': False,
      # Note that we need to use the class constructor for the initializer to
      # get deterministic initialization.
      'kernel_initializer': tf.keras.initializers.HeNormal(seed=seed),
  }
  # Override defaults with the passed kwargs.
  default_kwargs.update(kwargs)
  return tf.keras.layers.Conv2D(filters, **default_kwargs)


def basic_block(
    inputs: tf.Tensor,
    filters: int,
    strides: int,
    conv_l2: float,
    bn_l2: float,
    seed: int,
    version: int) -> tf.Tensor:
  """Basic residual block of two 3x3 convs.

  Args:
    inputs: tf.Tensor.
    filters: Number of filters for Conv2D.
    strides: Stride dimensions for Conv2D.
    conv_l2: L2 regularization coefficient for the conv kernels.
    bn_l2: L2 regularization coefficient for the batch norm layers.
    seed: random seed used for initialization.
    version: 1, indicating the original ordering from He et al. (2015); or 2,
      indicating the preactivation ordering from He et al. (2016).

  Returns:
    tf.Tensor.
  """
  x = inputs
  y = inputs
  if version == 2:
    y = BatchNormalization(beta_regularizer=tf.keras.regularizers.l2(bn_l2),
                           gamma_regularizer=tf.keras.regularizers.l2(bn_l2))(y)
    y = tf.keras.layers.Activation('relu')(y)
  seeds = tf.random.experimental.stateless_split([seed, seed + 1], 3)[:, 0]
  y = Conv2D(filters,
             strides=strides,
             seed=seeds[0],
             kernel_regularizer=tf.keras.regularizers.l2(conv_l2))(y)
  y = BatchNormalization(beta_regularizer=tf.keras.regularizers.l2(bn_l2),
                         gamma_regularizer=tf.keras.regularizers.l2(bn_l2))(y)
  y = tf.keras.layers.Activation('relu')(y)
  y = Conv2D(filters,
             strides=1,
             seed=seeds[1],
             kernel_regularizer=tf.keras.regularizers.l2(conv_l2))(y)
  if version == 1:
    y = BatchNormalization(beta_regularizer=tf.keras.regularizers.l2(bn_l2),
                           gamma_regularizer=tf.keras.regularizers.l2(bn_l2))(y)
  if not x.shape.is_compatible_with(y.shape):
    x = Conv2D(filters,
               kernel_size=1,
               strides=strides,
               seed=seeds[2],
               kernel_regularizer=tf.keras.regularizers.l2(conv_l2))(x)
  x = tf.keras.layers.add([x, y])
  if version == 1:
    x = tf.keras.layers.Activation('relu')(x)
  return x


def group(inputs, filters, strides, num_blocks, conv_l2, bn_l2, version, seed):
  """Group of residual blocks."""
  seeds = tf.random.experimental.stateless_split(
      [seed, seed + 1], num_blocks)[:, 0]
  x = basic_block(
      inputs,
      filters=filters,
      strides=strides,
      conv_l2=conv_l2,
      bn_l2=bn_l2,
      version=version,
      seed=seeds[0])
  for i in range(num_blocks - 1):
    x = basic_block(
        x,
        filters=filters,
        strides=1,
        conv_l2=conv_l2,
        bn_l2=bn_l2,
        version=version,
        seed=seeds[i + 1])
  return x


def _parse_hyperparameters(l2: float, hps: Dict[str, float]):
  """Extract the L2 parameters for the dense, conv and batch-norm layers."""

  assert_msg = ('Ambiguous hyperparameter specifications: either l2 or hps '
                'must be provided (received {} and {}).'.format(l2, hps))
  is_specified = lambda h: bool(h) and all(v is not None for v in h.values())
  only_l2_is_specified = l2 is not None and not is_specified(hps)
  only_hps_is_specified = l2 is None and is_specified(hps)
  assert only_l2_is_specified or only_hps_is_specified, assert_msg
  if only_hps_is_specified:
    assert_msg = 'hps must contain the keys {}!={}.'.format(HP_KEYS, hps.keys())
    assert set(hps.keys()).issuperset(HP_KEYS), assert_msg
    return hps
  else:
    return {k: l2 for k in HP_KEYS}


def wide_resnet_posterior_network(
    input_shape: Iterable[int],
    depth: int,
    width_multiplier: int,
    num_classes: int,
    l2: float,
    version: int = 2,
    seed: int = 42,
    class_counts: Optional[Iterable[int]] = None,
    latent_dim: int = 16,
    flow_depth: int = 6,
    flow_width: Optional[int] = None,
    flow_type: str = 'maf',
    hps: Optional[Dict[str, float]] = None) -> tf.keras.models.Model:
  """Builds Wide ResNet Posterior Network.

  Following Zagoruyko and Komodakis (2016), it accepts a width multiplier on the
  number of filters. Using three groups of residual blocks, the network maps
  spatial features of size 32x32 -> 16x16 -> 8x8.

  Args:
    input_shape: tf.Tensor storing the shape of the inputs.
    depth: Total number of convolutional layers. "n" in WRN-n-k. It differs from
      He et al. (2015)'s notation which uses the maximum depth of the network
      counting non-conv layers like dense.
    width_multiplier: Integer to multiply the number of typical filters by. "k"
      in WRN-n-k.
    num_classes: Number of output classes.
    l2: L2 regularization coefficient.
    version: 1, indicating the original ordering from He et al. (2015); or 2,
      indicating the preactivation ordering from He et al. (2016).
    seed: random seed used for initialization.
    class_counts: List of counts of training examples per class.
    latent_dim: Dimensionality of the latent space.
    flow_depth: Number of latent flows to stack into a deep flow.
    flow_width: Width of the hidden layers inside the MAF flows.
    flow_type: Type of the normalizing flow to be used; has to be one
                    of 'maf', 'radial', or 'affine'.
    hps: Fine-grained specs of the hyperparameters, as a Dict[str, float].

  Returns:
    tf.keras.Model.
  """
  l2_reg = tf.keras.regularizers.l2
  hps = _parse_hyperparameters(l2, hps)

  seeds = tf.random.experimental.stateless_split([seed, seed + 1], 5)[:, 0]
  if (depth - 4) % 6 != 0:
    raise ValueError('depth should be 6n+4 (e.g., 16, 22, 28, 40).')
  num_blocks = (depth - 4) // 6

  if class_counts is None:
    class_counts = tf.ones(num_classes)
  class_counts = tf.convert_to_tensor(class_counts, dtype=tf.float32)

  inputs = tf.keras.layers.Input(shape=input_shape)
  x = Conv2D(16,
             strides=1,
             seed=seeds[0],
             kernel_regularizer=l2_reg(hps['input_conv_l2']))(inputs)
  if version == 1:
    x = BatchNormalization(beta_regularizer=l2_reg(hps['bn_l2']),
                           gamma_regularizer=l2_reg(hps['bn_l2']))(x)
    x = tf.keras.layers.Activation('relu')(x)
  x = group(x,
            filters=16 * width_multiplier,
            strides=1,
            num_blocks=num_blocks,
            conv_l2=hps['group_1_conv_l2'],
            bn_l2=hps['bn_l2'],
            version=version,
            seed=seeds[1])
  x = group(x,
            filters=32 * width_multiplier,
            strides=2,
            num_blocks=num_blocks,
            conv_l2=hps['group_2_conv_l2'],
            bn_l2=hps['bn_l2'],
            version=version,
            seed=seeds[2])
  x = group(x,
            filters=64 * width_multiplier,
            strides=2,
            num_blocks=num_blocks,
            conv_l2=hps['group_3_conv_l2'],
            bn_l2=hps['bn_l2'],
            version=version,
            seed=seeds[3])
  if version == 2:
    x = BatchNormalization(beta_regularizer=l2_reg(hps['bn_l2']),
                           gamma_regularizer=l2_reg(hps['bn_l2']))(x)
    x = tf.keras.layers.Activation('relu')(x)
  x = tf.keras.layers.AveragePooling2D(pool_size=8)(x)
  x = tf.keras.layers.Flatten()(x)

  latents = tf.keras.layers.Dense(latent_dim)(x)
  postnet_layer = ed.layers.PosteriorNetworkLayer(
      num_classes=num_classes,
      flow_type=flow_type,
      flow_depth=flow_depth,
      flow_width=flow_width,
      class_counts=class_counts)
  alphas = postnet_layer(latents)

  return tf.keras.Model(
      inputs=inputs,
      outputs=alphas,
      name='wide_resnet-posterior_network-{}-{}'.format(depth,
                                                        width_multiplier))

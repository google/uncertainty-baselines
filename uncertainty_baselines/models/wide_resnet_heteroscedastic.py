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

"""Wide Residual Network."""

import functools
from typing import Any, Dict, Iterable, Optional

import edward2 as ed
import tensorflow as tf

HP_KEYS = ('bn_l2', 'input_conv_l2', 'group_1_conv_l2', 'group_2_conv_l2',
           'group_3_conv_l2', 'dense_kernel_l2', 'dense_bias_l2')

BatchNormalization = functools.partial(  # pylint: disable=invalid-name
    tf.keras.layers.BatchNormalization,
    epsilon=1e-5,  # using epsilon and momentum defaults from Torch
    momentum=0.9)
Conv2D = functools.partial(  # pylint: disable=invalid-name
    tf.keras.layers.Conv2D,
    kernel_size=3,
    padding='same',
    use_bias=False,
    kernel_initializer='he_normal')


def basic_block(
    inputs: tf.Tensor,
    filters: int,
    strides: int,
    conv_l2: float,
    bn_l2: float,
    version: int) -> tf.Tensor:
  """Basic residual block of two 3x3 convs.

  Args:
    inputs: tf.Tensor.
    filters: Number of filters for Conv2D.
    strides: Stride dimensions for Conv2D.
    conv_l2: L2 regularization coefficient for the conv kernels.
    bn_l2: L2 regularization coefficient for the batch norm layers.
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
  y = Conv2D(filters,
             strides=strides,
             kernel_regularizer=tf.keras.regularizers.l2(conv_l2))(y)
  y = BatchNormalization(beta_regularizer=tf.keras.regularizers.l2(bn_l2),
                         gamma_regularizer=tf.keras.regularizers.l2(bn_l2))(y)
  y = tf.keras.layers.Activation('relu')(y)
  y = Conv2D(filters,
             strides=1,
             kernel_regularizer=tf.keras.regularizers.l2(conv_l2))(y)
  if version == 1:
    y = BatchNormalization(beta_regularizer=tf.keras.regularizers.l2(bn_l2),
                           gamma_regularizer=tf.keras.regularizers.l2(bn_l2))(y)
  if not x.shape.is_compatible_with(y.shape):
    x = Conv2D(filters,
               kernel_size=1,
               strides=strides,
               kernel_regularizer=tf.keras.regularizers.l2(conv_l2))(x)
  x = tf.keras.layers.add([x, y])
  if version == 1:
    x = tf.keras.layers.Activation('relu')(x)
  return x


def group(inputs, filters, strides, num_blocks, conv_l2, bn_l2, version):
  """Group of residual blocks."""
  x = basic_block(
      inputs,
      filters=filters,
      strides=strides,
      conv_l2=conv_l2,
      bn_l2=bn_l2,
      version=version)
  for _ in range(num_blocks - 1):
    x = basic_block(
        x,
        filters=filters,
        strides=1,
        conv_l2=conv_l2,
        bn_l2=bn_l2,
        version=version)
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


def wide_resnet_heteroscedastic(
    input_shape: Iterable[int],
    depth: int,
    width_multiplier: int,
    num_classes: int,
    l2: float,
    version: int,
    temperature: float,
    num_factors: int,
    num_mc_samples: int = 5000,
    multiclass: bool = True,
    eps: float = 1e-5,
    hps: Optional[Dict[str, float]] = None) -> tf.keras.models.Model:
  """Builds Wide ResNet.

  Following Zagoruyko and Komodakis (2016), it accepts a width multiplier on the
  number of filters. Using three groups of residual blocks, the network maps
  spatial features of size 32x32 -> 16x16 -> 8x8.

  Args:
    input_shape: tf.Tensor.
    depth: Total number of convolutional layers. "n" in WRN-n-k. It differs from
      He et al. (2015)'s notation which uses the maximum depth of the network
      counting non-conv layers like dense.
    width_multiplier: Integer to multiply the number of typical filters by. "k"
      in WRN-n-k.
    num_classes: Number of output classes.
    l2: L2 regularization coefficient.
    version: 1, indicating the original ordering from He et al. (2015); or 2,
      indicating the preactivation ordering from He et al. (2016).
    temperature: Float or scalar `Tensor` representing the softmax
      temperature.
    num_factors: Integer. Number of factors to use in approximation to full
      rank covariance matrix. If num_factors <= 0, then the diagonal covariance
      method MCSoftmaxDense is used.
    num_mc_samples: The number of Monte-Carlo samples used to estimate the
        predictive distribution.
    multiclass: Boolean. If True then return a multiclass classifier, otherwise
      a multilabel classifier.
    eps: Float. Clip probabilities into [eps, 1.0] softmax or
        [eps, 1.0 - eps] sigmoid before applying log (softmax), or inverse
        sigmoid.
    hps: Fine-grained specs of the hyperparameters, as a Dict[str, float].

  Returns:
    tf.keras.Model.
  """
  l2_reg = tf.keras.regularizers.l2
  hps = _parse_hyperparameters(l2, hps)

  if (depth - 4) % 6 != 0:
    raise ValueError('depth should be 6n+4 (e.g., 16, 22, 28, 40).')
  num_blocks = (depth - 4) // 6
  inputs = tf.keras.layers.Input(shape=input_shape)
  x = Conv2D(16,
             strides=1,
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
            version=version)
  x = group(x,
            filters=32 * width_multiplier,
            strides=2,
            num_blocks=num_blocks,
            conv_l2=hps['group_2_conv_l2'],
            bn_l2=hps['bn_l2'],
            version=version)
  x = group(x,
            filters=64 * width_multiplier,
            strides=2,
            num_blocks=num_blocks,
            conv_l2=hps['group_3_conv_l2'],
            bn_l2=hps['bn_l2'],
            version=version)
  if version == 2:
    x = BatchNormalization(beta_regularizer=l2_reg(hps['bn_l2']),
                           gamma_regularizer=l2_reg(hps['bn_l2']))(x)
    x = tf.keras.layers.Activation('relu')(x)
  x = tf.keras.layers.AveragePooling2D(pool_size=8)(x)
  x = tf.keras.layers.Flatten()(x)

  het_layer_args = {'temperature': temperature,
                    'train_mc_samples': num_mc_samples,
                    'test_mc_samples': num_mc_samples,
                    'share_samples_across_batch': True,
                    'logits_only': True, 'eps': eps,
                    'dtype': tf.float32, 'name': 'fc100',
                    'kernel_regularizer': l2_reg(hps['dense_kernel_l2']),
                    'bias_regularizer': l2_reg(hps['dense_bias_l2'])}
  if multiclass:
    het_layer_args.update({'num_classes': num_classes})
    if num_factors <= 0:
      output_layer = ed.layers.MCSoftmaxDense(**het_layer_args)
    else:
      het_layer_args.update({'num_factors': num_factors})
      output_layer = ed.layers.MCSoftmaxDenseFA(**het_layer_args)
  else:
    het_layer_args.update({'num_outputs': num_classes,
                           'num_factors': num_factors})
    output_layer = ed.layers.MCSigmoidDenseFA(**het_layer_args)

  x = output_layer(x)

  return tf.keras.Model(
      inputs=inputs,
      outputs=x,
      name='wide_resnet-{}-{}'.format(depth, width_multiplier))


def create_model(
    batch_size: Optional[int],
    depth: int,
    width_multiplier: int,
    temperature: float,
    num_factors: int,
    input_shape: Iterable[int] = (32, 32, 3),
    num_classes: int = 10,
    l2_weight: float = 0.0,
    version: int = 1,
    num_mc_samples: int = 5000,
    multiclass: bool = True,
    eps: float = 1e-5,
    **unused_kwargs: Dict[str, Any]) -> tf.keras.models.Model:
  """Creates model."""
  del batch_size  # unused arg
  return wide_resnet_heteroscedastic(input_shape=input_shape,
                                     depth=depth,
                                     width_multiplier=width_multiplier,
                                     num_classes=num_classes,
                                     l2=l2_weight,
                                     version=version,
                                     temperature=temperature,
                                     num_factors=num_factors,
                                     num_mc_samples=num_mc_samples,
                                     multiclass=multiclass,
                                     eps=eps)

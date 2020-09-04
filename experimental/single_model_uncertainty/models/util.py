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

# Lint as: python3
"""Collection of shared utility functions."""

import functools
from typing import Optional, Tuple, Union
from absl import logging

import edward2 as ed
import tensorflow as tf


# pylint: disable=invalid-name
_GP_HPARAM_LIST = [
    'gp_input_dim', 'gp_hidden_dim', 'gp_scale', 'gp_bias',
    'gp_input_normalization', 'gp_cov_discount_factor', 'gp_cov_ridge_penalty'
]


def make_conv2d_layer(num_filters: Optional[int] = None,
                      kernel_size: Union[Tuple[int, int], int] = None,
                      strides: Union[Tuple[int, int], int] = None,
                      use_bias: Optional[bool] = True,
                      kernel_initializer: Optional[str] = 'glorot_uniform',
                      activation: Optional[str] = 'relu',
                      use_spec_norm: Optional[bool] = False,
                      spec_norm_bound: Optional[float] = None,
                      spec_norm_iteration: Optional[int] = None):
  """Creates an 2D convolutional layer with/without spectral normalization.

  Args:
    num_filters: (int) Number of filters to apply to input.
    kernel_size: (Tuple[int, int]) Tuple of 2 integers, specifying the width and
    height of the convolution window.
    strides: (Tuple[int, int]) Tuple of 2 integers, specifying the stride length
    of the convolution.
    use_bias: (bool) Whether to have a bias term.
    kernel_initializer: (str) Kernel initialization scheme to use.
    activation: (str) Whether to have an activation after the layer.
    use_spec_norm: (bool) Whether to apply spectral normalization.
    spec_norm_bound: (Optional[float]) Upper bound to spectral norm of weight
    matrices. Cannot be None if use_spec_norm=True.
    spec_norm_iteration: (Optional[int]) Number of power iterations to perform
    for estimating the spectral norm of weight matrices. Cannot be None if
    use_spec_norm=True.

  Returns:
    tf.keras.layers.Layer.

  Raises:
    ValueError: If use_spec_norm=True but spec_norm_iteration or
    spec_norm_iteration is None.
  """
  print('conv2d: use_spec_norm={}, spec_norm_iteration={}, spec_norm_bound={}'
        .format(use_spec_norm, spec_norm_iteration, spec_norm_bound))
  # padding='same' means adding paddings to the top and bottom of the inputs
  Conv2DBase = functools.partial(
      tf.keras.layers.Conv2D,
      filters=num_filters,
      kernel_size=kernel_size,
      strides=strides,
      padding='same',  # SpectralNormalizationConv2D only support padding='same'
      use_bias=use_bias,
      kernel_initializer=kernel_initializer,
      activation=activation,
      data_format='channels_last')

  if not use_spec_norm:
    return Conv2DBase

  if not (spec_norm_iteration and spec_norm_bound):
    raise ValueError(
        'use_spec_norm=True but spec_norm_iteration or spec_norm_bound '
        'is None.')

  def Conv2DNormed(*conv_args, **conv_kwargs):
    conv_layer = Conv2DBase(*conv_args, **conv_kwargs)
    return ed.layers.SpectralNormalizationConv2D(
        conv_layer,
        iteration=spec_norm_iteration,
        norm_multiplier=spec_norm_bound)

  logging.info('Use spectral normalizaiotn, iteration %s, norm bound %s',
               spec_norm_iteration, spec_norm_bound)

  return Conv2DNormed


def make_dense_layer(num_units: Optional[int] = None,
                     use_bias: Optional[bool] = True,
                     kernel_initializer: Optional[str] = 'glorot_uniform',
                     activation: Optional[str] = 'relu',
                     use_spec_norm: Optional[bool] = False,
                     spec_norm_bound: Optional[float] = None,
                     spec_norm_iteration: Optional[int] = None):
  """Creates a dense layer with/without spectral normalization.

  Args:
    num_units: (int) Dimensionality of the output space.
    use_bias: (bool) Whether to have a bias term.
    kernel_initializer: (str) Kernel initialization scheme to use.
    activation: (str) Whether to have an activation after the layer.
    use_spec_norm: (bool) Whether to apply spectral normalization.
    spec_norm_bound: (Optional[float]) Upper bound to spectral norm of weight
      matrices. Cannot be None if use_spec_norm=True.
    spec_norm_iteration: (Optional[int]) Number of power iterations to perform
      for estimating the spectral norm of weight matrices. Cannot be None if
      use_spec_norm=True.

  Returns:
    tf.keras.layers.Layer.

  Raises:
      ValueError: If use_spec_norm=True but spec_norm_iteration or
      spec_norm_iteration is None.
  """
  DenseBase = functools.partial(
      tf.keras.layers.Dense,
      units=num_units,
      use_bias=use_bias,
      kernel_initializer=kernel_initializer,
      activation=activation)

  if not use_spec_norm:
    return DenseBase

  if not (spec_norm_iteration and spec_norm_bound):
    raise ValueError(
        'use_spec_norm=True but spec_norm_iteration or spec_norm_iteration '
        'is None.')

  def DenseNormed(*dense_args, **dense_kwargs):
    dense_layer = DenseBase(*dense_args, **dense_kwargs)
    return ed.layers.SpectralNormalization(
        dense_layer,
        iteration=spec_norm_iteration,
        norm_multiplier=spec_norm_bound)

  logging.info('Use spectral normalizaiotn, iteration %s, norm bound %s',
               spec_norm_iteration, spec_norm_bound)

  return DenseNormed


def apply_dropout(inputs: tf.Tensor,
                  dropout_rate: float,
                  use_mc_dropout: bool,
                  filter_wise_dropout: bool = False,
                  name: str = None):
  """Applies a filter-wise dropout layer to the inputs."""
  logging.info('apply_dropout input shape %s', inputs.shape)
  if filter_wise_dropout:
    noise_shape = [1] * len(inputs.shape)
    noise_shape[0] = inputs.shape[0]
    noise_shape[-1] = inputs.shape[-1]
  else:
    noise_shape = None

  dropout_layer = tf.keras.layers.Dropout(
      dropout_rate, noise_shape=noise_shape, name=name)

  if use_mc_dropout:
    return dropout_layer(inputs, training=True)

  return dropout_layer(inputs)


def make_output_layer(gp_layer_hparams=None):
  """Make an output layer with/without Gaussian process."""
  if not gp_layer_hparams:
    return tf.keras.layers.Dense

  if not all(x in gp_layer_hparams.keys() for x in _GP_HPARAM_LIST):
    raise ValueError('GP layer is in use but hyperparameters are incomplete.')

  output_layer = functools.partial(
      ed.layers.RandomFeatureGaussianProcess,
      num_inducing=gp_layer_hparams['gp_hidden_dim'],
      gp_kernel_scale=gp_layer_hparams['gp_scale'],
      gp_output_bias=gp_layer_hparams['gp_bias'],
      normalize_input=gp_layer_hparams['gp_input_normalization'],
      gp_cov_momentum=gp_layer_hparams['gp_cov_discount_factor'],
      gp_cov_ridge_penalty=gp_layer_hparams['gp_cov_ridge_penalty'])

  return output_layer

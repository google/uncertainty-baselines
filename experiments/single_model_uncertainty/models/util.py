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
from typing import Optional, Tuple
from absl import logging
from edward2.experimental import sngp
import tensorflow.compat.v2 as tf


# pylint: disable=invalid-name


def mean_field_logits(logits, covmat, mean_field_factor=1.):
  """Adjust the predictive logits so its softmax approximates posterior mean."""
  logits_scale = tf.sqrt(1. + tf.linalg.diag_part(covmat) * mean_field_factor)
  if mean_field_factor > 0:
    logits = logits / tf.expand_dims(logits_scale, axis=-1)

  return logits


def make_conv2d_layer(num_filters: int, kernel_size: Tuple[int, int],
                      strides: Tuple[int, int], use_spec_norm: bool,
                      spec_norm_bound: Optional[float],
                      spec_norm_iteration: Optional[int]):
  """Creates an 2D convolutional layer with/without spectral normalization.

  Args:
    num_filters: (int) Number of filters to apply to input.
    kernel_size: (Tuple[int, int]) Tuple of 2 integers, specifying the width and
    height of the convolution window.
    strides: (Tuple[int, int]) Tuple of 2 integers, specifying the stride length
    of the convolution.
    use_spec_norm: (bool) Whether to apply spectral normalization.
    spec_norm_bound: (Optional[float]) Upper bound to spectral norm of weight
    matrices. Cannot be None if use_spec_norm=True.
    spec_norm_iteration: (Optional[int]) Number of power iterations to perform
    for estimating the spectral norm of weight matrices. Cannot be None if
    use_spec_norm=True.

  Returns:
    tf.keras.layers.Layer.
  """
  # padding='same' means adding paddings to the top and bottom of the inputs
  Conv2DBase = functools.partial(
      tf.keras.layers.Conv2D,
      filters=num_filters,
      kernel_size=kernel_size,
      strides=strides,
      padding='same',  # SpectralNormalizationConv2D only support padding='same'
      data_format='channels_last',
      activation=tf.keras.activations.relu,
      name='conv')

  if use_spec_norm:
    if spec_norm_iteration and spec_norm_bound:

      def Conv2DNormed(*conv_args, **conv_kwargs):
        conv_layer = Conv2DBase(*conv_args, **conv_kwargs)
        return sngp.SpectralNormalizationConv2D(
            conv_layer,
            iteration=spec_norm_iteration,
            norm_multiplier=spec_norm_bound)

      logging.info('Use spectral normalizaiotn, iteration %s, norm bound %s',
                   spec_norm_iteration, spec_norm_bound)
    else:
      raise ValueError(
          'use_spec_norm=True but spec_norm_iteration or spec_norm_iteration '
          'is None.')
    return Conv2DNormed

  return Conv2DBase


def apply_dropout(inputs: tf.Tensor, dropout_rate: float, use_mc_dropout: bool,
                  filter_wise_dropout: bool):
  """Applies a filter-wise dropout layer to the inputs."""
  logging.info('apply_dropout input shape %s', inputs.shape)
  noise_shape = [inputs.shape[0], 1, inputs.shape[2]
                ] if filter_wise_dropout else None
  dropout_layer = tf.keras.layers.Dropout(dropout_rate, noise_shape=noise_shape)

  if use_mc_dropout:
    return dropout_layer(inputs, training=True)

  return dropout_layer(inputs)

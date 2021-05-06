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

"""Utilities for variational models."""

import math
import warnings

try:
  import edward2 as ed  # pylint: disable=g-import-not-at-top
except ImportError as e:
  warnings.warn(f'Skipped due to ImportError: {e}')


def _compute_fans(shape):
  """Computes the number of input and output units for a weight shape.

  Args:
    shape: Integer shape tuple or TF tensor shape.

  Returns:
    A tuple of integer scalars (fan_in, fan_out).
  """
  if len(shape) < 1:  # Just to avoid errors for constants.
    fan_in = 1
    fan_out = 1
  elif len(shape) == 1:
    fan_in = shape[0]
    fan_out = shape[0]
  elif len(shape) == 2:
    fan_in = shape[0]
    fan_out = shape[1]
  else:
    # Assuming convolution kernels (2D, 3D, or more).
    # kernel shape: (..., input_depth, depth)
    receptive_field_size = 1
    for dim in shape[:-2]:
      receptive_field_size *= dim
    fan_in = shape[-2] * receptive_field_size
    fan_out = shape[-1] * receptive_field_size
  return int(fan_in), int(fan_out)


def get_kernel_regularizer_class(tied_mean_prior: bool):
  """Determine the regularizer class based on provided prior settings.

  Args:
    tied_mean_prior: bool, if True, fix the mean of the prior to that of the
      variational posterior, which causes the KL to only penalize the weight
      posterior's standard deviation, and not its mean.

  Returns:
    Regularizer class.
  """
  if tied_mean_prior:
    return ed.regularizers.NormalKLDivergenceWithTiedMean
  else:
    # Can optionally set the mean of the untied priors -
    # we default to a standard Normal
    return ed.regularizers.NormalKLDivergence


def init_kernel_regularizer(kernel_regularizer_class,
                            dataset_size,
                            prior_stddev,
                            inputs,
                            n_filters=None,
                            kernel_size=None,
                            n_outputs=None,
                            prior_stddev_scale=2.,
                            prior_stddev_mode='fan_in'):
  """Initialize the kernel regularizer.

  Works with:
    - 2D convolutional layers: must specify n_filters and kernel_size.
    - Dense layers: must specify n_outputs.

  If no fixed prior_stddev is provided, we compute
    `prior_stddev = sqrt(scale / n)`
  where n is:

    - number of input units in the weight tensor, if mode = "fan_in"
    - number of output units, if mode = "fan_out"
    - average of the numbers of input and output units, if mode = "fan_avg"

  With scale = 2 and mode = "fan_in", we recover the stddev used in
  He initialization.

  This implementation is based on keras.initializers.VarianceScaling.

  Args:
    kernel_regularizer_class: tf.keras.regularizers.Regularizer
    dataset_size: int, number of examples in the train data
    prior_stddev: float, a fixed stddev for the prior. If it is not supplied, we
    inputs: tf.Tensor.
    n_filters: int, number of filters at current 2D Conv layer.
    kernel_size: int, we assume that each channel of the 2D convolutional kernel
      has shape (kernel_size, kernel_size).
    n_outputs: int, number of outputs of the layer if it is Dense. compute a
      prior_stddev dependent on input size, defaulting to the stddev computation
      used in He initialization.
    prior_stddev_scale: float, used in prior_stddev computation sqrt(scale / n).
    prior_stddev_mode: str, determines mode used in prior_stddev computation
      sqrt(scale / n).

  Returns:
    Initialized tf.keras.regularizers.Regularizer
  """
  if prior_stddev is not None:
    return kernel_regularizer_class(
        stddev=prior_stddev, scale_factor=1. / dataset_size)

  # Only supports finding kernel shape for 2D convs and dense layers
  if kernel_size is not None and n_filters is not None:
    kernel_shape = get_2d_conv_kernel_shape(inputs, n_filters, kernel_size)
  elif n_outputs is not None:
    kernel_shape = get_dense_shape(inputs, n_outputs)
  else:
    raise NotImplementedError(
        'Only support finding kernel shape for 2D Conv and Dense layers.')

  fan_in, fan_out = _compute_fans(kernel_shape)

  if prior_stddev_mode == 'fan_in':
    prior_stddev_scale /= max(1., fan_in)
  elif prior_stddev_mode == 'fan_out':
    prior_stddev_scale /= max(1., fan_out)
  else:
    prior_stddev_scale /= max(1., (fan_in + fan_out) / 2.)

  stddev = math.sqrt(prior_stddev_scale)
  return kernel_regularizer_class(stddev=stddev, scale_factor=1. / dataset_size)


def get_2d_conv_kernel_shape(inputs, n_filters, kernel_size):
  """Compute the shape of a 2D convolutional kernel.

  This can be used to provide a shape-dependent value at initialization of the
  Layer itself.

  Args:
    inputs: tf.Tensor.
    n_filters: int, number of filters at current layer.
    kernel_size: int, we assume that each channel of the 2D convolutional kernel
      has shape (kernel_size, kernel_size).

  Returns:
     Tuple[int], integer shape tuple
  """
  # 2D kernel shape: (height, width, in_channels, out_channels)
  return kernel_size, kernel_size, inputs.shape[-1], n_filters


def get_dense_shape(inputs, n_outputs):
  """Compute the shape of a dense layer.

  Args:
    inputs: tf.Tensor.
    n_outputs: int

  Returns:
     Tuple[int], integer shape tuple
  """
  return inputs.shape[-1], n_outputs

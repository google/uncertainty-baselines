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

"""Library implementations the secondary losses.

Contains implementations for AvUC and Soft-binning ECE.
"""

import math
import numpy as np
import tensorflow as tf

EPS = 1e-5

def compute_squared_error_label_binning_tensorflow(logits,
                                                   y,
                                                   m,
                                                   temperature,
                                                   name_scope='sq_err_loss',
                                                   precision=32):
  """Computes and returns the squared soft-binned ECE (label-binned) tensor.

  Soft-binned ECE (label-binned, l2-norm) is defined in equation (12) in this
  paper: https://arxiv.org/abs/2108.00106. It is a softened version of ECE
  (label-binned) which is defined in equation (7).

  Args:
    logits: tensor of predicted logits of (batch-size, num-classes) shapes
    y: tensor of labels in [0,num-classes) of (batch-size,) shape
    m: number of bins
    temperature: soft binning temperature
    name_scope: name
    precision: precision for tf.int and tf.float dtypes

  Returns:
    A tensor of () shape containing a single value: squared soft-binned ECE.
  """

  if precision == 32:
    int_dtype = tf.int32
    float_dtype = tf.float32
  elif precision == 64:
    int_dtype = tf.int64
    float_dtype = tf.float64
  else:
    raise NotImplementedError

  with tf.name_scope(name_scope) as scope:  # pylint: disable=unused-variable

    # <float_dtype>[N, n]
    logits = tf.cast(logits, float_dtype, name='logits')

    # <int_dtype>[N]
    y = tf.cast(y, int_dtype, name='y')

    # <float_dtype>[m+1]
    bins = tf.linspace(
        tf.constant(1. / logits.shape[1], dtype=float_dtype),
        1,
        m + 1,
        name='bins')

    # <float_dtype>[m]
    b = tf.identity(bins[1:] * .5 + bins[:-1] * .5, name='b')

    # <float_dtype>[m]
    q_hat = tf.nn.softmax(logits, axis=1, name='q_hat')

    # <int_dtype>[N]
    y_hat = tf.math.argmax(q_hat, axis=1, output_type=int_dtype, name='y_hat')

    # <float_dtype>[N]
    p_hat = tf.reduce_max(q_hat, axis=1, name='y_hat')

    # <float_dtype>[N]
    a = tf.cast(tf.math.equal(y_hat, y), dtype=float_dtype, name='a')

    # <float_dtype>[N, m]
    c_numerator = tf.math.exp(
        -tf.math.pow(
            tf.expand_dims(b, 0) - tf.expand_dims(p_hat, 1),
            2,
        ) / temperature,
        name='c_numerator')

    # <float_dtype>[N]
    c_denominator = tf.einsum('ij->i', c_numerator, name='c_denominator')

    # <float_dtype>[N, m]
    c_pre = tf.identity(
        c_numerator / tf.expand_dims(c_denominator, 1), name='c_pre')

    # <float_dtype>[N, m]
    c = tf.identity((1 - EPS) * c_pre + EPS * (1 / m), name='c')

    # <float_dtype>[m]
    a_bar_numerator = tf.einsum('ij,i->j', c, a, name='a_bar_numerator')

    # <float_dtype>[m]
    a_bar_denominator = tf.einsum('ij->j', c, name='a_bar_denominator')

    # <float_dtype>[m]
    a_bar = tf.identity(a_bar_numerator / a_bar_denominator, name='a_bar')

    # <float_dtype>[1]
    squared_error = tf.reduce_sum(
        c * tf.math.pow(tf.expand_dims(a_bar, 0) - tf.expand_dims(p_hat, 1), 2),
        name='squared_error')

    # <float_dtype>[1]
    squared_error_scaled = squared_error / logits.shape[0]

    return squared_error_scaled


def get_soft_binning_ece_tensor(predictions, labels, soft_binning_bins,
                                soft_binning_use_decay,
                                soft_binning_decay_factor, soft_binning_temp):
  """Computes and returns the soft-binned ECE (binned) tensor.

  Soft-binned ECE (binned, l2-norm) is defined in equation (11) in this paper:
  https://arxiv.org/abs/2108.00106. It is a softened version of ECE (binned)
  which is defined in equation (6).

  Args:
    predictions: tensor of predicted confidences of (batch-size,) shape
    labels: tensor of incorrect(0)/correct(1) labels of (batch-size,) shape
    soft_binning_bins: number of bins
    soft_binning_use_decay: whether temp should be determined by decay factor
    soft_binning_decay_factor: approximate decay factor between successive bins
    soft_binning_temp: soft binning temperature

  Returns:
    A tensor of () shape containing a single value: the soft-binned ECE.
  """

  soft_binning_anchors = tf.convert_to_tensor(
      np.arange(1.0 / (2.0 * soft_binning_bins), 1.0, 1.0 / soft_binning_bins),
      dtype=float)

  predictions_tile = tf.tile(
      tf.expand_dims(predictions, 1), [1, tf.shape(soft_binning_anchors)[0]])
  predictions_tile = tf.expand_dims(predictions_tile, 2)
  bin_anchors_tile = tf.tile(
      tf.expand_dims(soft_binning_anchors, 0), [tf.shape(predictions)[0], 1])
  bin_anchors_tile = tf.expand_dims(bin_anchors_tile, 2)

  if soft_binning_use_decay:
    soft_binning_temp = 1 / (
        math.log(soft_binning_decay_factor) * soft_binning_bins *
        soft_binning_bins)

  predictions_bin_anchors_product = tf.concat(
      [predictions_tile, bin_anchors_tile], axis=2)
  # pylint: disable=g-long-lambda
  predictions_bin_anchors_differences = tf.math.reduce_sum(
      tf.scan(
          fn=lambda _, row: tf.scan(
              fn=lambda _, x: tf.convert_to_tensor(
                  [-((x[0] - x[1])**2) / soft_binning_temp, 0.]),
              elems=row,
              initializer=0 * tf.ones(predictions_bin_anchors_product.shape[2:])
          ),
          elems=predictions_bin_anchors_product,
          initializer=tf.zeros(predictions_bin_anchors_product.shape[1:])),
      axis=2,
  )
  # pylint: enable=g-long-lambda
  predictions_soft_binning_coeffs = tf.nn.softmax(
      predictions_bin_anchors_differences,
      axis=1,
  )

  sum_coeffs_for_bin = tf.reduce_sum(predictions_soft_binning_coeffs, axis=[0])

  intermediate_predictions_reshaped_tensor = tf.reshape(
      tf.repeat(predictions, soft_binning_anchors.shape),
      predictions_soft_binning_coeffs.shape)
  net_bin_confidence = tf.divide(
      tf.reduce_sum(
          tf.multiply(intermediate_predictions_reshaped_tensor,
                      predictions_soft_binning_coeffs),
          axis=[0]),
      tf.maximum(sum_coeffs_for_bin, EPS * tf.ones(sum_coeffs_for_bin.shape)))

  intermediate_labels_reshaped_tensor = tf.reshape(
      tf.repeat(labels, soft_binning_anchors.shape),
      predictions_soft_binning_coeffs.shape)
  net_bin_accuracy = tf.divide(
      tf.reduce_sum(
          tf.multiply(intermediate_labels_reshaped_tensor,
                      predictions_soft_binning_coeffs),
          axis=[0]),
      tf.maximum(sum_coeffs_for_bin, EPS * tf.ones(sum_coeffs_for_bin.shape)))

  bin_weights = tf.linalg.normalize(sum_coeffs_for_bin, ord=1)[0]
  soft_binning_ece = tf.sqrt(
      tf.tensordot(
          tf.square(tf.subtract(net_bin_confidence, net_bin_accuracy)),
          bin_weights,
          axes=1,
      ))

  return soft_binning_ece

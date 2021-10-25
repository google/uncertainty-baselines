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
    logits: tensor of predicted logits of (batch-size, num-classes) shape
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


def get_avuc_loss(probabilities, labels, avuc_stop_prob_gradients,
                  avuc_entropy_threshold):
  """Computes and returns the AvUC loss tensor.

  AvUC loss is defined in equation (3) in this paper:
  https://arxiv.org/pdf/2012.07923.pdf. AvUC-with-gradient-stopping is defined
  in the appendix of this paper: https://arxiv.org/pdf/2108.00106.pdf.

  Args:
    probabilities: tensor of predicted probabilities of
      (batch-size, num-classes) shape
    labels: tensor of labels in [0,num-classes) of (batch-size,) shape
    avuc_stop_prob_gradients: whether to use gradient-stopping
    avuc_entropy_threshold: entropy threshold (u_th in equation (3) cited above)

  Returns:
    A tensor of () shape containing a single value: the AvUC loss.
  """

  if avuc_stop_prob_gradients:
    confidences = tf.stop_gradient(tf.reduce_max(probabilities, axis=1))
  else:
    confidences = tf.reduce_max(probabilities, axis=1)

  accuracies = tf.dtypes.cast(
      tf.math.equal(
          tf.argmax(probabilities, axis=1),
          tf.cast(labels, tf.int64),
      ),
      tf.float32,
  )

  dim = int(probabilities.shape[1])
  uniform_probabilities = tf.convert_to_tensor([1.0 / dim] * dim)
  log_safe_probabilities = ((1.0 - EPS) * probabilities +
                            EPS * uniform_probabilities)
  log_probabilities = tf.math.log(log_safe_probabilities)
  entropies = tf.math.negative(
      tf.reduce_sum(
          tf.multiply(log_safe_probabilities, log_probabilities), axis=1))

  accuracies_entropies_and_confidences = tf.stack(
      [accuracies, entropies, confidences], axis=-1)

  ac_clause = (lambda aec: aec[1] < avuc_entropy_threshold and aec[0] > 0.5)
  au_clause = (lambda aec: aec[1] >= avuc_entropy_threshold and aec[0] > 0.5)
  ic_clause = (lambda aec: aec[1] < avuc_entropy_threshold and aec[0] < 0.5)
  iu_clause = (lambda aec: aec[1] >= avuc_entropy_threshold and aec[0] < 0.5)

  # pylint: disable=g-long-lambda
  nac_diff = tf.reduce_sum(
      tf.map_fn(
          elems=accuracies_entropies_and_confidences,
          fn=lambda aec: tf.convert_to_tensor(aec[2]) *
          (tf.constant(1.0) - tf.math.tanh(tf.convert_to_tensor(aec[1])))
          if ac_clause(aec) else 0.0))
  nau_diff = tf.reduce_sum(
      tf.map_fn(
          elems=accuracies_entropies_and_confidences,
          fn=lambda aec: tf.convert_to_tensor(aec[2]) * tf.math.tanh(
              tf.convert_to_tensor(aec[1])) if au_clause(aec) else 0.0))
  nic_diff = tf.reduce_sum(
      tf.map_fn(
          elems=accuracies_entropies_and_confidences,
          fn=lambda aec: (tf.constant(1.0) - tf.convert_to_tensor(aec[2])) *
          (tf.constant(1.0) - tf.math.tanh(tf.convert_to_tensor(aec[1])))
          if ic_clause(aec) else 0.0))
  niu_diff = tf.reduce_sum(
      tf.map_fn(
          elems=accuracies_entropies_and_confidences,
          fn=lambda aec:
          (tf.constant(1.0) - tf.convert_to_tensor(aec[2])) * tf.math.tanh(
              tf.convert_to_tensor(aec[1])) if iu_clause(aec) else 0.0))
  # pylint: enable=g-long-lambda

  avuc_loss = tf.math.log(
      tf.constant(1.0) +
      ((nau_diff + nic_diff) /
       tf.math.maximum(nac_diff + niu_diff, tf.constant(EPS))))

  return avuc_loss


def get_soft_avuc_loss(probabilities, labels, soft_avuc_use_deprecated_v0,
                       soft_avuc_temp, soft_avuc_theta):
  """Computes and returns the soft AvUC loss tensor.

  Soft AvUC loss is defined in equation (15) in this paper:
  https://arxiv.org/pdf/2108.00106.pdf.

  Args:
    probabilities: tensor of predicted probabilities of
      (batch-size, num-classes) shape
    labels: tensor of labels in [0,num-classes) of (batch-size,) shape
    soft_avuc_use_deprecated_v0: whether to use a deprecated formulation
    soft_avuc_temp: temperature > 0 (T in equation (15) cited above)
    soft_avuc_theta: threshold in (0,1) (kappa in equation (15) cited above)

  Returns:
    A tensor of () shape containing a single value: the soft AvUC loss.
  """

  accuracies = tf.dtypes.cast(
      tf.math.equal(
          tf.argmax(probabilities, axis=1),
          tf.cast(labels, tf.int64),
      ),
      tf.float32,
  )

  dim = int(probabilities.shape[1])
  uniform_probabilities = tf.convert_to_tensor([1.0 / dim] * dim)
  log_safe_probabilities = (1.0 -
                            EPS) * probabilities + EPS * uniform_probabilities
  log_probabilities = tf.math.log(log_safe_probabilities)
  entropies = tf.math.negative(
      tf.reduce_sum(
          tf.multiply(log_safe_probabilities, log_probabilities), axis=1))

  entmax = math.log(dim)

  # pylint: disable=g-long-lambda
  def soft_uncertainty(e, temp=1, theta=0.5):
    return tf.math.sigmoid(
        (1 / temp) * tf.math.log(e * (1 - theta) / ((1 - e) * theta)))

  if soft_avuc_use_deprecated_v0:
    xus = tf.map_fn(
        elems=entropies,
        fn=lambda ent: -((ent - entmax)**2),
    )
    xcs = tf.map_fn(
        elems=entropies,
        fn=lambda ent: -(ent**2),
    )
    qucs = tf.nn.softmax(tf.stack([xus, xcs], axis=1), axis=1)
    qus = tf.squeeze(tf.slice(qucs, [0, 0], [-1, 1]))
    qcs = tf.squeeze(tf.slice(qucs, [0, 1], [-1, 1]))
  else:
    qus = tf.map_fn(
        elems=entropies,
        fn=lambda ent: soft_uncertainty(
            ent / entmax, temp=soft_avuc_temp, theta=soft_avuc_theta),
    )
    qcs = tf.map_fn(
        elems=qus,
        fn=lambda qu: 1 - qu,
    )
  # pylint: enable=g-long-lambda

  accuracies_entropies_and_qucs = tf.stack([accuracies, entropies, qus, qcs],
                                           axis=1)

  # pylint: disable=g-long-lambda
  nac_diff = tf.reduce_sum(
      tf.map_fn(
          elems=accuracies_entropies_and_qucs,
          fn=lambda e: tf.convert_to_tensor(e[3]) *
          (tf.constant(1.0) - tf.math.tanh(tf.convert_to_tensor(e[1])))
          if e[0] > 0.5 else 0.0))
  nau_diff = tf.reduce_sum(
      tf.map_fn(
          elems=accuracies_entropies_and_qucs,
          fn=lambda e: tf.convert_to_tensor(e[2]) * tf.math.tanh(
              tf.convert_to_tensor(e[1])) if e[0] > 0.5 else 0.0))
  nic_diff = tf.reduce_sum(
      tf.map_fn(
          elems=accuracies_entropies_and_qucs,
          fn=lambda e: tf.convert_to_tensor(e[3]) *
          (tf.constant(1.0) - tf.math.tanh(tf.convert_to_tensor(e[1])))
          if e[0] < 0.5 else 0.0))
  niu_diff = tf.reduce_sum(
      tf.map_fn(
          elems=accuracies_entropies_and_qucs,
          fn=lambda e: tf.convert_to_tensor(e[2]) * tf.math.tanh(
              tf.convert_to_tensor(e[1])) if e[0] < 0.5 else 0.0))
  # pylint: enable=g-long-lambda

  avuc_loss = tf.math.log(
      tf.constant(1.0) + (nau_diff + nic_diff) /
      tf.math.maximum(nac_diff + niu_diff, tf.constant(EPS)))

  return avuc_loss

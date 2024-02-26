# coding=utf-8
# Copyright 2024 The Uncertainty Baselines Authors.
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

"""Metrics calculators in tensorflow."""
from typing import Callable, Tuple

import tensorflow as tf


# Maps logits=[a, b, c], labels=[b, 1] to float metric.
# Where a=num_enn_samples, b=batch_size, c=num_classes.
MetricsCalculator = Callable[[tf.Tensor, tf.Tensor], float]


def reshape_to_smaller_batches(
    logits: tf.Tensor,
    labels: tf.Tensor,
    batch_size: int,
) -> Tuple[tf.Tensor, tf.Tensor]:
  """Reshapes logits,labels to add leading batch_size dimension.

  Args:
    logits: has shape [num_enn_samples, num_data, num_classes]
    labels: has shape [num_data, 1]
    batch_size: desired output batch size.

  Returns:
    A tuple of batched_logits and batched_labels with shapes
      batched_logits: (num_batches, num_enn_samples, batch_size, num_classes)
      batched_labels: (num_batches, batch_size, 1)
  """
  num_enn_samples, unused_num_data, num_classes = logits.shape.as_list()

  # 1. Split num_data to batches of size batch_size
  batched_logits = tf.reshape(
      logits, [num_enn_samples, -1, batch_size, num_classes])
  batched_labels = tf.reshape(labels, [-1, batch_size, 1])

  # 2. We want num_batches to be the leading axis. It is already the case for
  # batched_labels, but we need to change axes for batched_logits.
  batched_logits = tf.transpose(batched_logits, [1, 0, 2, 3])
  tf.ensure_shape(batched_logits,
                  [None, num_enn_samples, batch_size, num_classes])

  return batched_logits, batched_labels


def categorical_log_likelihood(
    probs: tf.Tensor, labels: tf.Tensor) -> tf.Tensor:
  """Computes joint log likelihood based on probs and labels."""
  int_labels = tf.cast(labels, tf.int32)
  indexer = lambda x: tf.gather(x[0], x[1])
  assigned_probs = tf.vectorized_map(indexer, (probs, int_labels))
  return tf.math.log(assigned_probs)


def safe_average(x: tf.Tensor) -> float:
  max_val = tf.reduce_max(x)
  return tf.math.log(tf.reduce_mean(tf.exp(x - max_val))) + max_val


def make_nll_polyadic_calculator(num_classes: int,
                                 tau: int = 10,
                                 kappa: int = 2) -> MetricsCalculator:
  """Returns a MetricCalculator that computes d_{KL}^{tau, kappa} metric."""

  def joint_ll(inputs: Tuple[tf.Tensor, tf.Tensor, tf.Tensor]) -> float:
    """Calculates joint NLL tau inputs resampled from anchor points."""
    # Shape checking
    logits, labels, selected = inputs
    tf.ensure_shape(logits, [kappa, num_classes])
    tf.ensure_shape(labels, [kappa, 1])
    tf.ensure_shape(selected, [tau])

    # Compute log-likehood for each anchor point
    probs = tf.nn.softmax(logits)
    lls = categorical_log_likelihood(probs, labels)

    # Resampling randomly from anchor points.
    return tf.reduce_sum(tf.gather(lls, selected))

  def enn_nll(inputs: Tuple[tf.Tensor, tf.Tensor]) -> float:
    """Averages NLL over multiple ENN samples."""
    # Shape checking
    logits, labels = inputs
    tf.ensure_shape(logits, [None, kappa, num_classes])
    tf.ensure_shape(labels, [kappa, 1])

    # Batching labels and seeds by duplication
    num_enn_samples = tf.shape(logits)[0]
    batched_labels = tf.repeat(labels[None], num_enn_samples, axis=0)

    # Random allocation of anchor points for this batch
    selected = tf.random.uniform([tau], maxval=kappa, dtype=tf.int32)
    batched_selected = tf.repeat(selected[None], num_enn_samples, axis=0)

    # Vectorizing then averaging over ENN samples
    lls = tf.vectorized_map(
        joint_ll, (logits, batched_labels, batched_selected))
    return -1 * safe_average(lls)

  def polyadic_nll(logits: tf.Tensor, labels: tf.Tensor) -> float:
    """Returns polyadic NLL based on repeated inputs.

    Internally this function works by taking the batch of logits and then
    "melting" it to add an extra dimension so that the batches we evaluate
    likelihood are of size=kappa. This means that one batch_size=N*kappa becomes
    N batches of size=kappa. For each of these batches of size kappa, we then
    resample tau observations replacement from these two anchor points. The
    function then returns the joint nll evaluated over this synthetic batch.

    Args:
      logits: [num_enn_samples, batch_size, num_classes]
      labels: [batch_size, 1]
    """
    # Shape checking
    tf.ensure_shape(logits, [None, None, num_classes])
    tf.ensure_shape(labels, [logits.shape[1], 1])

    # Creating synthetic batches of size=kappa then use vmap.
    batched_logits, batched_labels = reshape_to_smaller_batches(
        logits, labels, batch_size=kappa)

    # Forming random seeds for dyadic resampling per batch
    # TODO(smasghari): Cannot use tf.vectorized_map, conflict with tf.random.*
    nlls = tf.map_fn(enn_nll, (batched_logits, batched_labels),
                     fn_output_signature=tf.TensorSpec(shape=[],
                                                       dtype=tf.float32))
    return tf.reduce_mean(nlls)

  return polyadic_nll



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

"""Metrics calculators in tensorflow."""
from typing import Callable, Tuple

import tensorflow as tf


# Maps logits=[a, b, c], labels=[b, 1] to float metric.
# Where a=num_enn_samples, b=batch_size, c=num_classes.
MetricsCalculator = Callable[[tf.Tensor, tf.Tensor], float]


def reshape_to_smaller_batches_tf(
    logits: tf.Tensor,
    labels: tf.Tensor,
    batch_size: int,
) -> Tuple[tf.Tensor, tf.Tensor]:
  """Reshapes logits,labels to add leading batch_size dimension.

  In case the size of logits and labels are such that they cannot be equally
  divided into batches of size batch_size, extra data is discarded.

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


def categorical_log_likelihood(probs: tf.Tensor, labels: tf.Tensor) -> float:
  """Computes joint log likelihood based on probs and labels."""
  indexer = lambda x: tf.gather(x[0], x[1])
  assigned_probs = tf.vectorized_map(indexer,
                                     (probs, tf.cast(labels, tf.int32)))
  return tf.reduce_sum(tf.math.log(assigned_probs))


def safe_average(x: tf.Tensor) -> float:
  max_val = tf.reduce_max(x)
  return tf.math.log(tf.reduce_mean(tf.exp(x - max_val))) + max_val


def make_nll_polyadic_calculator(
    num_classes: int, tau: int = 10, kappa: int = 2) -> MetricsCalculator:
  """Returns a MetricCalculator that computes d_{KL}^{tau, kappa} metric."""
  assert tau % kappa == 0

  def joint_ll_repeated(logits_labels: Tuple[tf.Tensor, tf.Tensor]) -> float:
    """Calculates joint NLL evaluated on anchor points repeated tau / kappa."""
    # Shape checking
    logits, labels = logits_labels
    tf.ensure_shape(logits, [kappa, num_classes])
    tf.ensure_shape(labels, [kappa, 1])

    # Compute log-likehood, and then multiply by tau / kappa repeats.
    probs = tf.nn.softmax(logits)
    ll = categorical_log_likelihood(probs, labels)
    num_repeat = tau / kappa
    return ll * num_repeat

  def enn_nll(logits_labels: Tuple[tf.Tensor, tf.Tensor]) -> float:
    """Averages NLL over multiple ENN samples."""
    # Shape checking
    logits, labels = logits_labels
    tf.ensure_shape(logits, [None, kappa, num_classes])
    tf.ensure_shape(labels, [kappa, 1])
    batched_labels = tf.repeat(labels[None], tf.shape(logits)[0], axis=0)

    # Averaging over ENN samples
    lls = tf.vectorized_map(joint_ll_repeated, (logits, batched_labels))
    return -1 * safe_average(lls)

  def polyadic_nll(logits: tf.Tensor, labels: tf.Tensor) -> float:
    # Shape checking
    tf.ensure_shape(labels, [logits.shape[1], 1])

    # Creating synthetic batches of size=kappa then use vmap.
    batched_logits, batched_labels = reshape_to_smaller_batches_tf(
        logits, labels, batch_size=kappa)
    nlls = tf.vectorized_map(enn_nll, (batched_logits, batched_labels))
    return tf.reduce_mean(nlls)

  return polyadic_nll

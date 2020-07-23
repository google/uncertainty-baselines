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
"""Utilities related to custom losses for one_vs_all classifiers."""

import tensorflow.compat.v2 as tf


def get(
    loss_name: str,
    from_logits: bool = True,
    dm_alpha: float = 1.0):
  """Returns a loss function for training a tf.keras.Model.

  Args:
    loss_name: the name of the loss_function to use.
    from_logits: bool indicating whether model outputs logits or normalized
      probabilities.
    dm_alpha: float indicating the value of the alpha term to use for
      distance-based loss functions.
  """
  loss_name = loss_name.lower()
  if loss_name == 'crossentropy':
    return tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=from_logits, reduction=tf.keras.losses.Reduction.SUM)
  elif loss_name == 'dm_loss':
    return dm_loss_fn(dm_alpha=dm_alpha, from_logits=from_logits)
  elif loss_name in ['one_vs_all', 'one_vs_all_dm']:
    return one_vs_all_loss_fn(dm_alpha=dm_alpha, from_logits=from_logits)
  else:
    raise ValueError('Unrecognized loss name: {}'.format(loss_name))


def get_normalized_probabilities(logits: tf.Tensor, loss_name: str):
  loss_name = loss_name.lower()
  if loss_name in ['crossentropy', 'dm_loss']:
    return tf.nn.softmax(logits, axis=-1)
  elif loss_name == 'one_vs_all':
    return tf.nn.sigmoid(logits)
  elif loss_name == 'one_vs_all_dm':
    return 2. * tf.nn.sigmoid(logits)
  else:
    raise ValueError('Unrecognized loss name: {}'.format(loss_name))


def dm_loss_fn(dm_alpha: float = 10., from_logits: bool = True):
  """Requires from_logits=True to calculate correctly."""
  if not from_logits:
    raise ValueError('Distinction Maximization loss requires inputs to the '
                     'loss function to be logits, not probabilities.')
  def dm_loss(labels: tf.Tensor, logits: tf.Tensor):
    """Implements the distinction maximization loss function.

    As implemented in https://arxiv.org/abs/1908.05569, multiplies the output
    logits by dm_alpha before taking the softmax and calculating cross_entropy
    loss. The prediction output of DM Loss does not have the alpha factor.
    Args:
      labels: Integer Tensor of dense labels, shape [batch_size].
      logits: Tensor of shape [batch_size, num_classes].
    Returns:
      Either binary_crossentropy or SparseCategoricalCrossentropy depending
        on whether binary or multiclass classification is being performed.
    """
    # For the loss function, multiply the logits by alpha before crossentropy.
    logits *= dm_alpha
    return tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction=tf.keras.losses.Reduction.SUM)(
            labels, logits)
  return dm_loss


def one_vs_all_loss_fn(dm_alpha: float = 1., from_logits: bool = True):
  """Requires from_logits=True to calculate correctly."""
  if not from_logits:
    raise ValueError('One-vs-all loss requires inputs to the '
                     'loss function to be logits, not probabilities.')

  def one_vs_all_loss(labels: tf.Tensor, logits: tf.Tensor):
    r"""Implements the one-vs-all loss function.

    As implemented in https://arxiv.org/abs/1709.08716, multiplies the output
    logits by dm_alpha (if using a distance-based formulation) before taking K
    independent sigmoid operations of each class logit, and then calculating the
    sum of the log-loss across classes. The loss function is calculated from the
    K sigmoided logits as follows -

    \mathcal{L} = \sum_{i=1}^{K} -\mathbb{I}(y = i) \log p(\hat{y}^{(i)} | x)
    -\mathbb{I} (y \neq i) \log (1 - p(\hat{y}^{(i)} | x))

    Args:
      labels: Integer Tensor of dense labels, shape [batch_size].
      logits: Tensor of shape [batch_size, num_classes].

    Returns:
      A scalar containing the mean over the batch for one-vs-all loss.
    """
    eps = 1e-6
    logits = logits * dm_alpha
    n_classes = tf.cast(logits.shape[1], tf.float32)

    one_vs_all_probs = tf.math.sigmoid(logits)
    labels = tf.cast(tf.squeeze(labels), tf.int32)
    row_ids = tf.range(tf.shape(one_vs_all_probs)[0], dtype=tf.int32)
    idx = tf.stack([row_ids, labels], axis=1)

    # Shape of class_probs is [batch_size,].
    class_probs = tf.gather_nd(one_vs_all_probs, idx)

    loss = (
        tf.reduce_mean(tf.math.log(class_probs + eps)) +
        n_classes * tf.reduce_mean(tf.math.log(1. - one_vs_all_probs + eps)) -
        tf.reduce_mean(tf.math.log(1. - class_probs + eps)))

    return -loss

  return one_vs_all_loss


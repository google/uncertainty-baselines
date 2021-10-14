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

"""Loss utils, including rebalancing based on class distribution."""

from typing import Dict, Union

import tensorflow as tf
import tensorflow.keras.backend as K
import torch


def get_diabetic_retinopathy_class_balance_weights(
  positive_empirical_prob: float = None
) -> Dict[int, float]:
  r"""Class weights used for rebalancing the dataset, by skewing the `loss`.

  Diabetic Retinopathy positive class proportions are imbalanced:
    Train: 19.6%
    Val: 18.8%
    Test: 19.2%

  Here, we compute appropriate class weights such that the following
  loss reweighting can be done multiplicatively for each element.

  \mathcal{L}= -\frac{1}{K n} \sum_{i=1}^{n}
  \frac{\mathcal{L}_{\text{cross-entropy}}}{p(k)}
  where we have K = 2 classes, n images in a minibatch, and the p(k) is the
  empirical probability of class k in the training dataset.

  Therefore, we here compute weights
    w_k = \frac{1}{K} * \frac{1}{p(k)}
  in order to apply the reweighting with an elementwise multiply over the
  batch losses.

  We can also use the empirical probabilities for a particular minibatch,
  i.e. p(k)_{\text{minibatch}}.

  Args:
    positive_empirical_prob: the empirical probability of a positive label.

  Returns:
    Reweighted positive and negative example probabilities.
  """

  if positive_empirical_prob is None:
    raise NotImplementedError(
      'Needs to be updated for APTOS / Severity shifts, '
      'different decision thresholds (Mild / Moderate classifiers).')
    positive_empirical_prob = 0.196

  return {
      0: (1 / 2) * (1 / (1 - positive_empirical_prob)),
      1: (1 / 2) * (1 / positive_empirical_prob)
  }


def get_positive_empirical_prob(labels: tf.Tensor) -> float:
  """
  Given a set of binary labels, determine the empirical probability of a
  positive label (i.e., the proportion of ones).

  Args:
    labels: tf.Tensor, batch of labels

  Returns:
    empirical probability of a positive label
  """
  n_pos_labels = tf.math.count_nonzero(labels)
  total_n_labels = labels.get_shape()[0]
  return n_pos_labels / total_n_labels


def get_weighted_binary_cross_entropy_keras(weights: Dict[int, float]):
  """Return a function to calculate weighted binary xent with multi-hot labels.

  Due to @menrfa
  (https://stackoverflow.com/questions/46009619/
    keras-weighted-binary-crossentropy)

  # Example
  >>> y_true = tf.convert_to_tensor([1, 0, 0, 0, 0, 0], dtype=tf.int64)
  >>> y_pred = tf.convert_to_tensor(
  ...            [0.6, 0.1, 0.1, 0.9, 0.1, 0.], dtype=tf.float32)
  >>> weights = {
  ...     0: 1.,
  ...     1: 2.
  ... }

  # With weights
  >>> loss_fn = get_weighted_binary_cross_entropy_keras(weights=weights)
  >>> loss_fn(y_true, y_pred)
  <tf.Tensor(0.6067193, shape=(), dtype=tf.float32)>

  # Without weights
  >>> loss_fn = tf.keras.losses.binary_crossentropy
  >>> loss_fn(y_true, y_pred)
  <tf.Tensor(0.52158177, shape=(), dtype=tf.float32)>

  # Another example
  >>> y_true = tf.convert_to_tensor([[0., 1.], [0., 0.]], dtype=tf.float32)
  >>> y_pred = tf.convert_to_tensor([[0.6, 0.4], [0.4, 0.6]], dtype=tf.float32)
  >>> weights = {
  ...     0: 1.,
  ...     1: 2.
  ... }

  # With weights
  >>> loss_fn = get_weighted_binary_cross_entropy_keras(weights=weights,
  from_logits=False)
  >>> loss_fn(y_true, y_pred)
  <tf.Tensor: shape=(2,), dtype=float32, numpy=array([1.3744358 , 0.71355796],
  dtype=float32)>

  # Without weights
  >>> loss_fn = tf.keras.losses.binary_crossentropy
  >>> loss_fn(y_true, y_pred)
  <tf.Tensor: shape=(2,), dtype=float32, numpy=array([0.9162905 , 0.71355796],
  dtype=float32)>

  Args:
    weights: dict, set weights for respective labels, e.g., {
          0: 1.
          1: 8. } In this case, we aim to compensate for the true (1) label
            occurring less in the training dataset than the false (0) label.
            e.g. [0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0]

  Returns:
    A function to calculate (weighted) binary cross entropy.
  """
  if 0 not in weights or 1 not in weights:
    raise NotImplementedError

  def weighted_cross_entropy_fn(y_true, y_pred, from_logits=False):
    tf_y_true = tf.cast(y_true, dtype=y_pred.dtype)
    tf_y_pred = tf.cast(y_pred, dtype=y_pred.dtype)

    weight_1 = tf.cast(weights[1], dtype=y_pred.dtype)
    weight_0 = tf.cast(weights[0], dtype=y_pred.dtype)
    weights_v = tf.where(tf.equal(tf_y_true, 1), weight_1, weight_0)
    ce = K.binary_crossentropy(tf_y_true, tf_y_pred, from_logits=from_logits)
    loss = K.mean(tf.multiply(ce, weights_v), axis=-1)
    return loss

  return weighted_cross_entropy_fn


def get_weighted_binary_cross_entropy_torch(weights: Dict[int, float]):
  """Return a function to calculate weighted binary xent with multi-hot labels.

  Based on implementation from @menrfa
  (https://stackoverflow.com/questions/46009619/
    keras-weighted-binary-crossentropy)

  # Example
  >>> y_true = torch.FloatTensor([1, 0, 0, 0, 0, 0])
  >>> y_pred = torch.FloatTensor([0.6, 0.1, 0.1, 0.9, 0.1, 0.])
  >>> weights = {
  ...     0: 1.,
  ...     1: 2.
  ... }

  # With weights
  >>> loss_fn = get_weighted_binary_cross_entropy_torch(weights=weights)
  >>> loss_fn(y_true, y_pred)
  tensor(0.6067)

  # Without weights
  >>> loss_fn = torch.nn.BCELoss()
  >>> loss_fn(input=y_pred, target=y_true)
  tensor(0.5216)

  Args:
    weights: dict, set weights for respective labels, e.g., {
          0: 1.
          1: 8. } In this case, we aim to compensate for the true (1) label
            occurring less in the training dataset than the false (0) label.
            e.g. [0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0]

  Returns:
    A function to calculate (weighted) binary cross entropy.
  """
  if 0 not in weights or 1 not in weights:
    raise NotImplementedError

  def weighted_cross_entropy_fn(
      y_true: torch.Tensor, y_pred: torch.Tensor, from_logits: bool = False
  ):
    assert y_true.dtype == torch.float32
    assert y_pred.dtype == torch.float32

    # weight_1 = torch.as_tensor(weights[1], dtype=y_pred.dtype)
    # weight_0 = torch.as_tensor(weights[0], dtype=y_pred.dtype)
    weights_v = torch.where(y_true == 1, weights[1], weights[0])
    if from_logits:
      ce = torch.nn.BCEWithLogitsLoss(weight=weights_v, reduction='none')
    else:
      ce = torch.nn.BCELoss(weight=weights_v, reduction='none')

    return torch.mean(ce(input=y_pred, target=y_true))

  return weighted_cross_entropy_fn


def get_diabetic_retinopathy_loss_fn(class_reweight_mode: Union[str, None],
                                     class_weights: Union[Dict[int, float],
                                                          None]):
  """Initialize loss function based on class reweighting setting.

  Return None for a minibatch loss, which must be defined per-minibatch,
  using the minibatch empirical label distribution.
  Args:
    class_reweight_mode: Union[str, None], None indicates no class reweighting,
      `constant` indicates reweighting with the training set empirical
      distribution, `minibatch` indicates reweighting with the minibatch
      empirical label distribution.
    class_weights: Union[Dict[int, float], None], class weights as produced by
      `get_diabetic_retinopathy_class_balance_weights`, should only be provided
      for the `constant` class_reweight_mode.

  Returns:
    None, or loss_fn
  """
  #
  if class_reweight_mode is None:
    loss_fn = tf.keras.losses.binary_crossentropy
  elif class_reweight_mode == 'constant':
    raise NotImplementedError
    # Initialize a reweighted BCE using the empirical class distribution
    # of the training dataset.
    loss_fn = get_weighted_binary_cross_entropy(weights=class_weights)
  elif class_reweight_mode == 'minibatch':
    # This loss_fn must be reinitialized for each batch, using the
    # minibatch empirical class distribution.
    loss_fn = None
  else:
    raise NotImplementedError(
        f'Reweighting mode {class_reweight_mode} unsupported.')

  return loss_fn


def get_minibatch_reweighted_loss_fn(labels: tf.Tensor, loss_fn_type='keras'):
  """The minibatch-reweighted loss function can only be initialized
  using the labels of a particular minibatch.

  Args:
    labels: tf.Tensor, the labels of a minibatch
    loss_fn_type: str, one of {'keras', 'torch', 'jax'}
  Returns:
    loss_fn, for use in a particular minibatch
  """
  minibatch_positive_empirical_prob = get_positive_empirical_prob(labels=labels)
  minibatch_class_weights = (
      get_diabetic_retinopathy_class_balance_weights(
          positive_empirical_prob=minibatch_positive_empirical_prob))

  if loss_fn_type == 'keras':
    batch_loss_fn = get_weighted_binary_cross_entropy_keras(
        weights=minibatch_class_weights)
  elif loss_fn_type == 'torch':
    for key, value in minibatch_class_weights.items():
      minibatch_class_weights[key] = value._numpy()
    batch_loss_fn = get_weighted_binary_cross_entropy_torch(
        weights=minibatch_class_weights)
  elif loss_fn_type == 'jax':
    raise NotImplementedError
  else:
    raise NotImplementedError

  return batch_loss_fn

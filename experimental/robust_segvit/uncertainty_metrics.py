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

"""Calculate uncertainty metrics for segmentation tasks."""
from typing import Optional, Tuple
from jax import lax
import jax.numpy as jnp
from scenic.model_lib.layers import nn_ops


def calculate_num_patches_binary_maps(
    binary_acc_map: jnp.ndarray, binary_unc_map: jnp.ndarray
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
  """Calculate conditional probabilities in confusion matrix.

  Args:
    binary_acc_map : binary accuracy map
    binary_unc_map : binary uncertainty map (1=certain, 0=uncertain)

  Returns:
    metrics to calculate uncertainty scores
  """
  # number of patches that are accurate and certain
  n_ac = jnp.sum(
      jnp.logical_and(
          jnp.equal(binary_acc_map, 1), jnp.equal(binary_unc_map, 1)),
      axis=(-1, -2))

  # number of patches that are inaccurate and certain
  n_ic = jnp.sum(
      jnp.logical_and(
          jnp.equal(binary_acc_map, 0), jnp.equal(binary_unc_map, 1)),
      axis=(-1, -2))
  # number of patches that are inaccurate and uncertain
  n_iu = jnp.sum(
      jnp.logical_and(
          jnp.equal(binary_acc_map, 0), jnp.equal(binary_unc_map, 0)),
      axis=(-1, -2))

  # number of patches that are accurate and uncertain
  n_au = jnp.sum(
      jnp.logical_and(
          jnp.equal(binary_acc_map, 1), jnp.equal(binary_unc_map, 0)),
      axis=(-1, -2))

  unc_confusion_matrix = jnp.stack((n_ac, n_ic, n_iu, n_au), axis=-1)

  unc_confusion_matrix = unc_confusion_matrix[jnp.newaxis, ...]  # Dummy batch dim.
  return unc_confusion_matrix


def get_pacc_cert(unc_confusion_matrix):
  """Calculate p(accurate | certain)."""

  n_ac, n_ic, _, _ = jnp.split(unc_confusion_matrix, jnp.arange(1, 4), axis=-1)

  return jnp.nan_to_num(n_ac / (n_ac + n_ic))


def get_puncert_inacc(unc_confusion_matrix):
  """Calculate p(uncertain | inaccurate)."""
  _, n_ic, n_iu, _ = jnp.split(unc_confusion_matrix, jnp.arange(1, 4), axis=-1)

  return jnp.nan_to_num(n_iu / (n_ic + n_iu))


def get_pavpu(unc_confusion_matrix):
  """Patch accuracy vs Patch uncertainty."""
  n_ac, n_ic, n_iu, n_au = jnp.split(
      unc_confusion_matrix, jnp.arange(1, 4), axis=-1)

  return jnp.nan_to_num((n_ac + n_iu) / (n_ac + n_au + n_ic + n_iu))


def get_uncertainty_confusion_matrix(
    logits: jnp.ndarray,
    labels: jnp.ndarray,
    uncertainty_measure: str = 'softmax',
    accuracy_measure : str = 'predictive_accuracy',
    weights: Optional[jnp.ndarray] = None,
    accuracy_th: Optional[float] = 0.5,
    uncertainty_th: Optional[float] = 0.5,
    window_size: Optional[int] = 2,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
  """Calculate counts of patches accurate/inacurate and certain/uncertain.

  Args:
    logits: predicted logits
    labels: true labels
    weights: weights which mask classes to ignore.
    accuracy_th: accuracy threshold.
    uncertainty_th: uncertainty threshold.
    window_size: size of window to split data.

  Returns:
    unc_confusion_matrix: array with counts of patches accurate/inaccurate and
    certain/uncertain.
  """
  # ---
  if labels.ndim == logits.ndim:  # One-hot targets.
    targets = jnp.argmax(labels, axis=-1)
  else:
    targets = labels

  preds = jnp.argmax(logits, axis=-1)

  # calculate binary accuracy map
  correct = jnp.equal(preds, targets).astype(jnp.float32)

  if weights is None:
    weights = jnp.ones(correct.shape)

  weights = weights.astype(jnp.float32)

  if accuracy_measure == 'predictive_accuracy':
    accuracy_map = correct
  else:
    raise NotImplementedError('Accuracy measure not implemented.')

  # A given patch is accurate if its acc > accuracy_threshold
  binary_acc_map = reduce_2dmap_weighted(accuracy_map,
                                         weights,
                                         window_size=window_size,
                                          threshold=accuracy_th).astype(jnp.float32)

  # Calculate uncertainty map:
  if uncertainty_measure == 'softmax':
    uncertainty_map = jnp.max(jnp.exp(logits) / jnp.sum(jnp.exp(logits), -1, keepdims=True), -1)
  elif uncertainty_measure == 'entropy':
    uncertainty_map = get_entropy_from_logits(logits)
  else:
    raise NotImplementedError(f'Uncertainty measure {uncertainty_measure} not implemented.')

  # A given patch is certain if its uncertainty > uncertainty_th
  binary_unc_map = reduce_2dmap_weighted(uncertainty_map,
                                         weights,
                                         window_size=window_size,
                                          threshold=uncertainty_th).astype(jnp.float32)

  # number of patches that are accurate and certain
  unc_confusion_matrix = calculate_num_patches_binary_maps(
      binary_acc_map, binary_unc_map)

  return unc_confusion_matrix


def get_entropy_from_logits(logits: jnp.ndarray) -> jnp.ndarray:
  # Calculate uncertainty map
  probs = jnp.exp(logits) / jnp.sum(jnp.exp(logits), -1, keepdims=True)
  entropy = -jnp.sum(probs * jnp.log(probs), axis=-1).astype(jnp.float32)
  return entropy


def reduce_2dmap(
    array_map: jnp.ndarray,
    window_size: int = 4,
    threshold: float = 0.5,
) -> jnp.ndarray:
  """Given a map, apply a 2d spatial strided convolution to avg adjacent values.

  Args:
    array_map: array to be split.  3-D Tensor;  With shape `[batch, in_rows, in_cols].
    window_size: size of window.
    threshold: threshold for binarization.

  Returns:
    binary_map: binary map.
  """
  # Expand dimension for dummy depth dimension 
  array_map = jnp.expand_dims(array_map, -1)

  # Create a kernel
  kernel = jnp.ones([window_size, window_size, 1, 1])

  dn = lax.conv_dimension_numbers(array_map.shape, kernel.shape,
                                  ('NHWC', 'HWIO', 'NHWC'))

  # Convolve map with kernel
  out = lax.conv_general_dilated(
      array_map,
      kernel, (window_size, window_size),
      'SAME',
      dimension_numbers=dn)

  # divide by window_size
  out = jnp.divide(out, window_size * window_size)

  # binarize_map according to threshold
  binary_map = jnp.greater(out, threshold)

  binary_map = jnp.squeeze(binary_map, -1)

  return binary_map.astype(jnp.int32)


def reduce_2dmap_weighted(
    array_map: jnp.ndarray,
    weights: jnp.ndarray,
    window_size: int = 4,
    threshold: float = 0.5,
) -> jnp.ndarray:
  """Given a map, apply a pooling operation to avg adjacent values.

  Args:
    array_map: array to be split.  3-D Tensor;  With shape `[batch, in_rows, in_cols].
    weights: array of weights.   3-D Tensor;  With shape `[batch, in_rows, in_cols].
    window_size: size of window.
    threshold: threshold for binarization.
    data_format: str; The format of the `lhs`. Must be either `'NHWC'` or `'NCHW'`.

  Returns:
    binary_map: binary map.
  """
  # Expand dimension for dummy feature dimension
  array_map = jnp.expand_dims(array_map, -1)

  window_shape = (window_size, window_size)

  outputs = nn_ops.weighted_avg_pool(
    array_map,
    weights,
    window_shape=window_shape,
    strides=window_shape,
    padding='VALID')

  # Binarize_map according to threshold
  binary_map = jnp.greater_equal(outputs, threshold)

  # Squeeze dummy feature dimension
  binary_map = jnp.squeeze(binary_map, -1)

  return binary_map.astype(jnp.int32)


class SegmentationUncertaintyMetrics(object):
  """Calculate uncertainty scores for image segmentation task."""

  def __init__(self,
               logits,
               labels,
               weights=None,
               window_size=4,
               accuracy_th=0.5,
               uncertainty_th=0.5):

    self.logits = logits
    self.labels = labels
    self.weights = weights
    self.window_size = window_size
    self.accuracy_th = accuracy_th
    self.uncertainty_th = uncertainty_th

  @property
  def unc_confusion_matrix(self):
    """Calculate uncertainty confusion matrix."""
    return get_uncertainty_confusion_matrix(
        logits=self.logits,
        labels=self.labels,
        weights=self.weights,
        accuracy_th=self.accuracy_th,
        uncertainty_th=self.uncertainty_th,
        window_size=self.window_size)

  @property
  def pacc_cert(self):
    return get_pacc_cert(self.unc_confusion_matrix)

  @property
  def puncert_inacc(self):
    return get_puncert_inacc(self.unc_confusion_matrix)

  @property
  def pavpu(self):
    return get_pavpu(self.unc_confusion_matrix)

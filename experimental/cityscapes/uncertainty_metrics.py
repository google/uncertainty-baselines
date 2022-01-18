"""
Include uncertainty metrics
"""
import jax.numpy as jnp
from typing import Optional, Any, Tuple, Union

import numpy as np

from scenic.model_lib.base_models.model_utils import apply_weights

from jax import lax

# TODO(kellybuchanan): consolidate metric calculation as class


def calculate_num_patches_binary_maps(
  binary_acc_map: jnp.ndarray,
  binary_unc_map: jnp.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
  """
  Calculate conditional probabilities in confusion matrix given binary
  accuracy and uncertainty maps
  """
  # number of patches that are accurate and certain
  n_ac = jnp.sum(jnp.logical_and(jnp.equal(binary_acc_map, 1),
                                 jnp.equal(binary_unc_map, 0)), axis=(-1, -2))

  # number of patches that are inaccurate and certain
  n_ic = jnp.sum(jnp.logical_and(jnp.equal(binary_acc_map, 0),
                                 jnp.equal(binary_unc_map, 0)), axis=(-1, -2)
                 )
  # number of patches that are inaccurate and uncertain
  n_iu = jnp.sum(jnp.logical_and(jnp.equal(binary_acc_map, 0),
                                 jnp.equal(binary_unc_map, 1)), axis=(-1, -2)
                 )

  # number of patches that are accurate and uncertain
  n_au = jnp.sum(jnp.logical_and(jnp.equal(binary_acc_map, 1),
                                 jnp.equal(binary_unc_map, 1)), axis=(-1, -2)
                 )

  return n_ac, n_ic, n_iu, n_au


def calculate_uncertainty_confusion_matrix(
  logits: jnp.ndarray,
  labels: jnp.ndarray,
  weights: Optional[jnp.ndarray] = None,
  accuracy_th: Optional[float] = 0.5,
  uncertainty_th: Optional[float] = 0.5,
  window_size: Optional[int] = 2) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
  """
  Calculate conditional probabilities
  TODO(kellybuchanan): include weights for entropy calculation
  """
  # ---
  if labels.ndim == logits.ndim:  # One-hot targets.
    targets = jnp.argmax(labels, axis=-1)
  else:
    targets = labels

  preds = jnp.argmax(logits, axis=-1)

  # calculate binary accuracy map
  correct = jnp.equal(preds, targets)

  # batch masking
  if weights is not None:
    correct = apply_weights(correct, weights)

  correct = correct.astype(jnp.float32)

  # A given patch is accurate if its acc > accuracy_threshold
  binary_acc_map = reduce_2dmap(correct, window_size, accuracy_th).astype(jnp.float32)

  # Calculate uncertainty map
  probs = jnp.exp(logits) / jnp.sum(jnp.exp(logits), -1, keepdims=True)
  entropy = -jnp.sum(probs*jnp.log(probs), axis=-1).astype(jnp.float32)

  # A given patch is uncertain if its uncertainty > uncertainty_th
  binary_unc_map = reduce_2dmap(entropy, window_size, uncertainty_th).astype(jnp.float32)

  # number of patches that are accurate and certain
  n_ac, n_ic, n_iu, n_au = calculate_num_patches_binary_maps(
      binary_acc_map, binary_unc_map)

  return n_ac, n_ic, n_iu, n_au


def calculate_puncert_inacc(
    logits: jnp.ndarray,
    labels: jnp.ndarray,
    weights: Optional[jnp.ndarray] = None,
    accuracy_th: Optional[float] = 0.5,
    uncertainty_th: Optional[float] = 0.4,
    window_size: Optional[int] = 2) -> jnp.ndarray:
  """
  Calculate p(uncertain | inaccurate)
  """

  n_ac, n_ic, n_iu, n_au = calculate_uncertainty_confusion_matrix(
      logits=logits,
      labels=labels,
      weights=weights,
      accuracy_th=accuracy_th,
      uncertainty_th=uncertainty_th,
      window_size=window_size)

  p_uncertain_inaccurate = n_iu / (n_ic + n_iu)

  return p_uncertain_inaccurate


def calculate_pacc_cert(
    logits: jnp.ndarray,
    labels: jnp.ndarray,
    weights: Optional[jnp.ndarray] = None,
    accuracy_th: Optional[float] = 0.5,
    uncertainty_th: Optional[float] = 0.4,
    window_size: Optional[int] = 2) -> jnp.ndarray:
  """
  Calculate p(accurate|certain)
  """
  # TODO(kellybuchanan): reconcile cases where there are no certain patches.

  n_ac, n_ic, n_iu, n_au = calculate_uncertainty_confusion_matrix(
      logits=logits,
      labels=labels,
      weights=weights,
      accuracy_th=accuracy_th,
      uncertainty_th=uncertainty_th,
      window_size=window_size)

  p_accurate_certain = n_ac / (n_ac + n_ic)
  return p_accurate_certain


def calculate_pavpu(
    logits: jnp.ndarray,
    labels: jnp.ndarray,
    weights: Optional[jnp.ndarray] = None,
    accuracy_th: Optional[float] = 0.5,
    uncertainty_th: Optional[float] = 0.4,
    window_size: Optional[int] = 2) -> jnp.ndarray:
  """
  Calculate PavPu
  """
  n_ac, n_ic, n_iu, n_au = calculate_uncertainty_confusion_matrix(
      logits=logits,
      labels=labels,
      weights=weights,
      accuracy_th=accuracy_th,
      uncertainty_th=uncertainty_th,
      window_size=window_size)

  # Patch accuracy vs Patch uncertainty
  pavpu = (n_ac + n_iu) / (n_ac + n_au + n_ic + n_iu)

  return pavpu


def reduce_2dmap(
  array_map: jnp.ndarray,
  window_size: int = 4,
  threshold: float = 0.5,
  ) -> jnp.ndarray:
  """
  Given a map, apply a 2d spatial strided convolution to avg adjacent values
  """
  reduce_dims = 0

  # Expand dims if necessary
  if array_map.ndim == 3:
    array_map = jnp.expand_dims(array_map, 0)
    reduce_dims = 1

  # Create a kernel
  kernel = jnp.ones(array_map.shape[:-2] + (window_size, window_size))

  # Convolve map with kernel
  out = lax.conv(array_map,    # lhs = NCHW image tensor
                 kernel,  # rhs = OIHW conv kernel tensor
                 (window_size, window_size),  # window strides
                 'SAME')  # padding mode

  # divide by window_size
  out = jnp.divide(out, window_size*window_size)

  # binarize_map according to threshold
  binary_map = jnp.greater(out, threshold)

  if reduce_dims:
    binary_map = jnp.squeeze(binary_map, 0)

  return binary_map.astype(jnp.int32)

"""
Include uncertainty metrics
"""
import jax.numpy as jnp
from typing import Optional, Any, Tuple, Union
from scenic.model_lib.base_models.model_utils import apply_weights

from jax import lax


def calculate_pavpu(
    labels: jnp.ndarray,
    logits: jnp.ndarray,
    weights: Optional[jnp.ndarray] = None,
    accuracy_th: Optional[float] = 0.5,
    uncertainty_th: Optional[float] = 0.5,
    window_size: Optional[int] = 2) -> jnp.ndarray:
  """
  Calculate PavPu
  """
  if labels.ndim == logits.ndim:  # One-hot targets.
    targets = jnp.argmax(labels, axis=-1)
  else:
    targets = labels

  preds = jnp.argmax(logits, axis=-1)

  # calculate binary accuracy map
  correct = jnp.equal(preds, targets)

  if weights is not None:
    correct = apply_weights(correct, weights)

  correct = correct.astype(jnp.float32)

  binary_acc_map = binarize_map(correct,window_size,accuracy_th)

  # calculate uncertainty map
  entropy = jnp.sum(logits*jnp.log(logits), axis=-1).astype(jnp.float32)

  binary_unc_map = binarize_map(entropy, window_size, uncertainty_th)

  # umber of patches that are accurate and certain
  n_ac = jnp.sum(jnp.logical_and(binary_acc_map, binary_unc_map))

  # number of patches that are inaccurate and certain
  n_ic = jnp.sum(jnp.logical_and(jnp.equal(binary_acc_map, 0),
                                 jnp.equal(binary_unc_map, 1))
                 )
  # number of patches that are inaccurate and uncertain
  n_iu = jnp.sum(jnp.logical_and(jnp.equal(binary_acc_map, 0),
                                 jnp.equal(binary_unc_map, 0))
                 )

  # number of patches that are accurate and uncertain
  n_au = jnp.sum(jnp.logical_and(jnp.equal(binary_acc_map, 1),
                                 jnp.equal(binary_unc_map, 0))
                 )

  # p_accurate_certain = n_ac / (n_ac + n_ic)
  # p_uncertain_inaccurate = n_iu / (n_ic + n_iu)

  # Patch accuracy vs Patch uncertainty
  pavpu = (n_ac + n_iu) / (n_ac + n_au + n_ic + n_iu)

  return pavpu


def binarize_map(
  array_map: jnp.ndarray,
  window_size: Optional[int] = 2,
  threshold:Optional[float] = 0.5,
  ) -> jnp.ndarray:
  """
  Given a map, apply a 2d spatial strided convolution to avg adjacent values
  """
  # expand dims if necessary
  if array_map.ndim == 3:
    array_map = jnp.expand_dims(array_map, 0)

  # create a kernel
  kernel = jnp.ones(array_map.shape[:-2] + (window_size, window_size))

  # Convolve map with kernel
  out = lax.conv(array_map,    # lhs = NCHW image tensor
                 kernel, # rhs = OIHW conv kernel tensor
                 (window_size, window_size),  # window strides
                 'SAME')  # padding mode

  # divide by window_size
  out = jnp.divide(out, window_size*window_size)

  # binarize_map according to threshold
  binary_map = jnp.greater_equal(out, threshold)

  return binary_map.astype(jnp.int32)


import os
import random as random_py

import jax
import numpy as np
import tensorflow as tf
import torch
import tree
from jax import jit
from jax import numpy as jnp

from baselines.diabetic_retinopathy_detection.fsvi_utils.jax_utils import KeyHelper


def initialize_random_keys(seed: int) -> KeyHelper:
  os.environ["PYTHONHASHSEED"] = str(seed)
  rng_key = jax.random.PRNGKey(seed)
  kh = KeyHelper(key=rng_key)
  random_py.seed(seed)
  np.random.seed(seed)
  tf.random.set_seed(seed)
  torch.random.manual_seed(seed)
  return kh


def to_one_hot(x, k):
  """Create a one-hot encoding of x of size k."""
  return jnp.array(x[:, None] == jnp.arange(k), jnp.float32)


@jit
def sigma_transform(params_log_var):
  return tree.map_structure(lambda p: jnp.exp(p), params_log_var)


@jit
def kl_divergence(
  mean_q, mean_p, cov_q, cov_p,
):
  """
  Return KL(q || p)

  All inputs are either of shape (batch_dim, output_dim).
  """
  function_kl = 0
  output_dim = mean_q.shape[1]
  for i in range(output_dim):
    mean_q_tp = mean_q[:, i]
    cov_q_tp = cov_q[:, i]
    mean_p_tp = mean_p[:, i]
    cov_p_tp = cov_p[:, i]
    function_kl += kl_diag(mean_q_tp, mean_p_tp, cov_q_tp, cov_p_tp, )
  return function_kl


@jit
def kl_diag(mean_q, mean_p, cov_q, cov_p) -> jnp.ndarray:
  """
  Return KL(q || p)
  NOte: all inputs are 1D arrays.

  @param cov_q: the diagonal of covariance
  @return:
      a scalar
  """
  kl_1 = jnp.log(cov_p ** 0.5) - jnp.log(cov_q ** 0.5)
  kl_2 = (cov_q + (mean_q - mean_p) ** 2) / (2 * cov_p)
  kl_3 = -1 / 2
  kl = jnp.sum(kl_1 + kl_2 + kl_3)
  return kl

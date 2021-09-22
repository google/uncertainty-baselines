from typing import Tuple, Callable

import haiku as hk
from jax import numpy as jnp

from baselines.diabetic_retinopathy_detection.fsvi_utils import utils
from baselines.diabetic_retinopathy_detection.fsvi_utils.haiku_mod import (
  map_variable_name,
)
from baselines.diabetic_retinopathy_detection.fsvi_utils.ntk_utils import explicit_ntk


def bnn_linearized_predictive(
  apply_fn: Callable,
  params_mean: hk.Params,
  params_log_var: hk.Params,
  params_deterministic: hk.Params,
  state: hk.State,
  inducing_inputs: jnp.ndarray,
  rng_key: jnp.ndarray,
  stochastic_linearization: bool,
  full_ntk: bool,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
  """
  Return the mean and covariance of output of linearized BNN

  Currently this function is used in the following places
      - in the definition of loss function when model type is "fsvi"
      - in the definition of bnn induced prior function
  Basically, whenever we need to calculate the function distribution of a BNN from its parameter
  distribution, we can use this function.

  @param stochastic_linearization: if True, linearize around sampled parameter; otherwise linearize around mean
      parameters.

  @return
      mean: array of shape (batch_dim, output_dim)
      cov: array of shape
          if full_ntk is True, then (batch_dim, output_dim, batch_dim, output_dim)
          otherwise, (batch_dim, output_dim)
  """
  params = hk.data_structures.merge(params_mean, params_log_var, params_deterministic)
  mean = apply_fn(
    params,
    state,
    None,
    inducing_inputs,
    rng_key,
    stochastic=stochastic_linearization,
    is_training=True,
  )[0]

  params_var = utils.sigma_transform(params_log_var)

  predict_fn_for_empirical_ntk = convert_predict_f_only_mean(
    apply_fn,
    inducing_inputs,
    params_log_var,
    params_deterministic,
    state,
    rng_key,
    stochastic_linearization,
  )
  renamed_params_var = map_variable_name(
    params_var, lambda n: f"{n.split('_')[0]}_mu"
  )
  # surprisingly, if I jit this function, there will be memory issue
  cov = explicit_ntk(
    fwd_fn=predict_fn_for_empirical_ntk,
    params=params_mean,
    sigma=renamed_params_var,
    diag=not full_ntk,
  )

  return mean, cov


def convert_predict_f_only_mean(
  apply_fn,
  inputs,
  params_log_var,
  params_batchnorm,
  state,
  rng_key,
  stochastic_linearization,
):
  def predict_f_only_mean(params_mean):
    params = hk.data_structures.merge(params_mean, params_log_var, params_batchnorm)
    return apply_fn(
      params,
      state,
      None,
      inputs,
      rng_key,
      stochastic=stochastic_linearization,
      is_training=True,
    )[0]

  return predict_f_only_mean

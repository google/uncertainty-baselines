from typing import Tuple, Callable

import haiku as hk
import jax
import tree
from jax import numpy as jnp

from baselines.diabetic_retinopathy_detection.fsvi_utils import utils


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


def explicit_ntk(
  fwd_fn: Callable, params: hk.Params, sigma: hk.Params, diag=False,
) -> jnp.ndarray:
  """
  Calculate J * diag(sigma) * J^T, where J is Jacobian of model with respect to model parameters
   using explicit implementation and einsum

  @param fwd_fn: a function that only takes in parameters and returns model output of shape (batch_dim, output_dim)
  @param params: the model parameters
  @param sigma: it has the same structure and array shapes as the parameters of model
  @param diag: if True, only calculating the diagonal of NTK
  @return:
      diag_ntk_sum_array: array of shape (batch_dim, output_dim) if diag==True else
       (batch_dim, output_dim, batch_dim, output_dim)
  """
  jacobian = jax.jacobian(fwd_fn)(params)

  def _get_diag_ntk(jac, sigma):
    # jac has shape (batch_dim, output_dim, params_dims...)
    # jac_2D has shape (batch_dim * output_dim, nb_params)
    batch_dim, output_dim = jac.shape[:2]
    jac_2D = jnp.reshape(jac, (batch_dim * output_dim, -1))
    # sigma_flatten has shape (nb_params,) and will be broadcasted to the same shape as jac_2D
    sigma_flatten = jnp.reshape(sigma, (-1,))
    # jac_sigma_product has the same shape as jac_2D
    jac_sigma_product = jnp.multiply(jac_2D, sigma_flatten)
    # diag_ntk has shape (batch_dim * output_dim,)
    if diag:
      ntk = jnp.einsum("ij,ji->i", jac_sigma_product, jac_2D.T)
      ntk = jnp.reshape(ntk, (batch_dim, output_dim))
      # ntk has shape (batch_dim, output_dim)
    else:
      ntk = jnp.matmul(jac_sigma_product, jac_2D.T)
      ntk = jnp.reshape(ntk, (batch_dim, output_dim, batch_dim, output_dim))
      # ntk has shape (batch_dim, output_dim, batch_dim, output_dim)
    return ntk

  diag_ntk = tree.map_structure(_get_diag_ntk, jacobian, sigma)
  diag_ntk_sum_array = jnp.stack(tree.flatten(diag_ntk), axis=0).sum(axis=0)
  return diag_ntk_sum_array


def map_variable_name(params: hk.Params, fn: Callable) -> hk.Params:
  params = hk.data_structures.to_mutable_dict(params)
  for module in params:
    params[module] = {
      fn(var_name): array for var_name, array in params[module].items()
    }
  return hk.data_structures.to_immutable_dict(params)
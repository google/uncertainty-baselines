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

"""FSVI linearization utils."""
# pylint: disable=g-bare-generic
# pylint: disable=g-doc-args
# pylint: disable=g-doc-return-or-yield
# pylint: disable=g-importing-member
# pylint: disable=g-no-space-after-docstring-summary
# pylint: disable=g-short-docstring-punctuation
# pylint: disable=logging-format-interpolation
# pylint: disable=logging-fstring-interpolation
# pylint: disable=missing-function-docstring
from typing import Callable, Tuple
from baselines.diabetic_retinopathy_detection.fsvi_utils import utils
import haiku as hk
import jax
from jax import numpy as jnp
import tree


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
  """Return the mean and covariance of output of linearized BNN on inducing inputs.

  Args:
    apply_fn: forward pass function of model.
    params_mean: variational mean.
    params_log_var: log variational variance.
    params_deterministic: other parameters like the one used in batch
      normalization.
    state: state of model, e.g. mean and std used in batch normalization.
    inducing_inputs: inducing inputs on which this function aims to produce an
      approximate posterior distribution.
    rng_key: jax random key.
    stochastic_linearization: if True, linearize model around a sampled
      parameter instead of mean parameter.
    full_ntk: if True, evaluate the full covariance, otherwise, only the
      diagonal.

  Returns:
    jnp.ndarray, array of shape (batch_dim, output_dim)
    jnp.ndarray, array of the following shape
        if full_ntk is True, then (batch_dim, output_dim, batch_dim, output_dim)
        otherwise, (batch_dim, output_dim)
  """
  params = hk.data_structures.merge(params_mean, params_log_var,
                                    params_deterministic)
  mean = apply_fn(
      params,
      state,
      None,
      inducing_inputs,
      rng_key,
      stochastic=stochastic_linearization,
      is_training=True,
  )[0]

  params_var = utils.exp_params(params_log_var)

  predict_fn_for_empirical_ntk = convert_predict_f_only_mean(
      apply_fn,
      inducing_inputs,
      params_log_var,
      params_deterministic,
      state,
      rng_key,
      stochastic_linearization,
  )
  renamed_params_var = rename_params(params_var,
                                     lambda n: f"{n.split('_')[0]}_mu")
  cov = explicit_ntk(
      fwd_fn=predict_fn_for_empirical_ntk,
      params=params_mean,
      sigma=renamed_params_var,
      diag=not full_ntk,
  )

  return mean, cov


def convert_predict_f_only_mean(
    apply_fn: Callable,
    inputs: jnp.ndarray,
    params_log_var: hk.Params,
    params_batchnorm: hk.Params,
    state: hk.State,
    rng_key: jnp.ndarray,
    stochastic_linearization: bool,
) -> Callable:
  """Return a function that takes the variational mean and returns the output"""

  def predict_f_only_mean(params_mean):
    params = hk.data_structures.merge(params_mean, params_log_var,
                                      params_batchnorm)
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
    fwd_fn: Callable,
    params: hk.Params,
    sigma: hk.Params,
    diag=False,
) -> jnp.ndarray:
  """Calculate J * diag(sigma) * J^T, where J is Jacobian of model with respect to model parameters

   using explicit implementation and einsum

  Args:
    fwd_fn: a function that only takes in parameters and returns model output of
      shape (batch_dim, output_dim).
    params: the model parameters.
    sigma: it has the same structure and array shapes as the parameters of
      model.
    diag: if True, only calculating the diagonal of NTK.

  Returns:
    jnp.ndarray, array of shape (batch_dim, output_dim) if diag==True else
       (batch_dim, output_dim, batch_dim, output_dim)
  """
  jacobian = jax.jacobian(fwd_fn)(params)

  def _get_diag_ntk(jac, sigma):
    # jac has shape (batch_dim, output_dim, params_dims...)
    # jac_2d has shape (batch_dim * output_dim, nb_params)
    batch_dim, output_dim = jac.shape[:2]
    jac_2d = jnp.reshape(jac, (batch_dim * output_dim, -1))
    # sigma_flatten has shape (nb_params,) and will be broadcasted to the same
    # shape as jac_2d
    sigma_flatten = jnp.reshape(sigma, (-1,))
    # jac_sigma_product has the same shape as jac_2d
    jac_sigma_product = jnp.multiply(jac_2d, sigma_flatten)
    # diag_ntk has shape (batch_dim * output_dim,)
    if diag:
      ntk = jnp.einsum("ij,ji->i", jac_sigma_product, jac_2d.T)
      ntk = jnp.reshape(ntk, (batch_dim, output_dim))
      # ntk has shape (batch_dim, output_dim)
    else:
      ntk = jnp.matmul(jac_sigma_product, jac_2d.T)
      ntk = jnp.reshape(ntk, (batch_dim, output_dim, batch_dim, output_dim))
      # ntk has shape (batch_dim, output_dim, batch_dim, output_dim)
    return ntk

  diag_ntk = tree.map_structure(_get_diag_ntk, jacobian, sigma)
  diag_ntk_sum_array = jnp.stack(tree.flatten(diag_ntk), axis=0).sum(axis=0)
  return diag_ntk_sum_array


def rename_params(params: hk.Params, fn: Callable[[str], str]) -> hk.Params:
  """Rename variables in params according to `fn`

  Args:
    params: parameters to rename
    fn: a renaming function that takes in old name and return new name.

  Returns:
    parameters with renamed variables.
  """
  params = hk.data_structures.to_mutable_dict(params)
  for module in params:
    params[module] = {
        fn(var_name): array for var_name, array in params[module].items()
    }
  return hk.data_structures.to_immutable_dict(params)

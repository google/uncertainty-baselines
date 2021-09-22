from typing import Callable

import haiku as hk
import jax
import tree
from jax import numpy as jnp


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

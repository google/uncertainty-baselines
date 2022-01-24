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

"""FSVI objectives."""
# pylint: disable=g-bare-generic
# pylint: disable=g-doc-args
# pylint: disable=g-doc-return-or-yield
# pylint: disable=g-importing-member
# pylint: disable=g-no-space-after-docstring-summary
# pylint: disable=g-short-docstring-punctuation
# pylint: disable=logging-format-interpolation
# pylint: disable=logging-fstring-interpolation
# pylint: disable=missing-function-docstring
from functools import partial
from typing import Dict, Tuple
from baselines.diabetic_retinopathy_detection.fsvi_utils import utils
from baselines.diabetic_retinopathy_detection.fsvi_utils import utils_linearization
from baselines.diabetic_retinopathy_detection.fsvi_utils.networks import Model
from baselines.diabetic_retinopathy_detection.utils import (
    get_diabetic_retinopathy_class_balance_weights,)
import haiku as hk
import jax
from jax import jit
import jax.numpy as jnp
import tree


class Loss:
  """Loss."""

  def __init__(
      self,
      model: Model,
      kl_scale: str,
      n_samples: int,
      stochastic_linearization: bool,
  ):
    """Args:

      model: wrapper of ResNet50FSVI
      kl_scale: the type of scale of kl, e.g. "equal", "normalized".
      n_samples: the number of Monte-Carlo samples used to estimate the
      posterior.
      stochastic_linearization: if True, linearize around a sampled parameter
      instead of around mean parameters.
    """
    self.model = model
    self.kl_scale = kl_scale
    self.n_samples = n_samples
    self.stochastic_linearization = stochastic_linearization

  def _crossentropy_log_likelihood(self, preds_f_samples: jnp.ndarray,
                                   targets: jnp.ndarray) -> jnp.ndarray:
    log_likelihood = jnp.mean(
        jnp.sum(
            jnp.sum(
                targets * jax.nn.log_softmax(preds_f_samples, axis=-1),
                axis=-1),
            axis=-1,
        ),
        axis=0,
    )
    return log_likelihood

  def _crossentropy_log_likelihood_with_class_weights(
      self, preds_f_samples: jnp.ndarray, targets: jnp.ndarray) -> jnp.ndarray:
    minibatch_positive_empirical_prob = targets[:, 1].sum() / targets.shape[0]
    minibatch_class_weights = get_diabetic_retinopathy_class_balance_weights(
        positive_empirical_prob=minibatch_positive_empirical_prob)

    log_likelihoods = jnp.mean(
        jnp.sum(
            targets * jax.nn.log_softmax(preds_f_samples, axis=-1), axis=-1),
        axis=0,
    )
    weights = jnp.where(targets[:, 1] == 1, minibatch_class_weights[1],
                        minibatch_class_weights[0])
    reduced_value = jnp.sum(jnp.multiply(log_likelihoods, weights))
    return reduced_value

  def _function_kl(
      self,
      params: hk.Params,
      state: hk.State,
      prior_mean: jnp.ndarray,
      prior_cov: jnp.ndarray,
      inputs: jnp.ndarray,
      inducing_inputs: jnp.ndarray,
      rng_key: jnp.ndarray,
  ) -> Tuple[jnp.ndarray, float]:
    """Evaluate the multi-output KL between the function distribution obtained by linearising BNN around

    params, and the prior function distribution represented by (`prior_mean`,
    `prior_cov`)

    Args:
      inputs: used for computing scale, only the shape is used
      inducing_inputs: used for computing scale and function distribution used
        in KL

    Returns:
        kl: scalar value of function KL
        scale: scale to multiple KL with
    """
    params_mean, params_log_var, params_deterministic = partition_params(params)
    scale = compute_scale(self.kl_scale, inputs, inducing_inputs.shape[0])

    mean, cov = utils_linearization.bnn_linearized_predictive(
        self.model.apply_fn,
        params_mean,
        params_log_var,
        params_deterministic,
        state,
        inducing_inputs,
        rng_key,
        self.stochastic_linearization,
        full_ntk=False,
    )

    kl = utils.kl_divergence_multi_output(
        mean,
        prior_mean,
        cov,
        prior_cov,
    )

    return kl, scale

  def _elbo_fsvi_classification(
      self,
      params: hk.Params,
      state: hk.State,
      prior_mean: jnp.ndarray,
      prior_cov: jnp.ndarray,
      inputs: jnp.ndarray,
      targets: jnp.ndarray,
      inducing_inputs: jnp.ndarray,
      rng_key: jnp.ndarray,
      is_training: bool,
      class_weight: bool,
      loss_type: int,
      l2_strength: float,
  ):
    preds_f_samples, _, _ = self.model.predict_f_multisample_jitted(
        params,
        state,
        inputs,
        rng_key,
        self.n_samples,
        is_training,
    )
    kl, scale = self.function_kl(
        params,
        state,
        prior_mean,
        prior_cov,
        inputs,
        inducing_inputs,
        rng_key,
    )

    log_likelihood = self.crossentropy_log_likelihood(preds_f_samples, targets,
                                                      class_weight)
    if loss_type == 1:
      elbo = log_likelihood - scale * kl
    elif loss_type == 2:
      elbo = (
          log_likelihood / inputs.shape[0] -
          scale * kl / inducing_inputs.shape[0])
    elif loss_type == 3:
      elbo = (log_likelihood - scale * kl) / inputs.shape[0]
    elif loss_type == 4:
      elbo = log_likelihood / inputs.shape[0]
    elif loss_type == 5:
      batch_norm_params = hk.data_structures.filter(predicate_batchnorm, params)
      l2_loss = jnp.sum(
          jnp.stack([jnp.sum(x * x) for x in tree.flatten(batch_norm_params)]))
      elbo = (log_likelihood -
              scale * kl) / inputs.shape[0] - l2_loss * l2_strength
    else:
      raise NotImplementedError(loss_type)

    return elbo, log_likelihood, kl, scale

  @partial(jit, static_argnums=(0, 3))
  def crossentropy_log_likelihood(self, preds_f_samples: jnp.ndarray,
                                  targets: jnp.ndarray,
                                  class_weight: bool) -> jnp.ndarray:
    if class_weight:
      return self._crossentropy_log_likelihood_with_class_weights(
          preds_f_samples, targets)
    else:
      return self._crossentropy_log_likelihood(preds_f_samples, targets)

  @partial(jit, static_argnums=(0,))
  def function_kl(
      self,
      params,
      state,
      prior_mean,
      prior_cov,
      inputs,
      inducing_inputs,
      rng_key,
  ):
    return self._function_kl(
        params=params,
        state=state,
        prior_mean=prior_mean,
        prior_cov=prior_cov,
        inputs=inputs,
        inducing_inputs=inducing_inputs,
        rng_key=rng_key,
    )

  @partial(jit, static_argnums=(0, 9, 10, 11))
  def nelbo_fsvi_classification(
      self,
      params: hk.Params,
      state: hk.State,
      prior_mean: jnp.ndarray,
      prior_cov: jnp.ndarray,
      inputs: jnp.ndarray,
      targets: jnp.ndarray,
      inducing_inputs: jnp.ndarray,
      rng_key: jnp.ndarray,
      class_weight: bool,
      loss_type: int,
      l2_strength: float,
  ) -> Tuple[float, Dict]:
    """Computes the ELBO objective for FSVI method.

    http://timrudner.com/papers/Rethinking_Function-Space_Variational_Inference_in_Bayesian_Neural_Networks/Rudner2021_Rethinking_Function-Space_Variational_Inference_in_Bayesian_Neural_Networks.pdf

    Args:
      params: parameters of model.
      state: state of model, e.g. mean and std of batch normalization layers.
      prior_mean: prior mean of model output.
      prior_cov: prior covariance of model output.
      inputs: inputs to the model.
      targets: labels.
      inducing_inputs: inducing inputs on which to compute the posterior
        distribution.
      rng_key: jax random key.
      class_weight: if True, address class imbalance by using different weights
        for each class in the log likelihood term.
      loss_type: select the exact option for the loss to optimise.
      l2_strength: strength of L2 regularisation.

    Returns:
      scalar jnp.ndarray, ELBO
      dict, a dictionary containing the constituents of the ELBO loss for
      debugging
    """
    is_training = True
    elbo, log_likelihood, kl, scale = self._elbo_fsvi_classification(
        params,
        state,
        prior_mean,
        prior_cov,
        inputs,
        targets,
        inducing_inputs,
        rng_key,
        is_training,
        class_weight,
        loss_type,
        l2_strength,
    )

    state = self.model.apply_fn(
        params,
        state,
        rng_key,
        inputs,
        rng_key,
        stochastic=True,
        is_training=is_training,
    )[1]

    return (
        -elbo,
        {
            "state": state,
            "elbo": elbo,
            "log_likelihood": log_likelihood,
            "kl": kl,
            "scale": scale,
            "loss": -elbo,
        },
    )


@partial(jit, static_argnums=(0,))
def compute_scale(kl_scale: str, inputs: jnp.ndarray,
                  n_inducing_inputs: int) -> float:
  if kl_scale == "none":
    scale = 1.0
  elif kl_scale == "equal":
    scale = inputs.shape[0] / n_inducing_inputs
  elif kl_scale == "normalized":
    scale = 1.0 / n_inducing_inputs
  else:
    scale = jnp.float32(kl_scale)
  return scale


@jit
def partition_params(
    params: hk.Params) -> Tuple[hk.Params, hk.Params, hk.Params]:
  params_log_var, params_rest = hk.data_structures.partition(
      predicate_var, params)

  def predicate_is_mu_with_log_var(module_name, name, value):
    logvar_name = f"{name.split('_')[0]}_logvar"
    return (predicate_mean(module_name, name, value) and
            module_name in params_log_var and
            logvar_name in params_log_var[module_name])

  params_mean, params_deterministic = hk.data_structures.partition(
      predicate_is_mu_with_log_var, params_rest)
  return params_mean, params_log_var, params_deterministic


def predicate_mean(module_name, name, value):
  del module_name
  del value
  return name == "w_mu" or name == "b_mu"


def predicate_var(module_name, name, value):
  del module_name
  del value
  return name == "w_logvar" or name == "b_logvar"


def predicate_batchnorm(module_name, name, value):
  del module_name
  del value
  return name not in {
      "w_mu",
      "b_mu",
      "w_logvar",
      "b_logvar",
  }

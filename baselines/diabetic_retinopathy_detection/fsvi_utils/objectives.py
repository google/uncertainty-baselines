import pdb
from functools import partial
from typing import Tuple

import haiku as hk
import jax
import jax.numpy as jnp
import tree
from jax import jit

from baselines.diabetic_retinopathy_detection.fsvi_utils import utils
from baselines.diabetic_retinopathy_detection.fsvi_utils import utils_linearization
from baselines.diabetic_retinopathy_detection.fsvi_utils.networks import Model
from baselines.diabetic_retinopathy_detection.fsvi_utils.haiku_mod import (
  partition_params,
  predicate_batchnorm,
)
from baselines.diabetic_retinopathy_detection.utils import (
  get_diabetic_retinopathy_class_balance_weights,
)


class Loss:
  def __init__(
    self, model: Model, kl_scale: str, n_samples, stochastic_linearization,
  ):
    self.model = model
    self.kl_scale = kl_scale
    self.n_samples = n_samples
    self.stochastic_linearization = stochastic_linearization

  def _crossentropy_log_likelihood(self, preds_f_samples, targets):
    log_likelihood = jnp.mean(
      jnp.sum(
        jnp.sum(
          targets * jax.nn.log_softmax(preds_f_samples, axis=-1), axis=-1
        ),
        axis=-1,
      ),
      axis=0,
    )
    return log_likelihood

  def _crossentropy_log_likelihood_with_class_weights(self, preds_f_samples, targets):
    # get_positive_empirical_prob
    # TODO: remove the hardcoded 1
    minibatch_positive_empirical_prob = targets[:, 1].sum() / targets.shape[0]
    minibatch_class_weights = get_diabetic_retinopathy_class_balance_weights(
      positive_empirical_prob=minibatch_positive_empirical_prob
    )

    log_likelihoods = jnp.mean(
      jnp.sum(targets * jax.nn.log_softmax(preds_f_samples, axis=-1), axis=-1),
      axis=0,
    )
    weights = jnp.where(
      targets[:, 1] == 1, minibatch_class_weights[1], minibatch_class_weights[0]
    )
    reduced_value = jnp.sum(jnp.multiply(log_likelihoods, weights))
    return reduced_value

  def _function_kl(
    self, params, state, prior_mean, prior_cov, inputs, inducing_inputs, rng_key,
  ) -> Tuple[jnp.ndarray, float]:
    """
    Evaluate the multi-output KL between the function distribution obtained by linearising BNN around
    params, and the prior function distribution represented by (`prior_mean`, `prior_cov`)

    @param inputs: used for computing scale, only the shape is used
    @param inducing_inputs: used for computing scale and function distribution used in KL

    @return:
        kl: scalar value of function KL
        scale: scale to multiple KL with
    """
    # TODO: Maybe change "params_deterministic" to "params_model"
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

    kl = utils.kl_divergence(mean, prior_mean, cov, prior_cov, )

    return kl, scale

  def _elbo_fsvi_classification(
    self,
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
  ):
    preds_f_samples, _, _ = self.model.predict_f_multisample_jitted(
      params, state, inputs, rng_key, self.n_samples, is_training,
    )
    kl, scale = self.function_kl(
      params, state, prior_mean, prior_cov, inputs, inducing_inputs, rng_key,
    )

    log_likelihood = self.crossentropy_log_likelihood(
      preds_f_samples, targets, class_weight
    )
    if loss_type == 1:
      elbo = log_likelihood - scale * kl
    elif loss_type == 2:
      elbo = (
        log_likelihood / inputs.shape[0] - scale * kl / inducing_inputs.shape[0]
      )
    elif loss_type == 3:
      elbo = (log_likelihood - scale * kl) / inputs.shape[0]
    elif loss_type == 4:
      elbo = log_likelihood / inputs.shape[0]
    elif loss_type == 5:
      batch_norm_params = hk.data_structures.filter(predicate_batchnorm, params)
      l2_loss = jnp.sum(
        jnp.stack([jnp.sum(x * x) for x in tree.flatten(batch_norm_params)])
      )
      elbo = (log_likelihood - scale * kl) / inputs.shape[
        0
      ] - l2_loss * l2_strength
    else:
      raise NotImplementedError(loss_type)

    return elbo, log_likelihood, kl, scale

  @partial(jit, static_argnums=(0, 3))
  def crossentropy_log_likelihood(self, preds_f_samples, targets, class_weight):
    if class_weight:
      return self._crossentropy_log_likelihood_with_class_weights(
        preds_f_samples, targets
      )
    else:
      return self._crossentropy_log_likelihood(preds_f_samples, targets)

  @partial(jit, static_argnums=(0,))
  def function_kl(
    self, params, state, prior_mean, prior_cov, inputs, inducing_inputs, rng_key,
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
    params,
    state,
    prior_mean,
    prior_cov,
    inputs,
    targets,
    inducing_inputs,
    rng_key,
    class_weight: bool,
    loss_type: int,
    l2_strength: float,
  ):
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
def compute_scale(kl_scale: str, inputs: jnp.ndarray, n_inducing_inputs: int) -> float:
  if kl_scale == "none":
    scale = 1.0
  elif kl_scale == "equal":
    scale = inputs.shape[0] / n_inducing_inputs
  elif kl_scale == "normalized":
    scale = 1.0 / n_inducing_inputs
  else:
    scale = jnp.float32(kl_scale)
  return scale

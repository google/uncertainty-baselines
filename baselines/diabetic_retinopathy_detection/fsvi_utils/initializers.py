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

"""FSVI initializers."""
# pylint: disable=g-bare-generic
# pylint: disable=g-doc-args
# pylint: disable=g-doc-return-or-yield
# pylint: disable=g-importing-member
# pylint: disable=g-no-space-after-docstring-summary
# pylint: disable=g-short-docstring-punctuation
# pylint: disable=logging-format-interpolation
# pylint: disable=logging-fstring-interpolation
# pylint: disable=missing-function-docstring
from typing import Any, Callable, List, Tuple
import haiku as hk
from jax import numpy as jnp
import optax
from .networks import CNN  # local file import from baselines.diabetic_retinopathy_detection.fsvi_utils.networks
from .networks import Model  # local file import from baselines.diabetic_retinopathy_detection.fsvi_utils.networks
from .objectives import Loss  # local file import from baselines.diabetic_retinopathy_detection.fsvi_utils.objectives

DEFAULT_NUM_EPOCHS = 90


class Initializer:
  """Initializer."""

  def __init__(
      self,
      activation: str,
      dropout_rate: float,
      input_shape: List[int],
      output_dim: int,
      kl_scale: str,
      stochastic_linearization: bool,
      n_samples: int,
      uniform_init_minval: float,
      uniform_init_maxval: float,
      init_strategy: str,
      prior_mean: str,
      prior_cov: str,
  ):
    """Constructor.

    Args:
      activation: activation function of ResNet50.
      dropout_rate: dropout rate.
      input_shape: input shape including batch dimension.
      output_dim: output dimension.
      kl_scale: the type of kl_scale, e.g. normalized, equal, etc.
      stochastic_linearization: if True, linearize around a sampled parameter
      instead of around mean parameters.
      n_samples: the number of Monte-Carlo samples for estimating the posterior.
      uniform_init_minval: lower bound of uniform distribution for log
      variational variance.
      uniform_init_maxval: upper bound of uniform distribution for log
      variational variance.
      init_strategy: the initialization strategy, e.g. "he_normal_and_zeros",
      "uniform".
      prior_mean: the prior mean value.
      prior_cov: the prior variance value.
    """
    self.activation = activation
    self.dropout_rate = dropout_rate
    self.input_shape = input_shape
    self.output_dim = output_dim
    self.kl_scale = kl_scale
    self.stochastic_linearization = stochastic_linearization
    self.n_samples = n_samples
    self.uniform_init_minval = uniform_init_minval
    self.uniform_init_maxval = uniform_init_maxval
    self.init_strategy = init_strategy

    self.prior_mean = prior_mean
    self.prior_cov = prior_cov

    if self.init_strategy == "he_normal_and_zeros":
      self.w_init = "he_normal"
      self.b_init = "zeros"
    elif self.init_strategy == "uniform":
      self.w_init = "uniform"
      self.b_init = "uniform"
    else:
      raise NotImplementedError(self.init_strategy)

    self.dropout = self.dropout_rate > 0
    print(
        f"Stochastic linearization (posterior): {self.stochastic_linearization}"
    )

  def initialize_model(
      self,
      rng_key: jnp.ndarray,
  ) -> Tuple[Model, Callable, hk.State, hk.Params]:
    model = self._compose_model()
    init_fn, apply_fn = model.forward
    x_init = jnp.ones(self.input_shape)
    params_init, state = init_fn(
        rng_key, x_init, rng_key, model.stochastic_parameters, is_training=True)
    return model, apply_fn, state, params_init

  def _compose_model(self) -> Model:
    model = CNN(
        output_dim=self.output_dim,
        activation_fn=self.activation,
        stochastic_parameters=True,
        linear_model=True,
        dropout=self.dropout_rate > 0,
        dropout_rate=self.dropout_rate,
        uniform_init_minval=self.uniform_init_minval,
        uniform_init_maxval=self.uniform_init_maxval,
        w_init=self.w_init,
        b_init=self.b_init,
    )
    return model

  def initialize_loss(self, model: Model) -> Callable:
    loss = Loss(
        model=model,
        kl_scale=self.kl_scale,
        n_samples=self.n_samples,
        stochastic_linearization=self.stochastic_linearization,
    )
    return loss.nelbo_fsvi_classification

  def initialize_prior(self,) -> Callable[[Tuple[Any, Any]], List[jnp.ndarray]]:
    f32_prior_mean, f32_prior_cov = (
        jnp.float32(self.prior_mean),
        jnp.float32(self.prior_cov),
    )

    def prior_fn(shape):
      prior_mean = jnp.ones(shape) * f32_prior_mean
      prior_cov = jnp.ones(shape) * f32_prior_cov
      return [prior_mean, prior_cov]

    return prior_fn


class OptimizerInitializer:
  """Optimizer initializer."""

  def __init__(
      self,
      optimizer: str,
      base_learning_rate: float,
      n_batches: int,
      epochs: int,
      one_minus_momentum: float,
      lr_warmup_epochs: int,
      lr_decay_ratio: float,
      lr_decay_epochs: List[int],
      final_decay_factor: float,
      lr_schedule: str,
  ):
    """Args:

      optimizer: the type of optimizer, e.g. "sgd", "adam".
      base_learning_rate: the base learning rate.
      n_batches: number of batches per epoch.
      epochs: number of training epochs.
      one_minus_momentum: momentum - 1 for sgd.
      lr_warmup_epochs: number of epochs for a linear warmup to the initial
      learning rate. Use 0 to do no warmup.
      lr_decay_ratio: amount to decay learning rate for sgd.
      lr_decay_epochs: epochs to decay learning rate by for sgd.
      final_decay_factor: how much to decay the LR by for sgd.
      lr_schedule: the type of learning rate schedule for sgd, e.g. "linear",
      "step".
    """
    self.optimizer = optimizer
    self.base_learning_rate = base_learning_rate
    self.n_batches = n_batches
    self.epochs = epochs
    self.one_minus_momentum = one_minus_momentum
    self.lr_warmup_epochs = lr_warmup_epochs
    self.lr_decay_ratio = lr_decay_ratio
    self.lr_decay_epochs = lr_decay_epochs
    self.final_decay_factor = final_decay_factor
    self.lr_schedule = lr_schedule

  def get(self) -> optax.GradientTransformation:
    if "adam" in self.optimizer:
      opt = optax.adam(self.base_learning_rate)
    elif "sgd" == self.optimizer and self.lr_schedule == "linear":
      lr_schedule = warm_up_polynomial_schedule(
          base_learning_rate=self.base_learning_rate,
          end_learning_rate=self.final_decay_factor * self.base_learning_rate,
          decay_steps=(self.n_batches * (self.epochs - self.lr_warmup_epochs)),
          warmup_steps=self.n_batches * self.lr_warmup_epochs,
          decay_power=1.0,
      )
      momentum = 1 - self.one_minus_momentum
      opt = optax.chain(
          optax.trace(decay=momentum, nesterov=True),
          optax.scale_by_schedule(lr_schedule),
          optax.scale(-1),
      )
    elif "sgd" in self.optimizer and self.lr_schedule == "step":
      lr_decay_epochs = [
          (int(start_epoch_str) * self.epochs) // DEFAULT_NUM_EPOCHS
          for start_epoch_str in self.lr_decay_epochs
      ]
      lr_schedule = warm_up_piecewise_constant_schedule(
          steps_per_epoch=self.n_batches,
          base_learning_rate=self.base_learning_rate,
          decay_ratio=self.lr_decay_ratio,
          decay_epochs=lr_decay_epochs,
          warmup_epochs=self.lr_warmup_epochs,
      )

      momentum = 1 - self.one_minus_momentum
      opt = optax.chain(
          optax.trace(decay=momentum, nesterov=True),
          optax.scale_by_schedule(lr_schedule),
          optax.scale(-1),
      )
    else:
      raise ValueError("No optimizer specified.")
    return opt


def warm_up_piecewise_constant_schedule(
    steps_per_epoch: int,
    base_learning_rate: float,
    warmup_epochs: int,
    decay_epochs: List[int],
    decay_ratio: float,
) -> Callable:
  """Please see uncertainty_baselines.schedules.WarmUpPiecewiseConstantSchedule.
  """

  def schedule(count):
    lr_epoch = jnp.array(count, jnp.float32) / steps_per_epoch
    learning_rate = base_learning_rate
    if warmup_epochs >= 1:
      learning_rate *= lr_epoch / warmup_epochs
    new_decay_epochs = [warmup_epochs] + decay_epochs
    for index, start_epoch in enumerate(new_decay_epochs):
      learning_rate = jnp.where(
          lr_epoch >= start_epoch,
          base_learning_rate * decay_ratio**index,
          learning_rate,
      )
    return learning_rate

  return schedule


def warm_up_polynomial_schedule(
    base_learning_rate: float,
    end_learning_rate: float,
    decay_steps: int,
    warmup_steps: int,
    decay_power: float,
) -> Callable:
  """Please see uncertainty_baselines.schedules.WarmUpPolynomialSchedule.
  """
  poly_schedule = optax.polynomial_schedule(
      init_value=base_learning_rate,
      end_value=end_learning_rate,
      power=decay_power,
      transition_steps=decay_steps,
  )

  def schedule(step):
    lr = poly_schedule(step)
    indicator = jnp.maximum(0.0, jnp.sign(warmup_steps - step))
    warmup_lr = base_learning_rate * step / warmup_steps
    lr = warmup_lr * indicator + (1 - indicator) * lr
    return lr

  return schedule

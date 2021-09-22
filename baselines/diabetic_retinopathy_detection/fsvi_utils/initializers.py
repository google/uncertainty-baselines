from typing import List, Tuple, Callable

import optax
from jax import numpy as jnp

from baselines.diabetic_retinopathy_detection.fsvi_utils.networks import CNN, Model
from baselines.diabetic_retinopathy_detection.fsvi_utils.objectives import Loss

DEFAULT_NUM_EPOCHS = 90


class Initializer:
    def __init__(
        self,
        activation: str,
        dropout_rate: float,
        input_shape: List[int],
        output_dim: int,
        kl_scale,
        stochastic_linearization: bool,
        n_samples,
        uniform_init_minval,
        uniform_init_maxval,
        w_init,
        b_init,
        init_strategy,
        prior_mean,
        prior_cov,
    ):
        """

        @param task: examples: continual_learning_pmnist, continual_learning_sfashionmnist
        @param n_inducing_inputs: number of inducing points to draw from each task
        @param output_dim: the task-specific number of output dimensions
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
        self.w_init = w_init
        self.b_init = b_init
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
        print(f"Stochastic linearization (posterior): {self.stochastic_linearization}")

    def initialize_model(
        self, rng_key,
    ):
        model = self._compose_model()
        init_fn, apply_fn = model.forward
        x_init = jnp.ones(self.input_shape)
        params_init, state = init_fn(
            rng_key, x_init, rng_key, model.stochastic_parameters, is_training=True
        )
        return model, init_fn, apply_fn, state, params_init

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

    def initialize_loss(self, model) -> Loss:
        metrics = Loss(
            model=model,
            kl_scale=self.kl_scale,
            n_samples=self.n_samples,
            stochastic_linearization=self.stochastic_linearization,
        )
        return metrics.nelbo_fsvi_classification

    def initialize_prior(self,) -> Callable[[Tuple], List[jnp.ndarray]]:
        """
        @predict_f_deterministic: function to do forward pass
        @param prior_mean: example: "0.0"
        @param prior_cov: example: "0.0"
        @return:
            inducing_input_fn
            prior_fn: a function that takes in an array of inducing input points and return the mean
                and covariance of the outputs at those points
        """
        _prior_mean, _prior_cov = (
            jnp.float32(self.prior_mean),
            jnp.float32(self.prior_cov),
        )

        def prior_fn(shape):
            prior_mean = jnp.ones(shape) * _prior_mean
            prior_cov = jnp.ones(shape) * _prior_cov
            return [prior_mean, prior_cov]

        return prior_fn


class OptimizerInitializer:
  def __init__(
    self,
    optimizer: str,
    base_learning_rate,
    n_batches,
    epochs,
    one_minus_momentum,
    lr_warmup_epochs,
    lr_decay_ratio,
    lr_decay_epochs,
    final_decay_factor,
    lr_schedule,
  ):
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
  steps_per_epoch, base_learning_rate, warmup_epochs, decay_epochs, decay_ratio,
):
  def schedule(count):
    lr_epoch = jnp.array(count, jnp.float32) / steps_per_epoch
    learning_rate = base_learning_rate
    if warmup_epochs >= 1:
      learning_rate *= lr_epoch / warmup_epochs
    _decay_epochs = [warmup_epochs] + decay_epochs
    for index, start_epoch in enumerate(_decay_epochs):
      learning_rate = jnp.where(
        lr_epoch >= start_epoch,
        base_learning_rate * decay_ratio ** index,
        learning_rate,
      )
    return learning_rate

  return schedule


def warm_up_polynomial_schedule(
  base_learning_rate, end_learning_rate, decay_steps, warmup_steps, decay_power,
):
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

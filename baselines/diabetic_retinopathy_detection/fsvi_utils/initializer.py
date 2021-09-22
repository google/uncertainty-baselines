from typing import List, Tuple, Callable

import jax
import jax.numpy as jnp

from baselines.diabetic_retinopathy_detection.fsvi_utils.networks import CNN, Model
from baselines.diabetic_retinopathy_detection.fsvi_utils.objectives import Objectives_hk as Objectives


class Initializer:
    def __init__(
        self,
        activation: str,
        dropout_rate,
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
        **kwargs,
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
        self,
        rng_key,
    ):
        model = self._compose_model()
        init_fn, apply_fn = model.forward
        # INITIALIZE NETWORK STATE + PARAMETERS
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
            dropout=self.dropout,
            dropout_rate=self.dropout_rate,
            uniform_init_minval=self.uniform_init_minval,
            uniform_init_maxval=self.uniform_init_maxval,
            w_init=self.w_init,
            b_init=self.b_init,
        )
        return model

    def initialize_objective(self, model) -> Objectives:
        metrics = Objectives(
            model=model,
            kl_scale=self.kl_scale,
            n_samples=self.n_samples,
            stochastic_linearization=self.stochastic_linearization,
        )
        return metrics

    @staticmethod
    def initialize_inducing_input_fn(
    ) -> Callable[[jnp.ndarray, jnp.ndarray, int], jnp.ndarray]:
        def inducing_input_fn(x_batch, rng_key, n_inducing_inputs):
            permutation = jax.random.permutation(key=rng_key, x=x_batch.shape[0])
            x_batch_permuted = x_batch[permutation, :]
            inducing_inputs = x_batch_permuted[:n_inducing_inputs]
            return inducing_inputs
        return inducing_input_fn

    @staticmethod
    def initialize_prior(
            prior_mean: str,
        prior_cov: str,
    ) -> Callable[[Tuple], List[jnp.ndarray]]:
        """
        @predict_f_deterministic: function to do forward pass
        @param prior_mean: example: "0.0"
        @param prior_cov: example: "0.0"
        @return:
            inducing_input_fn
            prior_fn: a function that takes in an array of inducing input points and return the mean
                and covariance of the outputs at those points
        """
        _prior_mean, _prior_cov = jnp.float32(prior_mean), jnp.float32(prior_cov)

        def prior_fn(shape):
            prior_mean = jnp.ones(shape) * _prior_mean
            prior_cov = jnp.ones(shape) * _prior_cov
            return [prior_mean, prior_cov]
        return prior_fn

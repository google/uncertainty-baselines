from functools import partial
from typing import Tuple, Callable, List

import jax
import jax.numpy as jnp
from jax import jit

import haiku as hk

from uncertainty_baselines.models.resnet50_fsvi import ResNet50FSVI

relu = jax.nn.relu
tanh = jnp.tanh

eps = 1e-6

ACTIVATION_DICT = {"tanh": jnp.tanh, "relu": jax.nn.relu}


class Model:
    def __init__(
        self,
        output_dim: int,
        architecture: List[int],
        activation_fn: str = "relu",
        stochastic_parameters: bool = False,
        linear_model: bool = False,
        regularization=0.0,
        dropout=False,
        dropout_rate=0.0,
        batch_normalization=False,
    ):
        """

        @param stochastic_parameters:
        @param linear_model: if True, then all the parameters except the last layer are set to be deterministic.
        """
        self.output_dim = output_dim
        self.regularization = regularization
        self.linear_model = linear_model
        self.dropout = dropout
        self.dropout_rate = dropout_rate
        self.activation_fn = ACTIVATION_DICT[activation_fn]
        self.architecture = architecture
        self.stochastic_parameters = stochastic_parameters

        self.forward = hk.transform_with_state(self.make_forward_fn())

    @property
    def apply_fn(self):
        return self.forward.apply

    def make_forward_fn(self):
        raise NotImplementedError

    @partial(jit, static_argnums=(0, 5,))
    def predict_f(self, params, state, inputs, rng_key, is_training):
        return self.forward.apply(
            params,
            state,
            rng_key,
            inputs,
            rng_key,
            stochastic=True,
            is_training=is_training,
        )[0]

    @partial(jit, static_argnums=(0, 5,))
    def predict_f_deterministic(self, params, state, inputs, rng_key, is_training):
        """
        Forward pass with mean parameters (hence deterministic)
        """
        return self.forward.apply(
            params,
            state,
            rng_key,
            inputs,
            rng_key,
            stochastic=False,
            is_training=is_training,
        )[0]

    @partial(jit, static_argnums=(0, 5,))
    def predict_y(self, params, state, inputs, rng_key, is_training):
        return jax.nn.softmax(
            self.predict_f(params, state, inputs, rng_key, is_training)
        )

    def predict_f_multisample(
        self, params, state, inputs, rng_key, n_samples: int, is_training: bool
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        @return:
            preds_samples: an array of shape (n_samples, inputs.shape[0], output_dimension)
            preds_mean: an array of shape (inputs.shape[0], output_dimension)
            preds_var: an array of shape (inputs.shape[0], output_dimension)
        """
        # TODO: test if vmap can accelerate this code
        return mc_sampling(
            fn=lambda rng_key: self.predict_f(
                params, state, inputs, rng_key, is_training
            ),
            n_samples=n_samples,
            rng_key=rng_key,
        )

    def predict_y_multisample(
        self, params, state, inputs, rng_key, n_samples, is_training
    ):
        return mc_sampling(
            fn=partial(self.predict_y, params, state, inputs, is_training=is_training),
            n_samples=n_samples,
            rng_key=rng_key,
        )

    @partial(jit, static_argnums=(0, 5, 6,))
    def predict_f_multisample_jitted(
        self, params, state, inputs, rng_key, n_samples: int, is_training: bool,
    ):
        """
        This is jitted version of predict_f_multisample
        """
        # loop 1 -- slower than vmap for n_samples = 10
        # rng_key, subkey = jax.random.split(rng_key)
        # preds_samples = jnp.expand_dims(self.predict_f(params, state, inputs, rng_key), 0)
        # for i in range(n_samples-1):
        #     rng_key, subkey = jax.random.split(rng_key)
        #     preds_samples = jnp.concatenate((preds_samples, jnp.expand_dims(self.predict_f(params, state, inputs, rng_key), 0)), 0)

        # loop 2 -- slower than vmap for n_samples = 10
        # preds_samples = jnp.expand_dims(self.predict_f(params, state, inputs, rng_key), 0)
        # for i in range(n_samples - 1):
        #     rng_key, subkey = jax.random.split(rng_key)
        #     preds_samples = jax.ops.index_update(
        #         preds_samples,  # the base array
        #         jax.ops.index[i:1+1, :, :],  # which indices we're accessing
        #         jnp.expand_dims(self.predict_f(params, state, inputs, rng_key), axis=0)
        #     )

        # vmap
        rng_keys = jax.random.split(rng_key, n_samples)
        _predict_multisample_fn = lambda rng_key: self.predict_f(
            params, state, inputs, rng_key, is_training,
        )
        predict_multisample_fn = jax.vmap(
            _predict_multisample_fn, in_axes=0, out_axes=0
        )  # fastest for n_samples=10
        # predict_multisample_fn = jit(jax.vmap(_predict_multisample_fn, in_axes=0, out_axes=0))
        # predict_multisample_fn = jax.vmap(partial(self.predict_f, params, state, inputs), in_axes=0, out_axes=0)
        # predict_multisample_fn = jit(jax.vmap(partial(self.predict_f, params, state, inputs), in_axes=0, out_axes=0))
        preds_samples = predict_multisample_fn(rng_keys)

        preds_mean = preds_samples.mean(axis=0)
        preds_var = preds_samples.std(axis=0) ** 2
        return preds_samples, preds_mean, preds_var

    @partial(jit, static_argnums=(0, 5, 6,))
    def predict_y_multisample_jitted(
        self, params, state, inputs, rng_key, n_samples, is_training
    ):
        # rng_key, subkey = jax.random.split(rng_key)
        # preds_y_samples = jnp.expand_dims(self.predict_y(params, state, inputs, rng_key), 0)
        # for i in range(n_samples-1):
        #     rng_key, subkey = jax.random.split(rng_key)
        #     preds_y_samples = jnp.concatenate((preds_y_samples, jnp.expand_dims(self.predict_y(params, state, inputs, rng_key), 0)), 0)
        # preds_y_mean = preds_y_samples.mean(0)
        # preds_y_var = preds_y_samples.std(0) ** 2
        rng_keys = jax.random.split(rng_key, n_samples)
        _predict_multisample_fn = lambda rng_key: self.predict_y(
            params, state, inputs, rng_key, is_training
        )
        predict_multisample_fn = jax.vmap(
            _predict_multisample_fn, in_axes=0, out_axes=0
        )
        preds_samples = predict_multisample_fn(rng_keys)
        preds_mean = preds_samples.mean(0)
        preds_var = preds_samples.std(0) ** 2
        return preds_samples, preds_mean, preds_var


class CNN(Model):
    def __init__(
        self,
        output_dim: int,
        architecture: List[int],
        activation_fn: str = "relu",
        stochastic_parameters: bool = False,
        linear_model: bool = False,
        regularization=0.0,
        dropout=False,
        dropout_rate=0.0,
        batch_normalization=False,
        uniform_init_minval: float = -20.,
        uniform_init_maxval: float = -18.,
    ):
        self.batch_normalization = batch_normalization
        self.uniform_init_minval = uniform_init_minval
        self.uniform_init_maxval = uniform_init_maxval
        super().__init__(
            output_dim=output_dim,
            architecture=architecture,
            activation_fn=activation_fn,
            stochastic_parameters=stochastic_parameters,
            linear_model=linear_model,
            regularization=regularization,
            dropout=dropout,
            dropout_rate=dropout_rate,
        )

    def make_forward_fn(self):
        if self.architecture == "resnet":

            def forward_fn(inputs, rng_key, stochastic, is_training):
                net = ResNet50FSVI(
                    output_dim=self.output_dim,
                    stochastic_parameters=self.stochastic_parameters,
                    dropout=self.dropout,
                    dropout_rate=self.dropout_rate,
                    linear_model=self.linear_model,
                    uniform_init_minval=self.uniform_init_minval,
                    uniform_init_maxval=self.uniform_init_maxval,
                )
                return net(inputs, rng_key, stochastic, is_training)

        else:
            raise NotImplementedError(self.architecture)
        return forward_fn


def mc_sampling(
    fn: Callable, n_samples: int, rng_key: jnp.ndarray
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Performs Monte Carlo sampling and returns the samples, the mean of samples and the variance of samples

    @param fn: a deterministic function that takes in a random key and returns one MC sample
    @param n_samples: number of MC samples
    @param rng_key: random key
    @return:
            preds_samples: an array of shape (n_samples, ) + `output_shape`, where `output_shape` is the shape
                of output of `fn`
            preds_mean: an array of shape `output_shape`
            preds_var: an array of shape `output_shape`
    """
    list_of_pred_samples = []
    for _ in range(n_samples):
        rng_key, subkey = jax.random.split(rng_key)
        output = fn(subkey)
        list_of_pred_samples.append(jnp.expand_dims(output, 0))
    preds_samples = jnp.concatenate(list_of_pred_samples, 0)
    preds_mean = preds_samples.mean(axis=0)
    preds_var = preds_samples.std(axis=0) ** 2
    return preds_samples, preds_mean, preds_var

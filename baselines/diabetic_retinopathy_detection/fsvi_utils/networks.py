# coding=utf-8
# Copyright 2022 The Uncertainty Baselines Authors.
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

"""FSVI networks."""
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
from typing import Callable, Tuple

import haiku as hk
import jax
import jax.numpy as jnp
from jax import jit

from uncertainty_baselines.models.resnet50_fsvi import resnet50_fsvi

ACTIVATION_DICT = {"tanh": jnp.tanh, "relu": jax.nn.relu}


class Model:
  """Model."""

  def __init__(
      self,
      output_dim: int,
      activation_fn: str = "relu",
      stochastic_parameters: bool = False,
      linear_model: bool = False,
      dropout: bool = False,
      dropout_rate: float = 0.0,
  ):
    """Wrapper of resnet50_fsvi

    Args:
    output_dim: the output dimension
    activation_fn: the type of activation function, e.g. "relu", "tanh"
    stochastic_parameters: if True, we keep a variational distribution of
      parameters.
    linear_model: if True, only put variational distribution on the last layer.
    dropout: if True, apply dropout.
    dropout_rate: dropout rate if we apply dropout.
    """
    self.output_dim = output_dim
    self.linear_model = linear_model
    self.dropout = dropout
    self.dropout_rate = dropout_rate
    self.activation_fn = ACTIVATION_DICT[activation_fn]
    self.stochastic_parameters = stochastic_parameters

    self.forward = hk.transform_with_state(self.make_forward_fn())

  @property
  def apply_fn(self) -> Callable:
    return self.forward.apply

  def make_forward_fn(self) -> Callable:
    raise NotImplementedError

  @partial(
      jit, static_argnums=(
          0,
          5,
      ))
  def predict_f(
      self,
      params: hk.Params,
      state: hk.State,
      inputs: jnp.ndarray,
      rng_key: jnp.ndarray,
      is_training: bool,
  ) -> jnp.ndarray:
    """Forward pass of model that returns pre-softmax output

    Args:
      params: parameters of model.
      state: state of model, e.g. the mean and std used in batch normalization.
      inputs: the input data.
      rng_key: jax random key.
      is_training: whether the model is in training mode.

    Returns:
      jax.numpy.ndarray, the pre-softmax output of the model
    """
    return self.forward.apply(
        params,
        state,
        rng_key,
        inputs,
        rng_key,
        stochastic=True,
        is_training=is_training,
    )[0]

  @partial(
      jit, static_argnums=(
          0,
          5,
      ))
  def predict_y(
      self,
      params: hk.Params,
      state: hk.State,
      inputs: jnp.ndarray,
      rng_key: jnp.ndarray,
      is_training: bool,
  ) -> jnp.ndarray:
    """Forward pass of model that returns post-softmax output"""
    return jax.nn.softmax(
        self.predict_f(params, state, inputs, rng_key, is_training))

  def predict_y_multisample(self, params, state, inputs, rng_key, n_samples,
                            is_training):
    """Monte-Carlo estimate of the post-softmax output using `n_samples` samples."""
    return mc_sampling(
        fn=partial(
            self.predict_y, params, state, inputs, is_training=is_training),
        n_samples=n_samples,
        rng_key=rng_key,
    )

  @partial(
      jit, static_argnums=(
          0,
          5,
          6,
      ))
  def predict_f_multisample_jitted(
      self,
      params,
      state,
      inputs,
      rng_key,
      n_samples: int,
      is_training: bool,
  ):
    """Jitted version of Monte-Carlo estimate of the pre-softmax output using `n_samples` samples."""
    rng_keys = jax.random.split(rng_key, n_samples)
    # pylint: disable=g-long-lambda
    predict_multisample_fn = lambda rng_key: self.predict_f(
        params,
        state,
        inputs,
        rng_key,
        is_training,
    )
    # pylint: enable=g-long-lambda
    predict_multisample_fn_vmapped = jax.vmap(
        predict_multisample_fn, in_axes=0, out_axes=0)
    preds_samples = predict_multisample_fn_vmapped(rng_keys)

    preds_mean = preds_samples.mean(axis=0)
    preds_var = preds_samples.std(axis=0)**2
    return preds_samples, preds_mean, preds_var

  @partial(
      jit, static_argnums=(
          0,
          5,
          6,
      ))
  def predict_y_multisample_jitted(self, params, state, inputs, rng_key,
                                   n_samples, is_training):
    """Jitted version of Monte-Carlo estimate of the post-softmax output using `n_samples` samples."""
    rng_keys = jax.random.split(rng_key, n_samples)
    # pylint: disable=g-long-lambda
    predict_multisample_fn = lambda rng_key: self.predict_y(
        params, state, inputs, rng_key, is_training)
    # pylint: enable=g-long-lambda
    predict_multisample_fn_vmapped = jax.vmap(
        predict_multisample_fn, in_axes=0, out_axes=0)
    preds_samples = predict_multisample_fn_vmapped(rng_keys)
    preds_mean = preds_samples.mean(0)
    preds_var = preds_samples.std(0)**2
    return preds_samples, preds_mean, preds_var


class CNN(Model):
  """CNN."""

  def __init__(
      self,
      output_dim: int,
      activation_fn: str = "relu",
      stochastic_parameters: bool = False,
      linear_model: bool = False,
      dropout: bool = False,
      dropout_rate: float = 0.0,
      uniform_init_minval: float = -20.0,
      uniform_init_maxval: float = -18.0,
      w_init: str = "uniform",
      b_init: str = "uniform",
  ):
    self.uniform_init_minval = uniform_init_minval
    self.uniform_init_maxval = uniform_init_maxval
    self.w_init = w_init
    self.b_init = b_init
    super().__init__(
        output_dim=output_dim,
        activation_fn=activation_fn,
        stochastic_parameters=stochastic_parameters,
        linear_model=linear_model,
        dropout=dropout,
        dropout_rate=dropout_rate,
    )

  def make_forward_fn(self) -> Callable:

    def forward_fn(inputs, rng_key, stochastic, is_training):
      net = resnet50_fsvi(
          output_dim=self.output_dim,
          stochastic_parameters=self.stochastic_parameters,
          dropout=self.dropout,
          dropout_rate=self.dropout_rate,
          linear_model=self.linear_model,
          uniform_init_minval=self.uniform_init_minval,
          uniform_init_maxval=self.uniform_init_maxval,
          w_init=self.w_init,
          b_init=self.b_init,
      )
      return net(inputs, rng_key, stochastic, is_training)

    return forward_fn


def mc_sampling(
    fn: Callable, n_samples: int,
    rng_key: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
  """Performs Monte Carlo sampling and returns the samples, the mean of samples

  and the variance of samples

  Args:
    fn: a deterministic function that takes in a random key and returns one MC
      sample.
    n_samples: number of MC samples.
    rng_key: jax random key.

  Returns:
    jax.numpy.ndarray, an array of shape (n_samples, ) + `output_shape`, where
    `output_shape` is the shape
          of output of `fn`
    jax.numpy.ndarray, an array of shape (output_shape,)
    jax.numpy.ndarray, an array of shape (output_shape,)
  """
  list_of_pred_samples = []
  for _ in range(n_samples):
    rng_key, subkey = jax.random.split(rng_key)
    output = fn(subkey)
    list_of_pred_samples.append(jnp.expand_dims(output, 0))
  preds_samples = jnp.concatenate(list_of_pred_samples, 0)
  preds_mean = preds_samples.mean(axis=0)
  preds_var = preds_samples.std(axis=0)**2
  return preds_samples, preds_mean, preds_var

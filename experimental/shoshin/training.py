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

"""Flax module for deep learning."""

import functools
import os
from typing import Sequence

from absl import logging
from clu import metric_writers
from flax import linen as nn
from flax.training import checkpoints
from flax.training import train_state  # Useful dataclass to keep train state
import jax
from jax.lib import xla_bridge
import jax.numpy as jnp
import numpy as np
import optax
import tensorflow as tf
import tf2jax


class MLP(nn.Module):
  hidden_sizes: Sequence[int]
  output_size: int

  @nn.compact
  def __call__(self, x):
    for sz in self.hidden_sizes:
      x = nn.relu(nn.Dense(sz)(x))
    x = nn.Dense(self.output_size)(x)
    return x


def cross_entropy_loss(*, logits, labels):
  logits = jnp.reshape(logits, [logits.shape[0], 1])
  labels = jnp.reshape(labels, [labels.shape[0], 1])
  return optax.sigmoid_binary_cross_entropy(logits=logits, labels=labels).mean()


def create_train_state(hidden_sizes, output_size, input_shape, rng,
                       learning_rate):
  """Creates initial `TrainState`."""
  mlp = MLP(hidden_sizes, output_size)
  params = mlp.init(rng, jnp.ones([1] + list(input_shape)))['params']
  tx = optax.adam(learning_rate)
  return train_state.TrainState.create(apply_fn=mlp.apply, params=params, tx=tx)


def compute_metrics(*, logits, labels):
  """Compute metrics."""
  loss = cross_entropy_loss(logits=logits, labels=labels)
  logits = jnp.reshape(logits, [logits.shape[0], 1])
  labels = jnp.reshape(labels, [labels.shape[0], 1])
  accuracy = jnp.mean(
      (2 * labels - 1) * logits > 0, axis=0)
  metrics = {
      'loss': loss,
      'accuracy': accuracy,
  }
  return metrics


def train_step(hidden_sizes, output_size, jax_func,
               state, jax_params, images, labels):
  """Train for a single step."""
  jax_output, jax_params = jax_func(jax_params, images)
  features = jnp.reshape(jax_output, [images.shape[0], -1])
  def loss_fn(params):
    logits = MLP(hidden_sizes, output_size).apply({'params': params}, features)
    loss = cross_entropy_loss(logits=logits, labels=labels)
    return loss, logits

  grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
  (_, logits), grads = grad_fn(state.params)
  state = state.apply_gradients(grads=grads)
  metrics = compute_metrics(logits=logits, labels=labels)
  return state, metrics, jax_params


def eval_step(hidden_sizes, output_size, jax_func,
              params, jax_params, images, labels):
  jax_output, jax_params = jax_func(jax_params, images)
  features = jnp.reshape(jax_output, [images.shape[0], -1])
  logits = MLP(hidden_sizes, output_size).apply({'params': params}, features)
  return compute_metrics(logits=logits, labels=labels), jax_params, logits


def eval_model(hidden_sizes, output_size, jax_func,
               params, jax_params, images):
  jax_output, jax_params = jax_func(jax_params, images)
  features = jnp.reshape(jax_output, [images.shape[0], -1])
  logits = MLP(hidden_sizes, output_size).apply({'params': params}, features)
  return jax.nn.sigmoid(logits)


def train_loop(config, workdir, train_ds, val_ds, preprocess):
  """Excute training loop based on parameters specified in config."""
  logging.info('JAX process: %d / %d', jax.process_index(), jax.process_count())
  logging.info('JAX local devices: %r', jax.local_devices())
  logging.info(xla_bridge.get_backend().platform)

  # Create pretrained resnet model
  # TODO(dvij): Remove hardcording of pretrained model here
  loaded_model = tf.keras.Sequential([
      tf.keras.applications.resnet50.ResNet50(
          include_top=False, weights='imagenet'),
      tf.keras.layers.GlobalAveragePooling2D()
  ])
  predictions = loaded_model(preprocess(next(train_ds))[0])
  f = tf.function(loaded_model, jit_compile=True)
  x = jnp.zeros([predictions.shape[0]] +
                list(config.input_shape))  # Define a sample input
  jax_func, jax_params = tf2jax.convert(f, x)

  # Initialize fine tuning model
  rng = jax.random.PRNGKey(config.train_seed)
  rng, init_rng = jax.random.split(rng)
  feature_shape = predictions.shape[1:]
  batch_size = predictions.shape[0]

  state = create_train_state(config.model.hidden_sizes,
                             config.model.output_size, feature_shape, init_rng,
                             config.optimizer.learning_rate)
  train_step_model = jax.jit(
      functools.partial(train_step, config.model.hidden_sizes,
                        config.model.output_size, jax_func))
  eval_step_model = jax.jit(
      functools.partial(eval_step, config.model.hidden_sizes,
                        config.model.output_size, jax_func))
  evaluate_model = jax.jit(
      functools.partial(eval_model, config.model.hidden_sizes,
                        config.model.output_size, jax_func))
  del init_rng  # Must not be used anymore.

  # Create Checkpointing and logging utilities
  best_accuracy_val = 0

  checkpoint_dir = os.path.join(workdir, 'checkpoints')
  logging.info('Checkpoint directory is %s', checkpoint_dir)
  step_init = 0
  ckpt_params = checkpoints.restore_checkpoint(checkpoint_dir,
                                               (step_init, state))
  step_init, state = ckpt_params
  writer = metric_writers.create_default_writer(
      workdir, just_logging=jax.process_index() > 0)
  writer.write_hparams(dict(config))
  best_params = state
  # Main training loop
  for step in range(step_init, config.optimizer.num_steps):
    images, labels = preprocess(next(train_ds))
    images = images[np.mod(np.arange(batch_size), images.shape[0]), ...]
    labels = labels[np.mod(np.arange(batch_size), labels.shape[0]), ...]
    state, _, jax_params = train_step_model(state, jax_params, images, labels)
    if step % config.logging_frequency == 0:
      logging.info(xla_bridge.get_backend().platform)
      images, labels = preprocess(next(val_ds))
      images = images[np.mod(np.arange(batch_size), images.shape[0]), ...]
      labels = labels[np.mod(np.arange(batch_size), labels.shape[0]), ...]
      eval_metrics, jax_params, _ = eval_step_model(
          state.params, jax_params, images, labels)
      if best_accuracy_val < eval_metrics['accuracy'].item():
        best_accuracy_val = eval_metrics['accuracy'].item()
        best_params = state
      measures = {
          'acc': eval_metrics['accuracy'].item(),
          'best_acc': best_accuracy_val
      }
      for key, val in measures.items():
        logging.info('%s: %f', key, val)
      writer.write_scalars(step, measures)
    if config.checkpoint_every:
      if (step % config.checkpoint_every == 0 or
          step == config.optimizer.num_steps):
        checkpoints.save_checkpoint(
            checkpoint_dir, (step, best_params), step=step, overwrite=True)

  def predictor(batch):
    images, _ = preprocess(batch)
    init_shape = images.shape[0]
    images = images[np.mod(np.arange(batch_size), images.shape[0]), ...]
    return evaluate_model(state.params, jax_params, images)[:init_shape]

  return predictor

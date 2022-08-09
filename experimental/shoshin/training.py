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

from absl import logging
from clu import metric_writers
from flax import linen as nn
from flax.training import checkpoints
from flax.training import train_state  # Useful dataclass to keep train state
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tensorflow as tf


class MLP(nn.Module):
  """A simple MLP model."""
  # TODO(dvij): Remove hardcording of layer sizes here

  @nn.compact
  def __call__(self, x):
    x = x.reshape((x.shape[0], -1))  # flatten
    for _ in range(2):
      x = nn.Dense(features=100)(x)
      x = nn.relu(x)
    x = nn.Dense(features=10)(x)
    return x


def preprocess_batch(loaded_model, input_shape, batch):
  image = tf.image.resize(batch['image'], input_shape[:2])
  image = tf.keras.applications.resnet50.preprocess_input(image)
  return np.squeeze(loaded_model(image).numpy()), batch['label']


def cross_entropy_loss(*, logits, labels):
  labels_onehot = jax.nn.one_hot(labels, num_classes=10)
  return optax.softmax_cross_entropy(logits=logits, labels=labels_onehot).mean()


def create_train_state(hidden_sizes, output_size,
                       input_shape, rng, learning_rate):
  """Creates initial `TrainState`."""
  del hidden_sizes, output_size
  mlp = MLP()
  params = mlp.init(rng, jnp.ones([1] + list(input_shape)))['params']
  tx = optax.adam(learning_rate)
  return train_state.TrainState.create(
      apply_fn=mlp.apply, params=params, tx=tx)


def compute_metrics(*, logits, labels):
  loss = cross_entropy_loss(logits=logits, labels=labels)
  accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
  metrics = {
      'loss': loss,
      'accuracy': accuracy,
  }
  return metrics


def train_step(hidden_sizes, output_size, state, features, labels):
  """Train for a single step."""
  del hidden_sizes, output_size
  def loss_fn(params):
    logits = MLP().apply({'params': params}, features)
    loss = cross_entropy_loss(logits=logits, labels=labels)
    return loss, logits
  grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
  (_, logits), grads = grad_fn(state.params)
  state = state.apply_gradients(grads=grads)
  metrics = compute_metrics(logits=logits, labels=labels)
  return state, metrics


def eval_step(hidden_sizes, output_size, params, features, labels):
  del hidden_sizes, output_size
  logits = MLP().apply({'params': params}, features)
  return compute_metrics(logits=logits, labels=labels)


def train_loop(config, workdir, train_ds, val_ds):
  """Excute training loop based on parameters specified in config."""
   # Create pretrained resnet model
  # TODO(dvij): Remove hardcording of pretrained model here
  loaded_model = tf.keras.Sequential([
      tf.keras.applications.resnet50.ResNet50(
          include_top=False, weights='imagenet'),
      tf.keras.layers.GlobalAveragePooling2D()
  ])
  preprocess = functools.partial(preprocess_batch, loaded_model,
                                 config.input_shape)

  # Initialize fine tuning model
  rng = jax.random.PRNGKey(config.train_seed)
  rng, init_rng = jax.random.split(rng)
  feature_shape = preprocess(next(train_ds))[0].shape[1:]

  state = create_train_state(config.model.hidden_sizes,
                             config.model.output_size, feature_shape, init_rng,
                             config.optimizer.learning_rate)
  train_step_model = jax.jit(
      functools.partial(train_step, config.model.hidden_sizes,
                        config.model.output_size))
  eval_step_model = jax.jit(
      functools.partial(eval_step, config.model.hidden_sizes,
                        config.model.output_size))
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
    features, labels = preprocess(next(train_ds))
    state, _ = train_step_model(state, features, labels)
    if step % config.logging_frequency == 0:
      features, labels = preprocess(next(val_ds))
      eval_metrics = eval_step_model(state.params, features, labels)
      if best_accuracy_val < eval_metrics['accuracy']:
        best_accuracy_val = eval_metrics['accuracy']
        best_params = state
      measures = {
          'acc': eval_metrics['accuracy'], 'best_acc': best_accuracy_val}
      for key, val in measures.items():
        logging.info('%s: %f', key, val)
      writer.write_scalars(step, measures)
    if config.checkpoint_every:
      if (step % config.checkpoint_every == 0 or
          step == config.optimizer.num_steps):
        checkpoints.save_checkpoint(
            checkpoint_dir, (step, best_params),
            step=step,
            overwrite=True)




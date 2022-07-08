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

"""Train a cross-validated ensemble.

Split the data k fold and train each model on a separate slice.
"""

import functools
import os
from typing import Callable, Tuple, Sequence, Any

from absl import app
from absl import flags
from absl import logging
from clu import metric_writers
from flax.training import checkpoints

# TODO(dvij): Move to flax
import haiku as hk
import jax
import jax.numpy as jnp
from ml_collections import config_flags
import numpy as np
import optax
import tensorflow as tf
import tensorflow_datasets as tfds

OptState = optax.OptState

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file(
    'config', default='config.py', help_string='config file', lock_config=False)

_WORKDIR = flags.DEFINE_string(
    'workdir', None, 'Work unit directory.', required=True)
flags.mark_flags_as_required(['config', 'workdir'])


# Training loss (cross-entropy).
def loss_fn(net: hk.Transformed, params: hk.Params, features: np.ndarray,
            label: np.ndarray) -> jnp.ndarray:
  """Compute the xent-loss of the network."""
  logits = net.apply(params, features)
  labels = jax.nn.one_hot(label, 10)

  softmax_xent = -jnp.sum(labels * jax.nn.log_softmax(logits))
  softmax_xent /= labels.shape[0]

  return softmax_xent


# Evaluation metric (classification accuracy).
def accuracy_fn(
    net: hk.Transformed,
) -> Callable[[hk.Params, jnp.ndarray, jnp.ndarray], jnp.ndarray]:
  """Get network accuracy on batch."""

  def accuracy(params: hk.Params, features: jnp.ndarray,
               label: jnp.ndarray) -> jnp.ndarray:
    predictions = net.apply(params, features)
    return jnp.mean(jnp.argmax(predictions, axis=-1) == label)

  return jax.jit(accuracy)


def update_fn(
    net: hk.Transformed,
    learning_rate: float,
) -> Tuple[Any, optax.GradientTransformation]:
  """Learning rule (stochastic gradient descent)."""
  opt = optax.adam(learning_rate)
  loss = functools.partial(loss_fn, net)

  def update(params: hk.Params, opt_state: optax.OptState, features: np.ndarray,
             label: np.ndarray) -> Tuple[hk.Params, optax.OptState]:
    grads = jax.grad(loss)(params, features, label)
    updates, opt_state = opt.update(grads, opt_state)
    new_params = optax.apply_updates(params, updates)
    return new_params, opt_state

  return jax.jit(update), opt


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  config = FLAGS.config
  batch_size = config.train_batch_size
  index = config.index  # index of CV fold to be used for training/validation
  input_shape = (224, 224, 3)  # pretrained-resnet expects this input size

  # Create train and validation splits
  vals_ds = tfds.load(
      'cifar10',
      split=[f'train[{k}%:{k+10}%]' for k in range(0, 100, 10)])[index]
  trains_ds = tfds.load(
      'cifar10',
      split=[f'train[:{k}%]+train[{k+10}%:]' for k in range(0, 100, 10)])[index]

  # Create dataset iterators
  train_ds = iter(
      tfds.as_numpy(trains_ds.cache().repeat().shuffle(
          10 * batch_size, seed=0).batch(batch_size)))
  val_ds = iter(
      tfds.as_numpy(vals_ds.cache().repeat().batch(config.eval_batch_size)))

  # Create pretrained resnet model
  loaded_model = tf.keras.Sequential([
      tf.keras.applications.resnet50.ResNet50(
          include_top=False, weights='imagenet'),
      tf.keras.layers.GlobalAveragePooling2D()
  ])

  # Create preprocessing compatible with pretrained resnet model

  def preprocess_batch(batch):
    image = tf.image.resize(batch['image'], input_shape[:2])
    image = tf.keras.applications.resnet50.preprocess_input(image)
    return np.squeeze(loaded_model(image).numpy()), batch['label']

  # Create Haiku module that will be finetuned on top of resnet
  key_seq = hk.PRNGSequence(23)
  haiku_fn = lambda x: hk.nets.MLP(list(config.hidden_sizes) + [10])(x)  # pylint: disable=unnecessary-lambda
  network = hk.without_apply_rng(hk.transform(haiku_fn))
  params = network.init(next(key_seq), preprocess_batch(next(train_ds))[0])
  train_step, opt = update_fn(network, config.learning_rate)
  eval_net = accuracy_fn(network)
  opt_state = opt.init(params)

  # Create Checkpointing and logging utilities
  best_accuracy_val = 0
  checkpoint_dir = os.path.join(_WORKDIR.value, 'checkpoints')
  logging.info('Checkpoint directory is %s', checkpoint_dir)
  step_init = 0
  ckpt_params = checkpoints.restore_checkpoint(checkpoint_dir,
                                               (step_init, params, opt_state))
  step_init, params, opt_state = ckpt_params
  writer = metric_writers.create_default_writer(
      FLAGS.workdir, just_logging=jax.process_index() > 0)
  writer.write_hparams(dict(config))
  best_params = params

  # Main training loop
  for step in range(step_init, config.num_steps):
    features, labels = preprocess_batch(next(train_ds))
    params, opt_state = train_step(params, opt_state, features, labels)
    if step % config.logging_frequency == 0:
      features, labels = preprocess_batch(next(val_ds))
      accuracy_val = eval_net(params, features, labels)
      if best_accuracy_val < accuracy_val:
        best_accuracy_val = accuracy_val
        best_params = params
      measures = {'acc': accuracy_val, 'best_acc': best_accuracy_val}
      for key, val in measures.items():
        logging.info('%s: %f', key, val)
      writer.write_scalars(step, measures)
    if config.checkpoint_every:
      if step % config.checkpoint_every == 0 or step == config.num_steps:
        checkpoints.save_checkpoint(
            checkpoint_dir, (step, best_params, opt_state),
            step=step,
            overwrite=True)


if __name__ == '__main__':
  app.run(main)

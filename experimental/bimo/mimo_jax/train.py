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

"""CIFAR-10 example."""

import ast
import functools
import math

from absl import app
from absl import flags
from absl import logging
from flax import jax_utils
from flax import optim
from flax.deprecated import nn
from flax.metrics import tensorboard
from flax.training import common_utils
from flax.training import lr_schedule
import jax
from jax import random
import jax.nn
import jax.numpy as jnp
import tensorflow as tf
import uncertainty_baselines as ub

import model as mimo_model  # local file import


FLAGS = flags.FLAGS

flags.DEFINE_float(
    'base_lr', default=0.1,
    help=('The learning rate for the momentum optimizer.'))

flags.DEFINE_string(
    'lr_decay_percs',
    default='[.4, .8, .9]',
    help='Learning rate schedule decay steps as a Python list.')

flags.DEFINE_float(
    'lr_decay_ratio',
    default=0.1,
    help='Amount to decay learning rate on decay steps.')

flags.DEFINE_integer(
    'lr_warmup_epochs',
    default=1,
    help='Number of epochs for a linear warmup of the learning rate.')

flags.DEFINE_float(
    'momentum', default=0.9,
    help=('The decay rate used for the momentum optimizer.'))

flags.DEFINE_float(
    'l2_reg', default=3e-4,
    help=('The amount of L2-regularization to apply.'))

flags.DEFINE_integer(
    'batch_size', default=512,
    help=('Batch size for training.'))

flags.DEFINE_integer(
    'num_epochs', default=250,
    help=('Number of training epochs.'))

flags.DEFINE_integer(
    'mimo_size', default=3,
    help=('Number of MIMO ensemble members.'))

flags.DEFINE_integer(
    'batch_repetitions', default=4,
    help='Number of repetitions in a batch.')

flags.DEFINE_enum(
    'arch', default='wrn28_10',
    enum_values=['wrn28_10', 'wrn28_2'],
    help=('Network architecture; wrn28_10, wrn28_2'))

flags.DEFINE_float(
    'wrn_dropout_rate', default=0.0,
    help=('Wide ResNet DropOut rate.'))

flags.DEFINE_integer(
    'rng', default=0,
    help=('Random seed for network initialization.'))

flags.DEFINE_string(
    'model_dir', default='/tmp/mimo',
    help=('Directory to store model data.'))


@functools.partial(jax.jit, static_argnums=(1, 2, 3))
def create_model(prng_key, batch_size, image_size, module):
  input_shape = (batch_size, image_size, image_size, 3 * FLAGS.mimo_size)
  with nn.stateful() as init_state:
    with nn.stochastic(jax.random.PRNGKey(0)):
      _, initial_params = module.init_by_shape(
          prng_key, [(input_shape, jnp.float32)])
      model = nn.Model(module, initial_params)
  return model, init_state


def create_optimizer(model, learning_rate, beta):
  optimizer_def = optim.Momentum(learning_rate=learning_rate,
                                 beta=beta,
                                 nesterov=True)
  optimizer = optimizer_def.create(model)
  optimizer = jax_utils.replicate(optimizer)
  return optimizer


def cross_entropy_loss(logits, labels):
  log_softmax_logits = jax.nn.log_softmax(logits)
  num_classes = log_softmax_logits.shape[-1]
  one_hot_labels = common_utils.onehot(labels, num_classes)
  return -jnp.sum(one_hot_labels * log_softmax_logits) / labels.size


def compute_metrics(logits, labels):
  loss = cross_entropy_loss(logits, labels)
  error_rate = jnp.mean(jnp.argmax(logits, -1) != labels)
  metrics = {
      'loss': loss,
      'error_rate': error_rate,
  }
  metrics = jax.lax.pmean(metrics, 'batch')
  return metrics


def train_step(optimizer, state, batch, prng_key, learning_rate_fn, l2_reg):
  """Perform a single training step."""

  orig_batch_size = batch['features'].shape[0]
  keys = random.split(prng_key, FLAGS.mimo_size)
  # [0, 1, ... , bs-1, 0, 1, ..., bs-1, ...]
  inds = jnp.tile(jnp.arange(orig_batch_size), [FLAGS.batch_repetitions])
  batches = []
  labels = []
  for i in range(FLAGS.mimo_size):
    order = random.permutation(keys[i], inds)
    batches.append(batch['features'][order])
    labels.append(batch['labels'][order])
  # batches is a list of length mimo_size containing
  # [orig_batch_size * batch_repetitions, H, W, num_channels] arrays.
  # labels is a list of length mimo_size containing
  # [orig_batch_size * batch_repetitions] arrays.
  # Concatenate only the images along channel dimension, Results in shape
  # [orig_batch_size * batch_repetitions, H, W, num_channels * mimo_size].
  batches = jnp.concatenate(batches, axis=3)

  def loss_fn(model):
    """loss function used for training."""
    with nn.stateful(state) as new_state:
      with nn.stochastic(prng_key):
        logits = model(batches)

    loss = 0.
    # logits is [orig_batch_size * batch_repetitions, mimo_size * 10]
    # labels is of length mimo_size containing tensors of shape
    # [orig_batch_size * batch_repetitions].
    for i in range(FLAGS.mimo_size):
      loss += cross_entropy_loss(logits[:, i*10:i*10+10], labels[i])
    logits = logits[:, :10]
    weight_penalty_params = jax.tree_leaves(model.params)
    print(model.params)
    # TODO(dieterichl): apply l2 reg only to weights, not biases or other params
    weight_l2 = sum([jnp.sum(x ** 2)
                     for x in weight_penalty_params
                     if x.ndim > 1])
    weight_penalty = l2_reg * weight_l2
    loss = loss + weight_penalty
    return loss, (new_state, logits)

  step = optimizer.state.step
  lr = learning_rate_fn(step)
  grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
  (_, (new_state, logits)), grad = grad_fn(optimizer.target)
  grad = jax.lax.pmean(grad, 'batch')
  new_optimizer = optimizer.apply_gradient(
      grad, learning_rate=lr)

  metrics = compute_metrics(logits, labels[0])
  metrics['learning_rate'] = lr
  return new_optimizer, new_state, metrics


def eval_step(model, state, batch):
  """Evaluate the model."""
  state = jax.lax.pmean(state, 'batch')
  with nn.stateful(state, mutable=False):
    batch_image = jnp.concatenate(
        [batch['features'] for i in range(FLAGS.mimo_size)], axis=3)
    # Array of shape [batch_size, mimo_size * 10]
    logits = model(batch_image, train=False)
    # Array of shape [mimo_size, batch_size , 10]
    normalized_logits = jnp.array(
        [jax.nn.log_softmax(logits[:, i * 10:i * 10 + 10])
         for i in range(FLAGS.mimo_size)])
    # Array of shape [batch_size, 10]
    avg_logits = (jax.nn.logsumexp(normalized_logits, axis=0)
                  - jnp.log(FLAGS.mimo_size))

    # logits = 1 / FLAGS.mimo_size * sum([
    #     nn.softmax(logits[:, i * 10:i * 10 + 10])
    #     for i in range(FLAGS.mimo_size)
    # ])
    # logits = jnp.log(logits)
  return compute_metrics(avg_logits, batch['labels'])


def load_and_shard_tf_batch(xs):
  del xs['_enumerate_added_per_step_id']

  local_device_count = jax.local_device_count()
  def _prepare(x):
    # Use _numpy() for zero-copy conversion between TF and NumPy.
    x = x._numpy()  # pylint: disable=protected-access
    return jax.numpy.array(x.reshape((local_device_count, -1) + x.shape[1:]))

  return jax.tree_map(_prepare, xs)


def make_lr_fn(base_learning_rate, steps_per_epoch, num_train_epochs):
  decay_percs = ast.literal_eval(FLAGS.lr_decay_percs)
  decay_epochs = [int(p * num_train_epochs) for p in decay_percs]
  lr_decay_sched = [[epoch, math.pow(FLAGS.lr_decay_ratio, i + 1)]
                    for i, epoch in enumerate(decay_epochs)]
  return lr_schedule.create_stepped_learning_rate_schedule(
      base_learning_rate, steps_per_epoch, lr_decay_sched,
      warmup_length=FLAGS.lr_warmup_epochs)


def make_model_module_fn():
  if FLAGS.arch == 'wrn28_10':
    return mimo_model.WideResnet.partial(
        blocks_per_group=4,
        channel_multiplier=10,
        num_outputs=10*FLAGS.mimo_size,
        dropout_rate=FLAGS.wrn_dropout_rate)

  if FLAGS.arch == 'wrn28_2':
    return mimo_model.WideResnet.partial(
        blocks_per_group=4,
        channel_multiplier=2,
        num_outputs=10*FLAGS.mimo_size,
        dropout_rate=FLAGS.wrn_dropout_rate)


def train(model_dir, batch_size, num_epochs, base_learning_rate,
          sgd_momentum, l2_reg=0.0005, run_seed=0):
  """Train model."""
  if jax.process_count() > 1:
    raise ValueError('CIFAR-10 example should not be run on '
                     'more than 1 host (for now)')

  summary_writer = tensorboard.SummaryWriter(model_dir)

  rng = random.PRNGKey(run_seed)
  train_batch_size = batch_size // FLAGS.batch_repetitions
  eval_batch_size = batch_size
  if batch_size % jax.device_count() > 0:
    raise ValueError('Batch size must be divisible by the number of devices')
  device_batch_size = batch_size // jax.device_count()

  train_builder = ub.datasets.Cifar10Dataset(split='train')
  train_ds = train_builder.load(batch_size=train_batch_size)

  eval_builder = ub.datasets.Cifar10Dataset(split='test')
  eval_ds = eval_builder.load(batch_size=eval_batch_size)

  # Compute steps per epoch and nb of eval steps
  steps_per_epoch = train_builder.num_examples // train_batch_size
  steps_per_eval = eval_builder.num_examples // eval_batch_size
  num_steps = steps_per_epoch * num_epochs

  # Create the model
  image_size = 32
  model, state = create_model(rng, device_batch_size, image_size,
                              make_model_module_fn())
  state = jax_utils.replicate(state)
  optimizer = create_optimizer(model, base_learning_rate, sgd_momentum)
  del model  # don't keep a copy of the initial model

  # Learning rate schedule
  train_lr = base_learning_rate * train_batch_size / 128
  learning_rate_fn = make_lr_fn(train_lr, steps_per_epoch, FLAGS.num_epochs)

  # pmap the train and eval functions
  p_train_step = jax.pmap(
      functools.partial(train_step, learning_rate_fn=learning_rate_fn,
                        l2_reg=l2_reg),
      axis_name='batch')
  p_eval_step = jax.pmap(eval_step, axis_name='batch')

  # Create dataset batch iterators
  train_iter = iter(train_ds)

  # Gather metrics
  train_metrics = []
  epoch = 1
  for step, batch in zip(range(num_steps), train_iter):
    # Generate a PRNG key that will be rolled into the batch
    rng, step_key = jax.random.split(rng)
    # Load and shard the TF batch
    batch = load_and_shard_tf_batch(batch)
    # Shard the step PRNG key
    sharded_keys = common_utils.shard_prng_key(step_key)

    # Train step
    optimizer, state, metrics = p_train_step(
        optimizer, state, batch, sharded_keys)
    train_metrics.append(metrics)

    if (step + 1) % steps_per_epoch == 0:
      # We've finished an epoch
      train_metrics = common_utils.get_metrics(train_metrics)
      # Get training epoch summary for logging
      train_summary = jax.tree_map(lambda x: x.mean(), train_metrics)
      # Send stats to TensorBoard
      for key, vals in train_metrics.items():
        tag = 'train_%s' % key
        for i, val in enumerate(vals):
          summary_writer.scalar(tag, val, step - len(vals) + i + 1)
      # Reset train metrics
      train_metrics = []

      # Evaluation
      eval_metrics = []
      eval_iter = iter(eval_ds)
      for _ in range(steps_per_eval):
        eval_batch = next(eval_iter)
        # Load and shard the TF batch
        eval_batch = load_and_shard_tf_batch(eval_batch)
        # Step
        metrics = p_eval_step(optimizer.target, state, eval_batch)
        eval_metrics.append(metrics)
      eval_metrics = common_utils.get_metrics(eval_metrics)
      # Get eval epoch summary for logging
      eval_summary = jax.tree_map(lambda x: x.mean(), eval_metrics)

      # Log epoch summary
      logging.info(
          'Epoch %d: TRAIN loss=%.6f, err=%.2f, EVAL loss=%.6f, err=%.2f',
          epoch, train_summary['loss'], train_summary['error_rate'] * 100.0,
          eval_summary['loss'], eval_summary['error_rate'] * 100.0)

      summary_writer.scalar('eval_loss', eval_summary['loss'], epoch)
      summary_writer.scalar('eval_error_rate', eval_summary['error_rate'],
                            epoch)
      summary_writer.flush()

      epoch += 1


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  # Make sure tf does not allocate gpu memory.
  tf.config.experimental.set_visible_devices([], 'GPU')
  train(FLAGS.model_dir, FLAGS.batch_size, FLAGS.num_epochs,
        FLAGS.base_lr, FLAGS.momentum,
        l2_reg=FLAGS.l2_reg, run_seed=FLAGS.rng)


if __name__ == '__main__':
  flags.mark_flag_as_required('model_dir')
  app.run(main)

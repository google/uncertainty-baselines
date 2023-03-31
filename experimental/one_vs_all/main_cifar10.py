# coding=utf-8
# Copyright 2023 The Uncertainty Baselines Authors.
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

r"""Wide ResNet 28-10 on CIFAR-10/100 trained with one-vs-all classifiers.

"""

import os
from absl import app
from absl import flags
from absl import logging

import numpy as np
import tensorflow as tf
import uncertainty_baselines as ub
import losses as loss_lib  # local file import from experimental.one_vs_all
import models as models_lib  # local file import from experimental.one_vs_all


# Flags relating to hyperparameters.
flags.DEFINE_integer('batch_size', 512, 'The training batch size.')
flags.DEFINE_integer('eval_batch_size', 100, 'The evaluation batch size.')
flags.DEFINE_string('optimizer', 'adam', 'The optimizer to train with.')
flags.DEFINE_float('learning_rate', 0.01, 'The learning rate.')
flags.DEFINE_float('weight_decay', None,
                   'The model decoupled weight decay rate.')
flags.DEFINE_integer('lr_warmup_epochs', 1,
                     'Number of epochs for a linear warmup to the initial '
                     'learning rate. Use 0 to do no warmup.')
flags.DEFINE_float('lr_decay_ratio', 0.2, 'Amount to decay learning rate.')
flags.DEFINE_list('lr_decay_epochs', ['60', '120', '160'],
                  'Epochs to decay learning rate by.')

# Flags relating to setting up the job.
flags.DEFINE_bool('use_tpu', False, 'Whether to run on CPU or TPU.')

# Flags relating to the training/eval loop.
flags.DEFINE_integer('eval_frequency', 100,
                     'How many steps between evaluating on the (validation and)'
                     'test set.')
flags.DEFINE_integer('train_steps', 2000, 'How many steps to train for.')
flags.DEFINE_integer('seed', 1337, 'Random seed.')
flags.DEFINE_enum('dataset', 'cifar10',
                  enum_values=['cifar10', 'cifar100'],
                  help='Dataset.')

# Misc flags
flags.DEFINE_integer('num_bins', 15, 'Number of bins for ECE.')
flags.DEFINE_string('output_dir', '/tmp/cifar', 'Output directory.')

# Accelerator flags.
flags.DEFINE_bool('use_gpu', False, 'Whether to run on GPU or otherwise TPU.')
flags.DEFINE_bool('use_bfloat16', False, 'Whether to use mixed precision.')
flags.DEFINE_integer('num_cores', 8, 'Number of TPU cores or number of GPUs.')
flags.DEFINE_string('tpu', None,
                    'Name of the TPU. Only used if use_gpu is False.')

# Loss-specific flags.
flags.DEFINE_float('dm_alpha', 1.0, 'DM Alpha parameter.')
flags.DEFINE_bool('distance_logits', False,
                  'Whether to use a distance-based last layer.')
flags.DEFINE_enum('loss_name', 'crossentropy',
                  enum_values=['crossentropy', 'dm_loss', 'one_vs_all',
                               'focal_loss'],
                  help='Loss function')

# Model flags.
flags.DEFINE_enum('model_name', 'wide_resnet',
                  enum_values=['resnet20', 'wide_resnet'],
                  help='Model to use for training.')
FLAGS = flags.FLAGS


def _check_batch_replica_divisible(
    total_batch_size: int,
    strategy: tf.distribute.Strategy):
  """Ensure the batch size is evenly divisible by the number of replicas."""
  if total_batch_size % strategy.num_replicas_in_sync != 0:
    raise ValueError(
        'Batch size must be evenly divisible by the number of replicas in the '
        'job. Total batch size: {}, num replicas: {}'.format(
            total_batch_size, strategy.num_replicas_in_sync))


def _ds_as_tuple(ds):
  return ds.map(lambda d: (d['features'], d['labels']))


def run(trial_dir: str):
  """Run the experiment."""
  tf.random.set_seed(FLAGS.seed)
  np.random.seed(FLAGS.seed)
  strategy = ub.strategy_utils.get_strategy(FLAGS.tpu, FLAGS.use_tpu)
  with strategy.scope():
    # Setup CIFAR-10 tf.data.Dataset splits.
    # Use 5000 validation images.
    train_dataset_builder = ub.datasets.Cifar10Dataset(
        split='train', validation_percent=0.1)
    train_dataset = train_dataset_builder.load(batch_size=FLAGS.batch_size)
    train_dataset = _ds_as_tuple(train_dataset)
    train_dataset = strategy.experimental_distribute_dataset(train_dataset)

    val_dataset_builder = ub.datasets.Cifar10Dataset(
        split='validation', validation_percent=0.1)
    val_dataset = val_dataset_builder.load(batch_size=FLAGS.eval_batch_size)
    val_dataset = _ds_as_tuple(val_dataset)
    val_dataset = strategy.experimental_distribute_dataset(val_dataset)

    test_dataset_builder = ub.datasets.Cifar10Dataset(split='test')
    test_dataset = test_dataset_builder.load(batch_size=FLAGS.eval_batch_size)
    test_dataset = _ds_as_tuple(test_dataset)
    test_dataset = strategy.experimental_distribute_dataset(test_dataset)

    # Setup optimizer.
    _check_batch_replica_divisible(FLAGS.batch_size, strategy)
    _check_batch_replica_divisible(FLAGS.eval_batch_size, strategy)
    optimizer = ub.optimizers.get(
        optimizer_name=FLAGS.optimizer,
        learning_rate_schedule='constant',
        learning_rate=FLAGS.learning_rate,
        weight_decay=FLAGS.weight_decay)

    # Setup model.
    # TODO(shreyaspadhy): How does one get the number of classes in dataset
    model = models_lib.create_model(
        batch_size=FLAGS.batch_size,
        l2_weight=None,
        num_classes=10,
        distance_logits=FLAGS.distance_logits)
    loss_fn = loss_lib.get(
        loss_name=FLAGS.loss_name,
        from_logits=True,
        dm_alpha=FLAGS.dm_alpha,
        focal_gamma=FLAGS.focal_gamma)
    model.compile(
        optimizer=optimizer,
        loss=loss_fn,
        metrics=['sparse_categorical_accuracy'])

    # Train and eval.
    steps_per_epoch = train_dataset_builder.num_examples // FLAGS.batch_size
    validation_steps = (
        val_dataset_builder.num_examples // FLAGS.eval_batch_size)
    history = model.fit(
        x=train_dataset,
        batch_size=FLAGS.batch_size,
        epochs=FLAGS.train_steps // steps_per_epoch,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_dataset,
        validation_steps=validation_steps,
        validation_freq=FLAGS.eval_frequency,
        shuffle=False)
    logging.info(history)

    test_steps = test_dataset_builder.num_examples // FLAGS.eval_batch_size
    test_result = model.evaluate(
        x=test_dataset,
        batch_size=FLAGS.eval_batch_size,
        steps=test_steps)
    logging.info(test_result)

    # Save a checkpoint after training.
    if trial_dir:
      model.save_weights(
          os.path.join(trial_dir, 'model.ckpt-{}'.format(FLAGS.train_steps)))




def main(argv):
  del argv
  logging.info('Starting CIFAR-10 ResNet-20 experiment!')
  trial_dir = os.path.join(FLAGS.output_dir, '0')
  logging.info('Saving to dir: %s', trial_dir)
  if not tf.io.gfile.exists(trial_dir):
    tf.io.gfile.makedirs(trial_dir)
  return run(trial_dir)


if __name__ == '__main__':
  app.run(main)

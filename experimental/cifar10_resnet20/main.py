# coding=utf-8
# Copyright 2020 The Uncertainty Baselines Authors.
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

# Lint as: python3
r"""CIFAR-10 ResNet-20 example for Uncertainty Baselines.

"""

import os.path

from absl import app
from absl import flags
from absl import logging

import numpy as np
import tensorflow as tf
import uncertainty_baselines as ub



# Flags relating to hyperparameters.
flags.DEFINE_integer('batch_size', 512, 'The training batch size.')
flags.DEFINE_integer('eval_batch_size', 100, 'The evaluation batch size.')
flags.DEFINE_string('optimizer', 'adam', 'The optimizer to train with.')
flags.DEFINE_float('learning_rate', 0.01, 'The learning rate.')
flags.DEFINE_float(
    'weight_decay',
    None,
    'The model decoupled weight decay rate.')

# Flags relating to setting up the job.
flags.DEFINE_bool('use_tpu', False, 'Whether to run on CPU or TPU.')
flags.DEFINE_string('tpu', '', 'Name of the TPU to use.')

# Flags relating to the training/eval loop.
flags.DEFINE_string('output_dir', None, 'Base output directory.')
flags.DEFINE_integer(
    'eval_frequency',
    100,
    'How many steps between evaluating on the (validation and) test set.')
flags.DEFINE_integer('train_steps', 2000, 'How many steps to train for.')
flags.DEFINE_integer('seed', 1337, 'Random seed.')


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


def run(trial_dir: str):
  """Run the experiment."""
  tf.random.set_seed(FLAGS.seed)
  np.random.seed(FLAGS.seed)
  strategy = ub.strategy_utils.get_strategy(FLAGS.tpu, FLAGS.use_tpu)
  with strategy.scope():
    # Setup CIFAR-10 tf.data.Dataset splits.
    dataset_builder = ub.datasets.Cifar10Dataset(
        batch_size=FLAGS.batch_size,
        eval_batch_size=FLAGS.eval_batch_size,
        validation_percent=0.1)  # Use 5000 validation images.
    train_dataset = ub.utils.build_dataset(
        dataset_builder, strategy, 'train', as_tuple=True)
    val_dataset = ub.utils.build_dataset(
        dataset_builder, strategy, 'validation', as_tuple=True)
    test_dataset = ub.utils.build_dataset(
        dataset_builder, strategy, 'test', as_tuple=True)

    # Setup optimizer.
    _check_batch_replica_divisible(FLAGS.batch_size, strategy)
    _check_batch_replica_divisible(FLAGS.eval_batch_size, strategy)
    optimizer = ub.optimizers.get(
        optimizer_name=FLAGS.optimizer,
        learning_rate_schedule='constant',
        learning_rate=FLAGS.learning_rate,
        weight_decay=FLAGS.weight_decay)

    # Setup model.
    model = ub.models.ResNet20Builder(
        batch_size=FLAGS.batch_size, l2_weight=None)
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['sparse_categorical_accuracy'])

    # Train and eval.
    steps_per_epoch = (
        dataset_builder.info['num_train_examples'] // FLAGS.batch_size)
    validation_steps = (
        dataset_builder.info['num_validation_examples'] //
        FLAGS.eval_batch_size)
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

    test_steps = (
        dataset_builder.info['num_test_examples'] //
        FLAGS.eval_batch_size)
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

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
r"""Entry point for Uncertainty Baselines.

"""

import os.path
from typing import Optional

from absl import app
from absl import flags
from absl import logging

import numpy as np
import tensorflow.compat.v2 as tf
import uncertainty_baselines as ub
import eval as eval_lib  # local file import
import flags as flags_lib  # local file import
import train as train_lib  # local file import
import uncertainty_metrics as um



FLAGS = flags.FLAGS


# TODO(znado): remove this and add padding to last batch.
def _check_batch_replica_divisible(
    batch_size: int,
    strategy: tf.distribute.Strategy):
  """Ensure the batch size is evenly divisible by the number of replicas."""
  if batch_size % strategy.num_replicas_in_sync != 0:
    raise ValueError(
        'Batch size must be evenly divisible by the number of replicas in the '
        'job. Batch size: {}, num replicas: {}'.format(
            batch_size, strategy.num_replicas_in_sync))


def _setup_trial_dir(trial_dir: str, flag_string: Optional[str], mode: str):
  # Do not write to the same flags.cfg file unless this is the trainer job.
  if not trial_dir or 'train' not in mode:
    return
  if not tf.io.gfile.exists(trial_dir):
    tf.io.gfile.makedirs(trial_dir)
  if flag_string:
    flags_filename = os.path.join(trial_dir, 'flags.cfg')
    with tf.io.gfile.GFile(flags_filename, 'w+') as flags_file:
      flags_file.write(flag_string)


def _maybe_setup_trial_dir(
    strategy,
    trial_dir: str,
    flag_string: Optional[str],
    mode: str):
  """Create `trial_dir` if it does not exist and save the flags if provided."""
  if trial_dir:
    logging.info('Saving to dir: %s', trial_dir)
  else:
    logging.warning('Not saving any experiment outputs!')
  if flag_string:
    logging.info('Running with flags:\n%s', flag_string)
  # Only write to the flags file on the first replica, otherwise can run into a
  # file writing error.
  if strategy.num_replicas_in_sync > 1:
    if strategy.cluster_resolver.task_id == 0:
      _setup_trial_dir(trial_dir, flag_string, mode)
  else:
    _setup_trial_dir(trial_dir, flag_string, mode)


class BrierScore(tf.keras.metrics.Mean):

  def update_state(self, y_true, y_pred, sample_weight=None):
    brier_score = um.brier_score(labels=y_true, probabilities=y_pred)
    super(BrierScore, self).update_state(brier_score)


def _get_hparams():
  """Get hyperparameter names and values to record in tensorboard."""
  hparams = {}
  possible_hparam_names = [
      name for name in FLAGS if name.startswith('optimizer_hparams_')
  ] + ['learning_rate', 'weight_decay']
  for name in possible_hparam_names:
    if FLAGS[name].default != FLAGS[name].value:
      if name.startswith('optimizer_hparams_'):
        hparam_name = name[len('optimizer_hparams_'):]
      else:
        hparam_name = name
      hparams[hparam_name] = FLAGS[name].value
  return hparams


def run(trial_dir: str, flag_string: Optional[str]):
  """Run the experiment.

  Args:
    trial_dir: String to the dir to write checkpoints to and read them from.
    flag_string: Optional string used to record what flags the job was run with.
  """
  tf.random.set_seed(FLAGS.seed)
  np.random.seed(FLAGS.seed)

  if not FLAGS.eval_frequency:
    FLAGS.eval_frequency = FLAGS.log_frequency

  if FLAGS.eval_frequency % FLAGS.log_frequency != 0:
    raise ValueError(
        'log_frequency ({}) must evenly divide eval_frequency '
        '({}).'.format(FLAGS.log_frequency, FLAGS.eval_frequency))

  strategy = ub.strategy_utils.get_strategy(FLAGS.master, FLAGS.use_tpu)
  with strategy.scope():
    _maybe_setup_trial_dir(strategy, trial_dir, flag_string, FLAGS.mode)

    # TODO(znado): pass all dataset and model kwargs.
    dataset_builder = ub.datasets.get(
        FLAGS.dataset_name,
        batch_size=FLAGS.batch_size,
        eval_batch_size=FLAGS.eval_batch_size,
        validation_percent=FLAGS.validation_percent,
        shuffle_buffer_size=FLAGS.shuffle_buffer_size)
    model = ub.models.get(
        FLAGS.model_name,
        batch_size=FLAGS.batch_size,
        num_motifs=FLAGS.num_motifs,
        len_motifs=FLAGS.len_motifs,
        num_denses=FLAGS.num_denses,
        l2_weight=FLAGS.l2_regularization,
        depth=FLAGS.wide_resnet_depth,
        width_multiplier=FLAGS.wide_resnet_width_multiplier,
        version=FLAGS.wide_resnet_version
        )

    metrics = {
        'accuracy': tf.keras.metrics.SparseCategoricalAccuracy(),
        'brier_score': BrierScore(name='brier_score'),
        'ece': um.ExpectedCalibrationError(num_bins=10, name='ece'),
        'loss': tf.keras.metrics.SparseCategoricalCrossentropy(),
    }

    # Record all non-default hparams in tensorboard.
    hparams = _get_hparams()

    if FLAGS.mode == 'eval':
      _check_batch_replica_divisible(FLAGS.eval_batch_size, strategy)
      eval_lib.run_eval_loop(
          dataset_builder=dataset_builder,
          model=model,
          trial_dir=trial_dir,
          train_steps=FLAGS.train_steps,
          strategy=strategy,
          metrics=metrics,
          checkpoint_step=FLAGS.checkpoint_step,
          hparams=hparams)
      return

    _check_batch_replica_divisible(FLAGS.batch_size, strategy)
    if FLAGS.mode == 'train_and_eval':
      _check_batch_replica_divisible(FLAGS.eval_batch_size, strategy)

    steps_per_epoch = (
        dataset_builder.info['num_train_examples'] // FLAGS.batch_size)
    optimizer_kwargs = {
        k[len('optimizer_hparams_'):]: FLAGS[k].value for k in FLAGS
        if k.startswith('optimizer_hparams_')
    }
    optimizer_kwargs.update({
        k[len('schedule_hparams_'):]: FLAGS[k].value for k in FLAGS
        if k.startswith('schedule_hparams_')
    })

    optimizer = ub.optimizers.get(
        optimizer_name=FLAGS.optimizer,
        learning_rate_schedule=FLAGS.learning_rate_schedule,
        learning_rate=FLAGS.learning_rate,
        weight_decay=FLAGS.weight_decay,
        steps_per_epoch=steps_per_epoch,
        model=model,
        **optimizer_kwargs)

    train_lib.run_train_loop(
        dataset_builder=dataset_builder,
        model=model,
        optimizer=optimizer,
        eval_frequency=FLAGS.eval_frequency,
        log_frequency=FLAGS.log_frequency,
        trial_dir=trial_dir,
        train_steps=FLAGS.train_steps,
        mode=FLAGS.mode,
        strategy=strategy,
        metrics=metrics,
        hparams=hparams)




def main(program_flag_names):
  logging.info(
      'Starting Uncertainty Baselines experiment %s', FLAGS.experiment_name)
  logging.info(
      '\n\nRun the following command to view outputs in tensorboard.dev:\n\n'
      'tensorboard dev upload --logdir %s --plugins scalars,graphs,hparams\n\n',
      FLAGS.output_dir)

  # TODO(znado): when open sourced tuning is supported, change this to include
  # the trial number.
  trial_dir = os.path.join(FLAGS.output_dir, '0')
  program_flags = {name: FLAGS[name].value for name in program_flag_names}
  flag_string = flags_lib.serialize_flags(program_flags)
  run(trial_dir, flag_string)


if __name__ == '__main__':
  defined_flag_names = flags_lib.define_flags()
  app.run(lambda _: main(defined_flag_names))

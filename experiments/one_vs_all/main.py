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
import uncertainty_baselines.experiments.one_vs_all.eval as eval_lib
import uncertainty_baselines.experiments.one_vs_all.flags as flags_lib
import uncertainty_baselines.experiments.one_vs_all.losses as loss_lib
import uncertainty_baselines.experiments.one_vs_all.models as models_lib
import uncertainty_baselines.experiments.one_vs_all.train as train_lib



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


def _setup_trial_dir(trial_dir: str, flag_string: Optional[str]):
  if not trial_dir:
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
    flag_string: Optional[str]):
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
      _setup_trial_dir(trial_dir, flag_string)
  else:
    _setup_trial_dir(trial_dir, flag_string)


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
    _maybe_setup_trial_dir(strategy, trial_dir, flag_string)

    # TODO(znado): pass all dataset and model kwargs.
    dataset_builder = ub.datasets.get(
        FLAGS.dataset_name,
        batch_size=FLAGS.batch_size,
        eval_batch_size=FLAGS.eval_batch_size,
        validation_percent=FLAGS.validation_percent)
    model = models_lib.create_model(
        batch_size=FLAGS.batch_size,
        num_classes=10,
        distance_logits=FLAGS.distance_logits)
    loss_fn = loss_lib.get(
        FLAGS.loss_name, from_logits=True, dm_alpha=FLAGS.dm_alpha)

    if FLAGS.mode == 'eval':
      _check_batch_replica_divisible(FLAGS.eval_batch_size, strategy)
      eval_lib.run_eval_loop(
          dataset_builder=dataset_builder,
          model=model,
          loss_fn=loss_fn,
          trial_dir=trial_dir,
          train_steps=FLAGS.train_steps,
          strategy=strategy,
          metric_names=['accuracy', 'loss'],
          checkpoint_step=FLAGS.checkpoint_step)
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
    optimizer = ub.optimizers.get(
        optimizer_name=FLAGS.optimizer,
        learning_rate_schedule=FLAGS.learning_rate_schedule,
        learning_rate=FLAGS.learning_rate,
        weight_decay=FLAGS.weight_decay,
        steps_per_epoch=steps_per_epoch,
        **optimizer_kwargs)

    train_lib.run_train_loop(
        dataset_builder=dataset_builder,
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        eval_frequency=FLAGS.eval_frequency,
        log_frequency=FLAGS.log_frequency,
        trial_dir=trial_dir,
        train_steps=FLAGS.train_steps,
        mode=FLAGS.mode,
        strategy=strategy,
        metric_names=['accuracy', 'loss'])




def main(program_flag_names):
  logging.info('Starting Uncertainty Baselines experiment!')
  logging.info(
      '\n\nRun the following command to view outputs in tensorboard.dev:\n\n'
      'tensorboard dev upload --logdir %s\n\n', FLAGS.model_dir)

  # TODO(znado): when open sourced tuning is supported, change this to include
  # the trial number.
  trial_dir = os.path.join(FLAGS.model_dir, '0')
  program_flags = {name: FLAGS[name].value for name in program_flag_names}
  flag_string = flags_lib.serialize_flags(program_flags)
  run(trial_dir, flag_string)


if __name__ == '__main__':
  defined_flag_names = flags_lib.define_flags()
  app.run(lambda _: main(defined_flag_names))

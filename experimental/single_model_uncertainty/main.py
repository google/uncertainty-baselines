# coding=utf-8
# Copyright 2024 The Uncertainty Baselines Authors.
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

# pylint: disable=line-too-long
r"""Entry point for Uncertainty Baselines.

"""
# pylint: enable=line-too-long

import os.path
from typing import Optional

from absl import app
from absl import flags
from absl import logging
import numpy as np
import robustness_metrics as rm
import tensorflow.compat.v2 as tf
import uncertainty_baselines as ub
import eval as eval_lib  # local file import from experimental.single_model_uncertainty
import flags as flags_lib  # local file import from experimental.single_model_uncertainty
import train as train_lib  # local file import from experimental.single_model_uncertainty
import models.models as ub_smu_models  # local file import from experimental.single_model_uncertainty

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
    if strategy.cluster_resolver is None or strategy.cluster_resolver.task_id == 0:
      _setup_trial_dir(trial_dir, flag_string)
  else:
    _setup_trial_dir(trial_dir, flag_string)


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

  use_tpu = not FLAGS.use_cpu and not FLAGS.use_gpu
  eval_drop_remainder = True if use_tpu else False

  if not FLAGS.eval_frequency:
    FLAGS.eval_frequency = FLAGS.log_frequency

  if FLAGS.eval_frequency % FLAGS.log_frequency != 0:
    raise ValueError(
        'log_frequency ({}) must evenly divide eval_frequency '
        '({}).'.format(FLAGS.log_frequency, FLAGS.eval_frequency))

  strategy = ub.strategy_utils.get_strategy(FLAGS.tpu, use_tpu=use_tpu)
  with strategy.scope():
    _maybe_setup_trial_dir(strategy, trial_dir, flag_string)

    # TODO(znado): pass all dataset and model kwargs.
    train_dataset_builder = ub.datasets.get(
        dataset_name=FLAGS.dataset_name,
        split='train',
        validation_percent=FLAGS.validation_percent,
        shuffle_buffer_size=FLAGS.shuffle_buffer_size)
    if FLAGS.validation_percent > 0:
      validation_dataset_builder = ub.datasets.get(
          dataset_name=FLAGS.dataset_name,
          split='validation',
          validation_percent=FLAGS.validation_percent,
          shuffle_buffer_size=FLAGS.shuffle_buffer_size,
          drop_remainder=eval_drop_remainder)
    else:
      validation_dataset_builder = None
    test_dataset_builder = ub.datasets.get(
        dataset_name=FLAGS.dataset_name,
        split='test',
        validation_percent=FLAGS.validation_percent,
        shuffle_buffer_size=FLAGS.shuffle_buffer_size,
        drop_remainder=eval_drop_remainder)

    if FLAGS.use_spec_norm:
      logging.info('Use spectral normalization.')
      spec_norm_hparams = {
          'spec_norm_bound': FLAGS.spec_norm_bound,
          'spec_norm_iteration': FLAGS.spec_norm_iteration
      }
    else:
      spec_norm_hparams = None

    if FLAGS.use_gp_layer:
      logging.info('Use GP for output layer.')
      gp_layer_hparams = {
          'gp_input_dim': FLAGS.gp_input_dim,
          'gp_hidden_dim': FLAGS.gp_hidden_dim,
          'gp_scale': FLAGS.gp_scale,
          'gp_bias': FLAGS.gp_bias,
          'gp_input_normalization': FLAGS.gp_input_normalization,
          'gp_cov_discount_factor': FLAGS.gp_cov_discount_factor,
          'gp_cov_ridge_penalty': FLAGS.gp_cov_ridge_penalty
      }
    else:
      gp_layer_hparams = None

    model = ub_smu_models.get(
        FLAGS.model_name,
        num_classes=FLAGS.num_classes,
        batch_size=FLAGS.batch_size,
        len_seqs=FLAGS.len_seqs,
        num_motifs=FLAGS.num_motifs,
        len_motifs=FLAGS.len_motifs,
        num_denses=FLAGS.num_denses,
        depth=FLAGS.wide_resnet_depth,
        width_multiplier=FLAGS.wide_resnet_width_multiplier,
        l2_weight=FLAGS.l2_regularization,
        dropout_rate=FLAGS.dropout_rate,
        before_conv_dropout=FLAGS.before_conv_dropout,
        use_mc_dropout=FLAGS.use_mc_dropout,
        spec_norm_hparams=spec_norm_hparams,
        gp_layer_hparams=gp_layer_hparams)

    metrics = {
        'accuracy': tf.keras.metrics.SparseCategoricalAccuracy(),
        'brier_score': rm.metrics.Brier(),
        'ece': rm.metrics.ExpectedCalibrationError(num_bins=FLAGS.num_bins),
        'loss': tf.keras.metrics.SparseCategoricalCrossentropy(),
    }

    # Record all non-default hparams in tensorboard.
    hparams = _get_hparams()

    ood_dataset_builder = None
    ood_metrics = None
    if FLAGS.run_ood:
      if 'cifar' in FLAGS.dataset_name and FLAGS.ood_dataset_name == 'svhn':
        svhn_normalize_by_cifar = True
      else:
        svhn_normalize_by_cifar = False

      ood_dataset_builder_cls = ub.datasets.DATASETS[FLAGS.ood_dataset_name]
      ood_dataset_builder_cls = ub.datasets.make_ood_dataset(
          ood_dataset_builder_cls)
      ood_dataset_builder = ood_dataset_builder_cls(
          in_distribution_dataset=test_dataset_builder,
          split='test',
          validation_percent=FLAGS.validation_percent,
          normalize_by_cifar=svhn_normalize_by_cifar,
          data_mode='ood',
          drop_remainder=eval_drop_remainder)
      _check_batch_replica_divisible(FLAGS.eval_batch_size, strategy)

      ood_metrics = {
          'auroc': tf.keras.metrics.AUC(
              curve='ROC', summation_method='interpolation'),
          'auprc': tf.keras.metrics.AUC(
              curve='PR', summation_method='interpolation')
      }

      aux_metrics = [
          ('spec_at_sen', tf.keras.metrics.SpecificityAtSensitivity,
           FLAGS.sensitivity_thresholds),
          ('sen_at_spec', tf.keras.metrics.SensitivityAtSpecificity,
           FLAGS.specificity_thresholds),
          ('prec_at_rec', tf.keras.metrics.PrecisionAtRecall,
           FLAGS.recall_thresholds),
          ('rec_at_prec', tf.keras.metrics.RecallAtPrecision,
           FLAGS.precision_thresholds)
      ]

      for metric_name, metric_fn, threshold_vals in aux_metrics:
        vals = [float(x) for x in threshold_vals]
        thresholds = np.linspace(vals[0], vals[1], int(vals[2]))
        for thresh in thresholds:
          name = f'{metric_name}_{thresh:.2f}'
          ood_metrics[name] = metric_fn(thresh)

    if FLAGS.mode == 'eval':
      _check_batch_replica_divisible(FLAGS.eval_batch_size, strategy)
      eval_lib.run_eval_loop(
          validation_dataset_builder=validation_dataset_builder,
          test_dataset_builder=test_dataset_builder,
          batch_size=FLAGS.eval_batch_size,
          model=model,
          trial_dir=trial_dir,
          train_steps=FLAGS.train_steps,
          strategy=strategy,
          metrics=metrics,
          checkpoint_step=FLAGS.checkpoint_step,
          hparams=hparams,
          ood_dataset_builder=ood_dataset_builder,
          ood_metrics=ood_metrics,
          mean_field_factor=FLAGS.gp_mean_field_factor,
          dempster_shafer_ood=FLAGS.dempster_shafer_ood)
      return

    if FLAGS.mode == 'train_and_eval':
      _check_batch_replica_divisible(FLAGS.eval_batch_size, strategy)

    steps_per_epoch = train_dataset_builder.num_examples // FLAGS.batch_size
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
        train_dataset_builder=train_dataset_builder,
        validation_dataset_builder=validation_dataset_builder,
        test_dataset_builder=test_dataset_builder,
        batch_size=FLAGS.batch_size,
        eval_batch_size=FLAGS.eval_batch_size,
        model=model,
        optimizer=optimizer,
        eval_frequency=FLAGS.eval_frequency,
        log_frequency=FLAGS.log_frequency,
        trial_dir=trial_dir,
        train_steps=FLAGS.train_steps,
        mode=FLAGS.mode,
        strategy=strategy,
        metrics=metrics,
        hparams=hparams,
        ood_dataset_builder=ood_dataset_builder,
        ood_metrics=ood_metrics,
        focal_loss_gamma=FLAGS.focal_loss_gamma,
        mean_field_factor=FLAGS.gp_mean_field_factor)




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

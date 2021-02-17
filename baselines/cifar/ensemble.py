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

"""Ensemble on CIFAR.

This script only performs evaluation, not training. We recommend training
ensembles by launching independent runs of `deterministic.py` over different
seeds.
"""

import os

from absl import app
from absl import flags
from absl import logging

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import uncertainty_baselines as ub
import utils  # local file import
import uncertainty_metrics as um

flags.DEFINE_string('checkpoint_dir', None,
                    'The directory where the model weights are stored.')
flags.mark_flag_as_required('checkpoint_dir')
flags.DEFINE_integer('seed', 42, 'Random seed.')
flags.DEFINE_integer('per_core_batch_size', 64, 'Batch size per TPU core/GPU.')
flags.DEFINE_enum('dataset', 'cifar10',
                  enum_values=['cifar10', 'cifar100'],
                  help='Dataset.')
flags.DEFINE_string('cifar100_c_path', None,
                    'Path to the TFRecords files for CIFAR-100-C. Only valid '
                    '(and required) if dataset is cifar100 and corruptions.')
flags.DEFINE_integer('num_bins', 15, 'Number of bins for ECE.')
flags.DEFINE_string('output_dir', '/tmp/cifar', 'Output directory.')

# Accelerator flags.
flags.DEFINE_bool('use_gpu', False, 'Whether to run on GPU or otherwise TPU.')
flags.DEFINE_integer('num_cores', 8, 'Number of TPU cores or number of GPUs.')
FLAGS = flags.FLAGS


def parse_checkpoint_dir(checkpoint_dir):
  """Parse directory of checkpoints."""
  paths = []
  subdirectories = tf.io.gfile.glob(os.path.join(checkpoint_dir, '*'))
  is_checkpoint = lambda f: ('checkpoint' in f and '.index' in f)
  for subdir in subdirectories:
    for path, _, files in tf.io.gfile.walk(subdir):
      if any(f for f in files if is_checkpoint(f)):
        latest_checkpoint_without_suffix = tf.train.latest_checkpoint(path)
        paths.append(os.path.join(path, latest_checkpoint_without_suffix))
        break
  return paths


def main(argv):
  del argv  # unused arg
  if not FLAGS.use_gpu:
    raise ValueError('Only GPU is currently supported.')
  if FLAGS.num_cores > 1:
    raise ValueError('Only a single accelerator is currently supported.')
  tf.random.set_seed(FLAGS.seed)
  tf.io.gfile.makedirs(FLAGS.output_dir)

  ds_info = tfds.builder(FLAGS.dataset).info
  batch_size = FLAGS.per_core_batch_size * FLAGS.num_cores
  steps_per_eval = ds_info.splits['test'].num_examples // batch_size
  num_classes = ds_info.features['label'].num_classes

  dataset = ub.datasets.get(
      FLAGS.dataset,
      split=tfds.Split.TEST).load(batch_size=batch_size)
  test_datasets = {'clean': dataset}
  extra_kwargs = {}
  if FLAGS.dataset == 'cifar100':
    extra_kwargs['data_dir'] = FLAGS.cifar100_c_path
  corruption_types, _ = utils.load_corrupted_test_info(FLAGS.dataset)
  for corruption_type in corruption_types:
    for severity in range(1, 6):
      dataset = ub.datasets.get(
          f'{FLAGS.dataset}_corrupted',
          corruption_type=corruption_type,
          severity=severity,
          split=tfds.Split.TEST,
          **extra_kwargs).load(batch_size=batch_size)
      test_datasets[f'{corruption_type}_{severity}'] = dataset

  model = ub.models.wide_resnet(
      input_shape=ds_info.features['image'].shape,
      depth=28,
      width_multiplier=10,
      num_classes=num_classes,
      l2=0.,
      version=2)
  logging.info('Model input shape: %s', model.input_shape)
  logging.info('Model output shape: %s', model.output_shape)
  logging.info('Model number of weights: %s', model.count_params())

  # Search for checkpoints from their index file; then remove the index suffix.
  ensemble_filenames = parse_checkpoint_dir(FLAGS.checkpoint_dir)
  ensemble_size = len(ensemble_filenames)
  logging.info('Ensemble size: %s', ensemble_size)
  logging.info('Ensemble number of weights: %s',
               ensemble_size * model.count_params())
  logging.info('Ensemble filenames: %s', str(ensemble_filenames))
  checkpoint = tf.train.Checkpoint(model=model)

  # Write model predictions to files.
  num_datasets = len(test_datasets)
  for m, ensemble_filename in enumerate(ensemble_filenames):
    checkpoint.restore(ensemble_filename)
    for n, (name, test_dataset) in enumerate(test_datasets.items()):
      filename = '{dataset}_{member}.npy'.format(dataset=name, member=m)
      filename = os.path.join(FLAGS.output_dir, filename)
      if not tf.io.gfile.exists(filename):
        logits = []
        test_iterator = iter(test_dataset)
        for _ in range(steps_per_eval):
          features = next(test_iterator)['features']  # pytype: disable=unsupported-operands
          logits.append(model(features, training=False))

        logits = tf.concat(logits, axis=0)
        with tf.io.gfile.GFile(filename, 'w') as f:
          np.save(f, logits.numpy())
      percent = (m * num_datasets + (n + 1)) / (ensemble_size * num_datasets)
      message = ('{:.1%} completion for prediction: ensemble member {:d}/{:d}. '
                 'Dataset {:d}/{:d}'.format(percent,
                                            m + 1,
                                            ensemble_size,
                                            n + 1,
                                            num_datasets))
      logging.info(message)

  metrics = {
      'test/negative_log_likelihood': tf.keras.metrics.Mean(),
      'test/gibbs_cross_entropy': tf.keras.metrics.Mean(),
      'test/accuracy': tf.keras.metrics.SparseCategoricalAccuracy(),
      'test/ece': um.ExpectedCalibrationError(num_bins=FLAGS.num_bins),
  }
  corrupt_metrics = {}
  for name in test_datasets:
    corrupt_metrics['test/nll_{}'.format(name)] = tf.keras.metrics.Mean()
    corrupt_metrics['test/accuracy_{}'.format(name)] = (
        tf.keras.metrics.SparseCategoricalAccuracy())
    corrupt_metrics['test/ece_{}'.format(name)] = (
        um.ExpectedCalibrationError(num_bins=FLAGS.num_bins))
  for i in range(ensemble_size):
    metrics['test/nll_member_{}'.format(i)] = tf.keras.metrics.Mean()
    metrics['test/accuracy_member_{}'.format(i)] = (
        tf.keras.metrics.SparseCategoricalAccuracy())
  test_diversity = {
      'test/disagreement': tf.keras.metrics.Mean(),
      'test/average_kl': tf.keras.metrics.Mean(),
      'test/cosine_similarity': tf.keras.metrics.Mean(),
  }
  metrics.update(test_diversity)

  # Evaluate model predictions.
  for n, (name, test_dataset) in enumerate(test_datasets.items()):
    logits_dataset = []
    for m in range(ensemble_size):
      filename = '{dataset}_{member}.npy'.format(dataset=name, member=m)
      filename = os.path.join(FLAGS.output_dir, filename)
      with tf.io.gfile.GFile(filename, 'rb') as f:
        logits_dataset.append(np.load(f))

    logits_dataset = tf.convert_to_tensor(logits_dataset)
    test_iterator = iter(test_dataset)
    for step in range(steps_per_eval):
      labels = next(test_iterator)['labels']  # pytype: disable=unsupported-operands
      logits = logits_dataset[:, (step*batch_size):((step+1)*batch_size)]
      labels = tf.cast(labels, tf.int32)
      negative_log_likelihood = um.ensemble_cross_entropy(labels, logits)
      per_probs = tf.nn.softmax(logits)
      probs = tf.reduce_mean(per_probs, axis=0)
      if name == 'clean':
        gibbs_ce = um.gibbs_cross_entropy(labels, logits)
        metrics['test/negative_log_likelihood'].update_state(
            negative_log_likelihood)
        metrics['test/gibbs_cross_entropy'].update_state(gibbs_ce)
        metrics['test/accuracy'].update_state(labels, probs)
        metrics['test/ece'].update_state(labels, probs)

        for i in range(ensemble_size):
          member_probs = per_probs[i]
          member_loss = tf.keras.losses.sparse_categorical_crossentropy(
              labels, member_probs)
          metrics['test/nll_member_{}'.format(i)].update_state(member_loss)
          metrics['test/accuracy_member_{}'.format(i)].update_state(
              labels, member_probs)
        diversity_results = um.average_pairwise_diversity(
            per_probs, ensemble_size)
        for k, v in diversity_results.items():
          test_diversity['test/' + k].update_state(v)
      else:
        corrupt_metrics['test/nll_{}'.format(name)].update_state(
            negative_log_likelihood)
        corrupt_metrics['test/accuracy_{}'.format(name)].update_state(
            labels, probs)
        corrupt_metrics['test/ece_{}'.format(name)].update_state(
            labels, probs)

    message = ('{:.1%} completion for evaluation: dataset {:d}/{:d}'.format(
        (n + 1) / num_datasets, n + 1, num_datasets))
    logging.info(message)

  corrupt_results = utils.aggregate_corrupt_metrics(corrupt_metrics,
                                                    corruption_types)
  total_results = {name: metric.result() for name, metric in metrics.items()}
  total_results.update(corrupt_results)
  logging.info('Metrics: %s', total_results)


if __name__ == '__main__':
  app.run(main)

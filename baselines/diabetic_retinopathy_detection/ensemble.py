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

"""Ensemble on ResNet50.

This script only performs evaluation, not training. We recommend training
ensembles by launching independent runs of `deterministic.py` over different
seeds.
"""

import functools
import os

from absl import app
from absl import flags
from absl import logging

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import uncertainty_baselines as ub
import utils  # local file import
import uncertainty_metrics.tensorflow as um

# Data load / output flags.
flags.DEFINE_string(
  'checkpoint_dir', '/tmp/diabetic_retinopathy_detection/deterministic',
  'The directory from which the trained deterministic '
  'model weights are retrieved.')
flags.DEFINE_string(
  'output_dir', '/tmp/diabetic_retinopathy_detection/ensemble',
  'The directory where the ensemble model weights '
  'and training/evaluation summaries are stored.')
flags.DEFINE_string(
  'data_dir', None,
  'Path to training and testing data.')
flags.mark_flag_as_required('data_dir')

# General model flags.
flags.DEFINE_integer('seed', 42, 'Random seed.')
flags.DEFINE_integer('eval_batch_size', 32,
                     'Eval batch size per TPU core/GPU.')
flags.DEFINE_integer('num_bins', 15, 'Number of bins for ECE.')

# Accelerator flags.
flags.DEFINE_bool('force_use_cpu', False, 'If True, force usage of CPU')
flags.DEFINE_bool('use_gpu', True, 'Whether to run on GPU.')
flags.DEFINE_bool('use_bfloat16', False, 'Whether to use mixed precision.')
flags.DEFINE_integer(
  'num_cores', 1,
  'Number of TPU cores or number of GPUs - only support 1 GPU for now.')
FLAGS = flags.FLAGS


def parse_checkpoint_dir(checkpoint_dir):
  """Parse directory of checkpoints."""
  paths = []
  subdirectories = tf.io.gfile.glob(checkpoint_dir)
  is_checkpoint = lambda f: ('checkpoint' in f and '.index' in f)
  for subdir in subdirectories:
    for path, _, files in tf.io.gfile.walk(subdir):
      if any(f for f in files if is_checkpoint(f)):
        latest_checkpoint_without_suffix = tf.train.latest_checkpoint(path)
        paths.append(os.path.join(path, latest_checkpoint_without_suffix))
  return paths


def main(argv):
  del argv  # unused arg
  if not FLAGS.use_gpu:
    raise ValueError('Only GPU is currently supported.')
  if FLAGS.num_cores > 1:
    raise ValueError('Only a single accelerator is currently supported.')
  tf.random.set_seed(FLAGS.seed)
  tf.io.gfile.makedirs(FLAGS.output_dir)

  if FLAGS.force_use_cpu:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
  elif FLAGS.use_gpu:
    logging.info('Use GPU')

  ds_info = tfds.builder('diabetic_retinopathy_detection').info
  batch_size = FLAGS.eval_batch_size * FLAGS.num_cores
  steps_per_eval = ds_info.splits['test'].num_examples // batch_size

  dataset_test_builder = ub.datasets.get(
    "diabetic_retinopathy_detection",
    split='test',
    data_dir=FLAGS.data_dir)
  dataset_test = dataset_test_builder.load(batch_size=FLAGS.eval_batch_size)

  if FLAGS.use_bfloat16:
    policy = tf.keras.mixed_precision.experimental.Policy('mixed_bfloat16')
    tf.keras.mixed_precision.experimental.set_policy(policy)

  summary_writer = tf.summary.create_file_writer(
      os.path.join(FLAGS.output_dir, 'summaries'))

  logging.info('Building Keras ResNet-50 model')

  # Shape tuple access depends on number of distributed devices
  try:
    shape_tuple = dataset_test.element_spec['features'].shape
  except AttributeError:  # Multiple TensorSpec in a (nested) PerReplicaSpec.
    tensor_spec_list = dataset_test.element_spec[  # pylint: disable=protected-access
      'features']._flat_tensor_specs
    shape_tuple = tensor_spec_list[0].shape

  model = ub.models.resnet50_deterministic(
    input_shape=shape_tuple.as_list()[1:],
    num_classes=1)  # binary classification task

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
  for member, ensemble_filename in enumerate(ensemble_filenames):
    checkpoint.restore(ensemble_filename)
    filename = f'{member}.npy'
    filename = os.path.join(FLAGS.output_dir, filename)
    if not tf.io.gfile.exists(filename):
      logits = []
      test_iterator = iter(dataset_test)
      for i in range(steps_per_eval):
        inputs = next(test_iterator)  # pytype: disable=attribute-error
        images = inputs['features']
        logits.append(model(images, training=False))
        if i % 100 == 0:
          logging.info(
            f'Ensemble member {member + 1}/{ensemble_size}: '
            f'Completed {i + 1} of {steps_per_eval} eval steps.')

      logits = tf.concat(logits, axis=0)
      with tf.io.gfile.GFile(filename, 'w') as f:
        np.save(f, logits.numpy())
    percent = member / ensemble_size
    message = (
      '{:.1%} completion for prediction: ensemble member {:d}/{:d}.'.format(
        percent, member + 1, ensemble_size))
    logging.info(message)

  metrics = {
      'test/negative_log_likelihood': tf.keras.metrics.Mean(),
      'test/gibbs_cross_entropy': tf.keras.metrics.Mean(),
      'test/accuracy': tf.keras.metrics.BinaryAccuracy(),
      'test/ece': um.ExpectedCalibrationError(num_bins=FLAGS.num_bins),
      'test/auc': tf.keras.metrics.AUC(),
  }

  for i in range(ensemble_size):
    metrics['test/nll_member_{}'.format(i)] = tf.keras.metrics.Mean()
    metrics['test/accuracy_member_{}'.format(i)] = (
        tf.keras.metrics.BinaryAccuracy())
  test_diversity = {
      'test/disagreement': tf.keras.metrics.Mean(),
      'test/average_kl': tf.keras.metrics.Mean(),
      'test/cosine_similarity': tf.keras.metrics.Mean(),
  }
  metrics.update(test_diversity)

  # Evaluate model predictions.
  logits_dataset = []
  for member in range(ensemble_size):
    filename = f'{member}.npy'
    filename = os.path.join(FLAGS.output_dir, filename)
    with tf.io.gfile.GFile(filename, 'rb') as f:
      logits_dataset.append(np.load(f))

  logits_dataset = tf.convert_to_tensor(logits_dataset)
  test_iterator = iter(dataset_test)

  for step in range(steps_per_eval):
    inputs = next(test_iterator)  # pytype: disable=attribute-error
    labels = inputs['labels']
    logits = logits_dataset[:, (step*batch_size):((step+1)*batch_size)]
    labels = tf.cast(labels, tf.float32)
    logits = tf.cast(logits, tf.float32)
    negative_log_likelihood = um.ensemble_cross_entropy(
      tf.expand_dims(labels, axis=-1), logits, binary=True)
    per_probs = tf.nn.sigmoid(logits)
    probs = tf.reduce_mean(per_probs, axis=0)
    gibbs_ce = um.gibbs_cross_entropy(
      tf.expand_dims(labels, axis=-1), logits, binary=True)
    metrics['test/negative_log_likelihood'].update_state(
        negative_log_likelihood)
    metrics['test/gibbs_cross_entropy'].update_state(gibbs_ce)
    metrics['test/accuracy'].update_state(labels, probs)
    metrics['test/ece'].update_state(labels, probs)
    metrics['test/auc'].update_state(labels, probs)

    for i in range(ensemble_size):
      member_probs = per_probs[i]
      member_loss = tf.keras.losses.binary_crossentropy(
          labels, member_probs)
      metrics['test/nll_member_{}'.format(i)].update_state(member_loss)
      metrics['test/accuracy_member_{}'.format(i)].update_state(
          labels, member_probs)
    diversity_results = um.average_pairwise_diversity(
        per_probs, ensemble_size)
    for k, v in diversity_results.items():
      test_diversity['test/' + k].update_state(v)

  total_results = {name: metric.result() for name, metric in metrics.items()}
  logging.info('Metrics: %s', total_results)


if __name__ == '__main__':
  app.run(main)

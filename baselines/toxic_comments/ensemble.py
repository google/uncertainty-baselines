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

"""Ensemble on Toxic Comments Detection.

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
import bert_utils  # local file import
import deterministic_model_bert as bert_model  # local file import
# import toxic_comments.deterministic to inherit its flags
import deterministic  # pylint:disable=unused-import  # local file import
from uncertainty_baselines.datasets import base
from uncertainty_baselines.datasets import toxic_comments as ds
import uncertainty_metrics as um


# TODO(trandustin): We inherit
# FLAGS.{dataset,per_core_batch_size,output_dir,seed} from deterministic. This
# is not intuitive, which suggests we need to either refactor to avoid importing
# from a binary or duplicate the model definition here.

# Model flags
flags.DEFINE_string('checkpoint_dir', None,
                    'The directory where the model weights are stored.')
flags.DEFINE_integer('num_models', 10, 'Number of models to be included '
                                       'in the ensemble')
flags.mark_flag_as_required('checkpoint_dir')
flags.mark_flag_as_required('num_models')
FLAGS = flags.FLAGS

_MAX_SEQ_LENGTH = 512


def main(argv):
  del argv  # unused arg
  if not FLAGS.use_gpu:
    raise ValueError('Only GPU is currently supported.')
  if FLAGS.num_cores > 1:
    raise ValueError('Only a single accelerator is currently supported.')

  tf.random.set_seed(FLAGS.seed)
  logging.info('Model checkpoint will be saved at %s', FLAGS.output_dir)
  tf.io.gfile.makedirs(FLAGS.output_dir)

  batch_size = FLAGS.per_core_batch_size * FLAGS.num_cores
  test_batch_size = batch_size
  data_buffer_size = batch_size * 10

  ind_dataset_builder = ds.WikipediaToxicityDataset(
      batch_size=batch_size,
      eval_batch_size=test_batch_size,
      bert_dataset_dir=FLAGS.in_dataset_dir,
      shuffle_buffer_size=data_buffer_size)
  ood_dataset_builder = ds.CivilCommentsDataset(
      batch_size=batch_size,
      eval_batch_size=test_batch_size,
      bert_dataset_dir=FLAGS.ood_dataset_dir,
      shuffle_buffer_size=data_buffer_size)

  dataset_builders = {
      'ind': ind_dataset_builder,
      'ood': ood_dataset_builder,
  }

  ds_info = ind_dataset_builder.info
  feature_size = _MAX_SEQ_LENGTH
  num_classes = ds_info['num_classes']  # Positive and negative classes.

  test_datasets = {}
  steps_per_eval = {}
  for dataset_name, dataset_builder in dataset_builders.items():
    test_datasets[dataset_name] = dataset_builder.build(
        split=base.Split.TEST)
    steps_per_eval[dataset_name] = (
        dataset_builder.info['num_test_examples'] // test_batch_size)

  logging.info('Building %s model', FLAGS.model_family)

  bert_config_dir, bert_ckpt_dir = deterministic.resolve_bert_ckpt_and_config_dir(
      FLAGS.bert_dir, FLAGS.bert_config_dir, FLAGS.bert_ckpt_dir)
  bert_config = bert_utils.create_config(bert_config_dir)
  model, bert_encoder = bert_model.create_model(
      num_classes=num_classes,
      feature_size=feature_size,
      bert_config=bert_config)

  logging.info('Model input shape: %s', model.input_shape)
  logging.info('Model output shape: %s', model.output_shape)
  logging.info('Model number of weights: %s', model.count_params())

  # Search for checkpoints from their index file; then remove the index suffix.
  ensemble_filenames = tf.io.gfile.glob(
      os.path.join(FLAGS.checkpoint_dir, '**/*.index'))
  ensemble_filenames = [filename[:-6] for filename in ensemble_filenames]
  if FLAGS.num_models > len(ensemble_filenames):
    raise ValueError('Number of models to be included in the ensemble '
                     'should be less than total number of models in '
                     'the checkpoint_dir.')
  ensemble_filenames = ensemble_filenames[:FLAGS.num_models]
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
        for step in range(steps_per_eval[name]):
          try:
            inputs = next(test_iterator)
          except StopIteration:
            continue
          features, labels, _ = deterministic.create_feature_and_label(inputs)
          logits.append(model(features, training=False))

        logits = tf.concat(logits, axis=0)
        with tf.io.gfile.GFile(filename, 'w') as f:
          np.save(f, logits.numpy())
      percent = (m * num_datasets + (n + 1)) / (ensemble_size * num_datasets)
      message = ('{:.1%} completion for prediction: ensemble member {:d}/{:d}. '
                 'Dataset {:d}/{:d}'.format(percent, m + 1, ensemble_size,
                                            n + 1, num_datasets))
      logging.info(message)

  metrics = {
      'test/negative_log_likelihood': tf.keras.metrics.Mean(),
      'test/auroc': tf.keras.metrics.AUC(curve='ROC'),
      'test/aupr': tf.keras.metrics.AUC(curve='PR'),
      'test/brier': tf.keras.metrics.MeanSquaredError(),
      'test/ece': um.ExpectedCalibrationError(num_bins=FLAGS.num_bins),
  }
  for fraction in FLAGS.fractions:
    metrics.update({
        'test_collab_acc/collab_acc_{}'.format(fraction):
            um.OracleCollaborativeAccuracy(
                fraction=float(fraction), num_bins=FLAGS.num_bins)
    })
  for dataset_name, test_dataset in test_datasets.items():
    if dataset_name != 'ind':
      metrics.update({
          'test/nll_{}'.format(dataset_name):
              tf.keras.metrics.Mean(),
          'test/auroc_{}'.format(dataset_name):
              tf.keras.metrics.AUC(curve='ROC'),
          'test/aupr_{}'.format(dataset_name):
              tf.keras.metrics.AUC(curve='PR'),
          'test/brier_{}'.format(dataset_name):
              tf.keras.metrics.MeanSquaredError(),
          'test/ece_{}'.format(dataset_name):
              um.ExpectedCalibrationError(num_bins=FLAGS.num_bins),
      })
      for fraction in FLAGS.fractions:
        metrics.update({
            'test_collab_acc/collab_acc_{}_{}'.format(fraction, dataset_name):
                um.OracleCollaborativeAccuracy(
                    fraction=float(fraction), num_bins=FLAGS.num_bins)
        })

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
    for step in range(steps_per_eval[name]):
      try:
        inputs = next(test_iterator)
      except StopIteration:
        continue
      features, labels, _ = deterministic.create_feature_and_label(inputs)
      logits = logits_dataset[:, (step * batch_size):((step + 1) * batch_size)]
      loss_logits = tf.squeeze(logits, axis=-1)
      negative_log_likelihood = um.ensemble_cross_entropy(
          labels, loss_logits, binary=True)

      per_probs = tf.nn.sigmoid(logits)
      probs = tf.reduce_mean(per_probs, axis=0)
      # Cast labels to discrete for ECE computation
      ece_labels = tf.cast(labels > FLAGS.ece_label_threshold, tf.float32)
      ece_probs = tf.concat([1. - probs, probs], axis=1)
      auc_probs = tf.squeeze(probs, axis=1)

      if name == 'ind':
        metrics['test/negative_log_likelihood'].update_state(
            negative_log_likelihood)
        metrics['test/auroc'].update_state(labels, auc_probs)
        metrics['test/aupr'].update_state(labels, auc_probs)
        metrics['test/brier'].update_state(labels, auc_probs)
        metrics['test/ece'].update_state(ece_labels, ece_probs)
        for fraction in FLAGS.fractions:
          metrics['test_collab_acc/collab_acc_{}'.format(
              fraction)].update_state(ece_labels, ece_probs)
      else:
        metrics['test/nll_{}'.format(dataset_name)].update_state(
            negative_log_likelihood)
        metrics['test/auroc_{}'.format(dataset_name)].update_state(
            labels, auc_probs)
        metrics['test/aupr_{}'.format(dataset_name)].update_state(
            labels, auc_probs)
        metrics['test/brier_{}'.format(dataset_name)].update_state(
            labels, auc_probs)
        metrics['test/ece_{}'.format(dataset_name)].update_state(
            ece_labels, ece_probs)
        for fraction in FLAGS.fractions:
          metrics['test_collab_acc/collab_acc_{}_{}'.format(
              fraction, dataset_name)].update_state(ece_labels, ece_probs)

    message = ('{:.1%} completion for evaluation: dataset {:d}/{:d}'.format(
        (n + 1) / num_datasets, n + 1, num_datasets))
    logging.info(message)

  total_results = {name: metric.result() for name, metric in metrics.items()}
  logging.info('Metrics: %s', total_results)


if __name__ == '__main__':
  app.run(main)

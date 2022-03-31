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

"""Ensemble on Toxic Comments Detection.

This script only performs evaluation, not training. We recommend training
ensembles by launching independent runs of `deterministic.py` over different
seeds.
"""

import collections
import os
from typing import Dict

from absl import app
from absl import flags
from absl import logging
import numpy as np
import robustness_metrics as rm
import tensorflow as tf
from tensorflow_addons import metrics as tfa_metrics

import uncertainty_baselines as ub
# import toxic_comments.deterministic to inherit its flags
import deterministic  # pylint:disable=unused-import  # local file import from baselines.toxic_comments
import metrics as tc_metrics  # local file import from baselines.toxic_comments
import utils  # local file import from baselines.toxic_comments
from uncertainty_baselines.datasets import toxic_comments as ds


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
      split='test',
      data_dir=FLAGS.in_dataset_dir if FLAGS.use_local_data else None,
      shuffle_buffer_size=data_buffer_size,
      tf_hub_preprocessor_url=FLAGS.bert_tokenizer_tf_hub_url)
  ood_dataset_builder = ds.CivilCommentsDataset(
      split='test',
      data_dir=FLAGS.ood_dataset_dir if FLAGS.use_local_data else None,
      shuffle_buffer_size=data_buffer_size,
      tf_hub_preprocessor_url=FLAGS.bert_tokenizer_tf_hub_url)
  ood_identity_dataset_builder = ds.CivilCommentsIdentitiesDataset(
      split='test',
      data_dir=FLAGS.identity_dataset_dir if FLAGS.use_local_data else None,
      shuffle_buffer_size=data_buffer_size,
      tf_hub_preprocessor_url=FLAGS.bert_tokenizer_tf_hub_url)

  test_dataset_builders = {
      'ind': ind_dataset_builder,
      'ood': ood_dataset_builder,
      'ood_identity': ood_identity_dataset_builder,
  }
  if FLAGS.prediction_mode and FLAGS.identity_prediction:
    for dataset_name in utils.IDENTITY_LABELS:
      if utils.NUM_EXAMPLES[dataset_name]['test'] > 100:
        test_dataset_builders[dataset_name] = ds.CivilCommentsIdentitiesDataset(
            split='test',
            data_dir=(os.path.join(FLAGS.identity_specific_dataset_dir,
                                   dataset_name)
                      if FLAGS.use_local_data else None),
            shuffle_buffer_size=data_buffer_size,
            tf_hub_preprocessor_url=FLAGS.bert_tokenizer_tf_hub_url)
    for dataset_name in utils.IDENTITY_TYPES:
      if utils.NUM_EXAMPLES[dataset_name]['test'] > 100:
        test_dataset_builders[dataset_name] = ds.CivilCommentsIdentitiesDataset(
            split='test',
            data_dir=(os.path.join(FLAGS.identity_type_dataset_dir,
                                   dataset_name)
                      if FLAGS.use_local_data else None),
            shuffle_buffer_size=data_buffer_size,
            tf_hub_preprocessor_url=FLAGS.bert_tokenizer_tf_hub_url)

  class_weight = utils.create_class_weight(
      test_dataset_builders=test_dataset_builders)
  logging.info('class_weight: %s', str(class_weight))

  ds_info = ind_dataset_builder.tfds_info
  feature_size = _MAX_SEQ_LENGTH
  # Positive and negative classes.
  num_classes = ds_info.metadata['num_classes']

  test_datasets = {}
  steps_per_eval = {}
  for dataset_name, dataset_builder in test_dataset_builders.items():
    test_datasets[dataset_name] = dataset_builder.load(
        batch_size=test_batch_size)
    if dataset_name in ['ind', 'ood', 'ood_identity']:
      steps_per_eval[dataset_name] = (
          dataset_builder.num_examples // test_batch_size)
    else:
      steps_per_eval[dataset_name] = (
          utils.NUM_EXAMPLES[dataset_name]['test'] // test_batch_size)

  logging.info('Building %s model', FLAGS.model_family)

  bert_config_dir, _ = utils.resolve_bert_ckpt_and_config_dir(
      FLAGS.bert_model_type, FLAGS.bert_dir, FLAGS.bert_config_dir,
      FLAGS.bert_ckpt_dir)
  bert_config = utils.create_config(bert_config_dir)
  model, _ = ub.models.bert_model(
      num_classes=num_classes,
      max_seq_length=feature_size,
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
    checkpoint.restore(ensemble_filename).assert_existing_objects_matched()
    for n, (dataset_name, test_dataset) in enumerate(test_datasets.items()):
      filename = '{dataset}_{member}.npy'.format(dataset=dataset_name, member=m)
      filename = os.path.join(FLAGS.output_dir, filename)
      if not tf.io.gfile.exists(filename):
        logits = []
        test_iterator = iter(test_dataset)
        for step in range(steps_per_eval[dataset_name]):
          try:
            inputs = next(test_iterator)
          except StopIteration:
            continue
          features, labels, _ = utils.create_feature_and_label(inputs)
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
      'test/negative_log_likelihood':
          tf.keras.metrics.Mean(),
      'test/auroc':
          tf.keras.metrics.AUC(curve='ROC'),
      'test/aupr':
          tf.keras.metrics.AUC(curve='PR'),
      'test/brier':
          tf.keras.metrics.MeanSquaredError(),
      'test/brier_weighted':
          tf.keras.metrics.MeanSquaredError(),
      'test/ece':
          rm.metrics.ExpectedCalibrationError(num_bins=FLAGS.num_ece_bins),
      'test/acc':
          tf.keras.metrics.Accuracy(),
      'test/acc_weighted':
          tf.keras.metrics.Accuracy(),
      'test/precision':
          tf.keras.metrics.Precision(),
      'test/recall':
          tf.keras.metrics.Recall(),
      'test/f1':
          tfa_metrics.F1Score(
              num_classes=num_classes,
              average='micro',
              threshold=FLAGS.ece_label_threshold)
  }

  for policy in ('uncertainty', 'toxicity'):
    metrics.update({
        'test_{}/calibration_auroc'.format(policy):
            tc_metrics.CalibrationAUC(curve='ROC'),
        'test_{}/calibration_auprc'.format(policy):
            tc_metrics.CalibrationAUC(curve='PR')
    })

    for fraction in FLAGS.fractions:
      metrics.update({
          'test_{}/collab_acc_{}'.format(policy, fraction):
              rm.metrics.OracleCollaborativeAccuracy(
                  fraction=float(fraction), num_bins=FLAGS.num_approx_bins),
          'test_{}/abstain_prec_{}'.format(policy, fraction):
              tc_metrics.AbstainPrecision(
                  abstain_fraction=float(fraction),
                  num_approx_bins=FLAGS.num_approx_bins),
          'test_{}/abstain_recall_{}'.format(policy, fraction):
              tc_metrics.AbstainRecall(
                  abstain_fraction=float(fraction),
                  num_approx_bins=FLAGS.num_approx_bins),
          'test_{}/collab_auroc_{}'.format(policy, fraction):
              tc_metrics.OracleCollaborativeAUC(
                  oracle_fraction=float(fraction),
                  num_bins=FLAGS.num_approx_bins),
          'test_{}/collab_auprc_{}'.format(policy, fraction):
              tc_metrics.OracleCollaborativeAUC(
                  oracle_fraction=float(fraction),
                  curve='PR',
                  num_bins=FLAGS.num_approx_bins),
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
          'test/brier_weighted_{}'.format(dataset_name):
              tf.keras.metrics.MeanSquaredError(),
          'test/ece_{}'.format(dataset_name):
              rm.metrics.ExpectedCalibrationError(num_bins=FLAGS.num_ece_bins),
          'test/acc_weighted_{}'.format(dataset_name):
              tf.keras.metrics.Accuracy(),
          'test/acc_{}'.format(dataset_name):
              tf.keras.metrics.Accuracy(),
          'test/precision_{}'.format(dataset_name):
              tf.keras.metrics.Precision(),
          'test/recall_{}'.format(dataset_name):
              tf.keras.metrics.Recall(),
          'test/f1_{}'.format(dataset_name):
              tfa_metrics.F1Score(
                  num_classes=num_classes,
                  average='micro',
                  threshold=FLAGS.ece_label_threshold)
      })

      for policy in ('uncertainty', 'toxicity'):
        metrics.update({
            'test_{}/calibration_auroc_{}'.format(policy, dataset_name):
                tc_metrics.CalibrationAUC(curve='ROC'),
            'test_{}/calibration_auprc_{}'.format(policy, dataset_name):
                tc_metrics.CalibrationAUC(curve='PR'),
        })

        for fraction in FLAGS.fractions:
          metrics.update({
              'test_{}/collab_acc_{}_{}'.format(policy, fraction, dataset_name):
                  rm.metrics.OracleCollaborativeAccuracy(
                      fraction=float(fraction), num_bins=FLAGS.num_approx_bins),
              'test_{}/abstain_prec_{}_{}'.format(policy, fraction,
                                                  dataset_name):
                  tc_metrics.AbstainPrecision(
                      abstain_fraction=float(fraction),
                      num_approx_bins=FLAGS.num_approx_bins),
              'test_{}/abstain_recall_{}_{}'.format(policy, fraction,
                                                    dataset_name):
                  tc_metrics.AbstainRecall(
                      abstain_fraction=float(fraction),
                      num_approx_bins=FLAGS.num_approx_bins),
              'test_{}/collab_auroc_{}_{}'.format(policy, fraction,
                                                  dataset_name):
                  tc_metrics.OracleCollaborativeAUC(
                      oracle_fraction=float(fraction),
                      num_bins=FLAGS.num_approx_bins),
              'test_{}/collab_auprc_{}_{}'.format(policy, fraction,
                                                  dataset_name):
                  tc_metrics.OracleCollaborativeAUC(
                      oracle_fraction=float(fraction),
                      curve='PR',
                      num_bins=FLAGS.num_approx_bins),
          })

  @tf.function
  def generate_sample_weight(labels, class_weight, label_threshold=0.7):
    """Generate sample weight for weighted accuracy calculation."""
    if label_threshold != 0.7:
      logging.warning('The class weight was based on `label_threshold` = 0.7, '
                      'and weighted accuracy/brier will be meaningless if '
                      '`label_threshold` is not equal to this value, which is '
                      'recommended by Jigsaw Conversation AI team.')
    labels_int = tf.cast(labels > label_threshold, tf.int32)
    sample_weight = tf.gather(class_weight, labels_int)
    return sample_weight

  # Evaluate model predictions.
  for n, (dataset_name, test_dataset) in enumerate(test_datasets.items()):
    logits_dataset = []
    for m in range(ensemble_size):
      filename = '{dataset}_{member}.npy'.format(dataset=dataset_name, member=m)
      filename = os.path.join(FLAGS.output_dir, filename)
      with tf.io.gfile.GFile(filename, 'rb') as f:
        logits_dataset.append(np.load(f))

    logits_dataset = tf.convert_to_tensor(logits_dataset)
    test_iterator = iter(test_dataset)
    texts_list = []
    logits_list = []
    labels_list = []
    # Use dict to collect additional labels specified by additional label names.
    # Here we use  `OrderedDict` to get consistent ordering for this dict so
    # we can retrieve the predictions for each identity labels in Colab.
    additional_labels_dict = collections.OrderedDict()
    for step in range(steps_per_eval[dataset_name]):
      try:
        inputs: Dict[str, tf.Tensor] = next(test_iterator)  # pytype: disable=annotation-type-mismatch
      except StopIteration:
        continue
      features, labels, additional_labels = (
          utils.create_feature_and_label(inputs))
      logits = logits_dataset[:, (step * batch_size):((step + 1) * batch_size)]
      loss_logits = tf.squeeze(logits, axis=-1)
      negative_log_likelihood_metric = rm.metrics.EnsembleCrossEntropy(
          binary=True)
      negative_log_likelihood_metric.add_batch(loss_logits, labels=labels)
      negative_log_likelihood = list(
          negative_log_likelihood_metric.result().values())[0]

      per_probs = tf.nn.sigmoid(logits)
      probs = tf.reduce_mean(per_probs, axis=0)
      # Cast labels to discrete for ECE computation
      ece_labels = tf.cast(labels > FLAGS.ece_label_threshold, tf.float32)
      one_hot_labels = tf.one_hot(tf.cast(ece_labels, tf.int32),
                                  depth=num_classes)
      ece_probs = tf.concat([1. - probs, probs], axis=1)
      pred_labels = tf.math.argmax(ece_probs, axis=-1)
      auc_probs = tf.squeeze(probs, axis=1)

      # Use normalized binary predictive variance as the confidence score.
      # Since the prediction variance p*(1-p) is within range (0, 0.25),
      # normalize it by maximum value so the confidence is between (0, 1).
      calib_confidence = 1. - probs * (1. - probs) / .25

      texts_list.append(inputs['input_ids'])
      logits_list.append(logits)
      labels_list.append(labels)
      if 'identity' in dataset_name:
        for identity_label_name in utils.IDENTITY_LABELS:
          if identity_label_name not in additional_labels_dict:
            additional_labels_dict[identity_label_name] = []
          additional_labels_dict[identity_label_name].append(
              additional_labels[identity_label_name].numpy())

      sample_weight = generate_sample_weight(
          labels, class_weight['test/{}'.format(dataset_name)],
          FLAGS.ece_label_threshold)
      if dataset_name == 'ind':
        metrics['test/negative_log_likelihood'].update_state(
            negative_log_likelihood)
        metrics['test/auroc'].update_state(labels, auc_probs)
        metrics['test/aupr'].update_state(labels, auc_probs)
        metrics['test/brier'].update_state(labels, auc_probs)
        metrics['test/brier_weighted'].update_state(
            tf.expand_dims(labels, -1), probs, sample_weight=sample_weight)
        metrics['test/ece'].add_batch(ece_probs, label=ece_labels)
        metrics['test/acc'].update_state(ece_labels, pred_labels)
        metrics['test/acc_weighted'].update_state(
            ece_labels, pred_labels, sample_weight=sample_weight)
        metrics['test/precision'].update_state(ece_labels, pred_labels)
        metrics['test/recall'].update_state(ece_labels, pred_labels)
        metrics['test/f1'].update_state(one_hot_labels, ece_probs)

        for policy in ('uncertainty', 'toxicity'):
          # calib_confidence or decreasing toxicity score.
          confidence = 1. - probs if policy == 'toxicity' else calib_confidence
          binning_confidence = tf.reshape(confidence, [-1])

          metrics['test_{}/calibration_auroc'.format(policy)].update_state(
              ece_labels, pred_labels, confidence)
          metrics['test_{}/calibration_auprc'.format(policy)].update_state(
              ece_labels, pred_labels, confidence)

          for fraction in FLAGS.fractions:
            metrics['test_{}/collab_acc_{}'.format(policy, fraction)].add_batch(
                ece_probs,
                label=ece_labels,
                custom_binning_score=binning_confidence)
            metrics['test_{}/abstain_prec_{}'.format(
                policy, fraction)].update_state(ece_labels, pred_labels,
                                                confidence)
            metrics['test_{}/abstain_recall_{}'.format(
                policy, fraction)].update_state(ece_labels, pred_labels,
                                                confidence)
            metrics['test_{}/collab_auroc_{}'.format(
                policy, fraction)].update_state(
                    labels, auc_probs, custom_binning_score=binning_confidence)
            metrics['test_{}/collab_auprc_{}'.format(
                policy, fraction)].update_state(
                    labels, auc_probs, custom_binning_score=binning_confidence)

      else:
        metrics['test/nll_{}'.format(dataset_name)].update_state(
            negative_log_likelihood)
        metrics['test/auroc_{}'.format(dataset_name)].update_state(
            labels, auc_probs)
        metrics['test/aupr_{}'.format(dataset_name)].update_state(
            labels, auc_probs)
        metrics['test/brier_{}'.format(dataset_name)].update_state(
            labels, auc_probs)
        metrics['test/brier_weighted_{}'.format(dataset_name)].update_state(
            tf.expand_dims(labels, -1), probs, sample_weight=sample_weight)
        metrics['test/ece_{}'.format(dataset_name)].add_batch(
            ece_probs, label=ece_labels)
        metrics['test/acc_{}'.format(dataset_name)].update_state(
            ece_labels, pred_labels)
        metrics['test/acc_weighted_{}'.format(dataset_name)].update_state(
            ece_labels, pred_labels, sample_weight=sample_weight)
        metrics['test/precision_{}'.format(dataset_name)].update_state(
            ece_labels, pred_labels)
        metrics['test/recall_{}'.format(dataset_name)].update_state(
            ece_labels, pred_labels)
        metrics['test/f1_{}'.format(dataset_name)].update_state(
            one_hot_labels, ece_probs)

        for policy in ('uncertainty', 'toxicity'):
          # calib_confidence or decreasing toxicity score.
          confidence = 1. - probs if policy == 'toxicity' else calib_confidence
          binning_confidence = tf.reshape(confidence, [-1])

          metrics['test_{}/calibration_auroc_{}'.format(
              policy, dataset_name)].update_state(ece_labels, pred_labels,
                                                  confidence)
          metrics['test_{}/calibration_auprc_{}'.format(
              policy, dataset_name)].update_state(ece_labels, pred_labels,
                                                  confidence)

          for fraction in FLAGS.fractions:
            metrics['test_{}/collab_acc_{}_{}'.format(
                policy, fraction, dataset_name)].add_batch(
                    ece_probs,
                    label=ece_labels,
                    custom_binning_score=binning_confidence)
            metrics['test_{}/abstain_prec_{}_{}'.format(
                policy, fraction,
                dataset_name)].update_state(ece_labels, pred_labels, confidence)
            metrics['test_{}/abstain_recall_{}_{}'.format(
                policy, fraction,
                dataset_name)].update_state(ece_labels, pred_labels, confidence)
            metrics['test_{}/collab_auroc_{}_{}'.format(
                policy, fraction, dataset_name)].update_state(
                    labels, auc_probs, custom_binning_score=binning_confidence)
            metrics['test_{}/collab_auprc_{}_{}'.format(
                policy, fraction, dataset_name)].update_state(
                    labels, auc_probs, custom_binning_score=binning_confidence)

    texts_all = tf.concat(texts_list, axis=0)
    logits_all = tf.concat(logits_list, axis=1)
    labels_all = tf.concat(labels_list, axis=0)
    additional_labels_all = []
    if additional_labels_dict:
      additional_labels_all = list(additional_labels_dict.values())

    utils.save_prediction(
        texts_all.numpy(),
        path=os.path.join(FLAGS.output_dir, 'texts_{}'.format(dataset_name)))
    utils.save_prediction(
        labels_all.numpy(),
        path=os.path.join(FLAGS.output_dir, 'labels_{}'.format(dataset_name)))
    utils.save_prediction(
        logits_all.numpy(),
        path=os.path.join(FLAGS.output_dir, 'logits_{}'.format(dataset_name)))
    if 'identity' in dataset_name:
      utils.save_prediction(
          np.array(additional_labels_all),
          path=os.path.join(FLAGS.output_dir,
                            'additional_labels_{}'.format(dataset_name)))

    message = ('{:.1%} completion for evaluation: dataset {:d}/{:d}'.format(
        (n + 1) / num_datasets, n + 1, num_datasets))
    logging.info(message)

  total_results = {name: metric.result() for name, metric in metrics.items()}
  # Metrics from Robustness Metrics (like ECE) will return a dict with a
  # single key/value, instead of a scalar.
  total_results = {
      k: (list(v.values())[0] if isinstance(v, dict) else v)
      for k, v in total_results.items()
  }
  logging.info('Metrics: %s', total_results)


if __name__ == '__main__':
  app.run(main)

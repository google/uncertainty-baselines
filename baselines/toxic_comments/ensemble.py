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

import uncertainty_baselines as ub
# import toxic_comments.deterministic to inherit its flags
import deterministic  # pylint:disable=unused-import  # local file import from baselines.toxic_comments
import utils  # local file import from baselines.toxic_comments


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

  dataset_kwargs = dict(
      shuffle_buffer_size=data_buffer_size,
      tf_hub_preprocessor_url=FLAGS.bert_tokenizer_tf_hub_url)

  (_, test_dataset_builders,
   train_split_name) = utils.make_train_and_test_dataset_builders(
       in_dataset_dir=FLAGS.in_dataset_dir,
       ood_dataset_dir=FLAGS.ood_dataset_dir,
       identity_dataset_dir=FLAGS.identity_dataset_dir,
       train_dataset_type=FLAGS.dataset_type,
       test_dataset_type='tfds',
       use_cross_validation=FLAGS.use_cross_validation,
       num_folds=FLAGS.num_folds,
       train_fold_ids=FLAGS.train_fold_ids,
       return_train_split_name=True,
       cv_split_name=FLAGS.train_cv_split_name,
       train_on_identity_subgroup_data=FLAGS.train_on_identity_subgroup_data,
       train_on_bias_label=FLAGS.train_on_bias_label,
       test_on_identity_subgroup_data=FLAGS.test_on_identity_subgroup_data,
       test_on_challenge_data=FLAGS.test_on_challenge_data,
       identity_type_dataset_dir=FLAGS.identity_type_dataset_dir,
       identity_specific_dataset_dir=FLAGS.identity_specific_dataset_dir,
       challenge_dataset_dir=FLAGS.challenge_dataset_dir,
       **dataset_kwargs)

  if FLAGS.prediction_mode:
    prediction_dataset_builders = utils.make_prediction_dataset_builders(
        add_identity_datasets=FLAGS.identity_prediction,
        identity_dataset_dir=FLAGS.identity_specific_dataset_dir,
        add_cross_validation_datasets=FLAGS.use_cross_validation,
        cv_dataset_dir=FLAGS.in_dataset_dir,
        cv_dataset_type=FLAGS.dataset_type,
        num_folds=FLAGS.num_folds,
        train_fold_ids=FLAGS.train_fold_ids,
        cv_split_name=FLAGS.test_cv_split_name,
        **dataset_kwargs)

    # Removes `cv_eval` since it overlaps with the `cv_eval_fold_*` datasets.
    test_dataset_builders.pop('cv_eval', None)
    test_dataset_builders.update(prediction_dataset_builders)

  class_weight = utils.create_class_weight(
      test_dataset_builders=test_dataset_builders)
  logging.info('class_weight: %s', str(class_weight))
  logging.info('train_split_name: %s', train_split_name)

  ds_info = test_dataset_builders['ind'].tfds_info
  feature_size = _MAX_SEQ_LENGTH
  # Positive and negative classes.
  num_classes = ds_info.metadata['num_classes']

  # Build datasets.
  _, test_datasets, _, steps_per_eval = (
      utils.build_datasets({}, test_dataset_builders,
                           batch_size, test_batch_size,
                           per_core_batch_size=FLAGS.per_core_batch_size))
  logging.info('steps_per_eval: %s', steps_per_eval)

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

  metrics = utils.create_train_and_test_metrics(
      test_datasets,
      num_classes=num_classes,
      num_ece_bins=FLAGS.num_ece_bins,
      ece_label_threshold=FLAGS.ece_label_threshold,
      eval_collab_metrics=FLAGS.eval_collab_metrics,
      num_approx_bins=FLAGS.num_approx_bins,
      # Do not eval on bias predictions for now.
      train_on_bias_label=False,
      log_eval_time=False)

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
    ids_list = []
    texts_list = []
    text_ids_list = []
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
      ids = inputs['id']
      texts = inputs['features']
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

      ids_list.append(ids)
      texts_list.append(texts)
      text_ids_list.append(inputs['input_ids'])
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

      # Avoid directly modifying global variable `metrics` (which leads to an
      # assign-before-use error) by creating an update function instead.
      update_fn = utils.make_test_metrics_update_fn(
          dataset_name,
          sample_weight,
          num_classes,
          labels,
          probs,
          negative_log_likelihood=negative_log_likelihood,
          eval_collab_metrics=FLAGS.eval_collab_metrics,
          # Do not eval on bias predictions for now.
          train_on_bias_label=False,
          bias_labels=None,
          bias_probs=None)
      update_fn(metrics)

    ids_all = tf.concat(ids_list, axis=0)
    texts_all = tf.concat(texts_list, axis=0)
    text_ids_all = tf.concat(text_ids_list, axis=0)
    logits_all = tf.concat(logits_list, axis=1)
    labels_all = tf.concat(labels_list, axis=0)
    additional_labels_all = []
    if additional_labels_dict:
      additional_labels_all = list(additional_labels_dict.values())

    utils.save_prediction(
        ids_all.numpy(),
        path=os.path.join(FLAGS.output_dir, 'ids_{}'.format(dataset_name)))
    utils.save_prediction(
        texts_all.numpy(),
        path=os.path.join(FLAGS.output_dir, 'texts_{}'.format(dataset_name)))
    utils.save_prediction(
        text_ids_all.numpy(),
        path=os.path.join(FLAGS.output_dir, 'text_ids_{}'.format(dataset_name)))
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

    # record results
    total_results = {}
    for name, metric in metrics.items():
      try:
        total_results[name] = metric.result()
      except tf.errors.InvalidArgumentError:
        logging.info('Error for metric "%s". Recording 0.', name)
        total_results[name] = 0

  # Metrics from Robustness Metrics (like ECE) will return a dict with a
  # single key/value, instead of a scalar.
  total_results = {
      k: (list(v.values())[0] if isinstance(v, dict) else v)
      for k, v in total_results.items()
  }
  logging.info('Metrics: %s', total_results)


if __name__ == '__main__':
  app.run(main)

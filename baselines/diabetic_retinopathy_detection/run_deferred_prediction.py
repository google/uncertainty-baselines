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

"""Evaluates the Deferred Prediction task using specified model
on the Diabetic Retinopathy dataset.

This script works with the following models:
(1) `deterministic`: see deterministic.py
(2) `dropout`: MC Dropout (Gal and Ghahramani 2016), see dropout.py
(3) `radial`: Radial Bayesian Neural Networks (Farquhar et al. 2020),
  see radial.py
(4) `variational_inference`: see variational_inference.py
(5) `ensemble`: Deep Ensembles, see ensemble.py
(6) `dropoutensemble`: Ensembles of MC Dropout models (Gal and Ghahramani 2016,
  Smith and Gal 2018), see dropoutensemble.py

Our hyperparameters in this script default to MC Dropout. In order to run
the script for another model, the user should set the following hyperparameters
  always: `model_type`, `checkpoint_dir`, `output_dir`
  as desired: `deferred_prediction_fractions`, `uncertainty_type`
  as needed (see note below): `num_mc_samples`

Note the following defaults for various model types.
  (1) All models: we set the training mode to False. This disables
    BatchNormalization but does not affect our stochastic forward passes.
    This is not the norm for many MC Dropout / VI implementations.
    We are able to do this because of the following circumstances:
    - `dropout`: dropout has been manually enabled, circumventing
        training=False; see the `apply_dropout` method in `resnet50_dropout.py`
    - `radial` and `variational_inference`: flipout works with training=False.
  (2) Models using stochastic forward passes (`dropout`, `radial`,
    `variational_inference`, `dropoutensemble`): we set the number of MC samples
    using hyperparameter `num_mc_samples`.
  (3) Ensembles (`ensemble`, `dropoutensemble`): we use all checkpoints found in
    the FLAGS.checkpoint_dir in the ensemble.

Task Description:
  In Deferred Prediction, the model predictive uncertainty is used to choose a
  subset of the test set for which predictions will be evaluated. In particular,
  the uncertainty per test input forms a ranking, and the model's performance
  is evaluated on the X% of test inputs with the least uncertainty. X is
  referred to as the `retain percentage`, and the other (100 - X)% of the data
  is `deferred`.
  Standard evaluation therefore uses a `retain fraction` = [1], i.e., the full
  test set is retained. The user may set an array of such retain fractions,
  e.g., [0.5, 0.6, 0.7, 0.8, 0.9, 1], with the deferred_prediction_fractions
  hyperparameter.

Real-World Relevance:
  We may wish to use a predictive model of diabetic retinopathy to ease the
  burden on clinical practitioners. Under deferred prediction, the model
  refers the examples on which it is least confident to expert doctors.
  We can tune the `retain fraction` parameter based on practitioner
  availability, and a model with well-calibrated uncertainty will have high
  performance on metrics such as AUC/accuracy on the retained evaluation data,
  because its uncertainty and predictive performance are correlated.
"""

import os
import time

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from absl import app
from absl import flags
from absl import logging

import deferred_prediction  # local file import
import uncertainty_baselines as ub
import utils  # local file import

# Data load / output flags.
flags.DEFINE_string(
  'model_type', 'dropout',
  'The type of model being loaded and evaluated in Deferred Prediction. This '
  'is used to retrieve the correct wrapper for obtaining predictive '
  'uncertainty, as implemented in deferred_prediction.py.')
flags.DEFINE_string(
    'checkpoint_dir', '/tmp/diabetic_retinopathy_detection/dropout',
    'The directory from which the trained model weights are retrieved.')
flags.DEFINE_string(
    'output_dir',
    '/tmp/diabetic_retinopathy_detection/dropout_deferred_prediction',
    'The directory where the evaluation summaries are stored.')
flags.DEFINE_string('data_dir', None, 'Path to training and testing data.')
flags.mark_flag_as_required('data_dir')

# General model flags.
flags.DEFINE_integer('seed', 42, 'Random seed.')
flags.DEFINE_integer('eval_batch_size', 32,
                     'The per-core validation/test batch size.')

# Deferred Prediction flags.
flags.DEFINE_list(
    'deferred_prediction_fractions', [0.5, 0.6, 0.7, 0.8, 0.9, 1],
    'The proportion of data that should be retained.')
flags.DEFINE_string(
    'uncertainty_type', 'entropy',
    'The manner of quantifying predictive uncertainty. Currently supported: '
    '`entropy`, `variance`. See deferred_prediction.py.')
flags.DEFINE_integer(
  'num_mc_samples', 10,
  'Number of Monte Carlo samples to use for prediction. Used by the `dropout`,'
  '`dropoutensemble`, `radial`, and `variational_inference` models.')

# Metric flags.
flags.DEFINE_integer('num_bins', 15, 'Number of bins for ECE.')

# Accelerator flags.
flags.DEFINE_bool('use_gpu', True, 'Whether to run on GPU, otherwise CPU.')
flags.DEFINE_bool('use_bfloat16', False, 'Whether to use mixed precision.')
flags.DEFINE_integer(
    'num_cores', 1,
    'Number of TPU cores or number of GPUs; only support 1 GPU for now '
    '(TPUStrategy distribution does not support eager execution, which we '
    'use in the Deferred Prediction API).')
FLAGS = flags.FLAGS


def main(argv):
  del argv  # unused arg
  tf.io.gfile.makedirs(FLAGS.output_dir)
  logging.info(
    'Saving Deferred Prediction evaluation summary to %s', FLAGS.output_dir)
  tf.random.set_seed(FLAGS.seed)

  if FLAGS.num_cores > 1:
    raise ValueError('Only a single accelerator is currently supported.')

  if FLAGS.use_gpu:
    logging.info('Use GPU')
  else:
    logging.info('Use CPU')
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

  model_type = FLAGS.model_type
  eval_batch_size = FLAGS.eval_batch_size * FLAGS.num_cores
  num_mc_samples = FLAGS.num_mc_samples

  # Deferred Prediction flags
  deferred_prediction_fractions = sorted(FLAGS.deferred_prediction_fractions)
  uncertainty_type = FLAGS.uncertainty_type

  try:
    uncertainty_estimator_fn = (
      deferred_prediction.RETINOPATHY_MODEL_TO_UNCERTAINTY_ESTIMATOR[
        model_type])
  except KeyError:
    raise NotImplementedError(
      'Unsupported model type. Try implementing a wrapper to retrieve '
      'predictive uncertainty, as in deferred_prediction.py.')

  # Load test set
  # As per the Kaggle challenge, we have split sizes:
  # train: 35,126
  # validation: 10,906 (currently unused)
  # test: 42,670
  ds_info = tfds.builder('diabetic_retinopathy_detection').info
  steps_per_test_eval = ds_info.splits['test'].num_examples // eval_batch_size
  data_dir = FLAGS.data_dir
  dataset_test_builder = ub.datasets.get(
    'diabetic_retinopathy_detection', split='test', data_dir=data_dir)
  dataset_test = dataset_test_builder.load(batch_size=eval_batch_size)

  if FLAGS.use_bfloat16:
    policy = tf.keras.mixed_precision.experimental.Policy('mixed_bfloat16')
    tf.keras.mixed_precision.experimental.set_policy(policy)

  summary_path = os.path.join(FLAGS.output_dir, 'summaries')
  summary_writer = tf.summary.create_file_writer(summary_path)

  logging.info(f'Building Keras ResNet-50 {model_type} model.')

  # Initialize test metrics
  # For each type of metric, e.g., AUC, initialize
  # one aggregator per deferred prediction fraction.
  metrics = utils.get_diabetic_retinopathy_base_test_metrics(
    use_tpu=False, num_bins=FLAGS.num_bins,
    deferred_prediction_fractions=deferred_prediction_fractions)
  test_metric_fns = utils.get_diabetic_retinopathy_test_metric_fns(
    use_tpu=False)
  metrics.update(utils.get_diabetic_retinopathy_cpu_test_metrics(
    deferred_prediction_fractions=deferred_prediction_fractions))
  metrics.update({'test/ms_per_example': tf.keras.metrics.Mean()})

  # Load from checkpoint
  def load_keras_model(checkpoint):
    model = tf.keras.models.load_model(checkpoint, compile=False)
    logging.info(
      f'Successfully loaded model from checkpoint {checkpoint}.')
    logging.info('Model input shape: %s', model.input_shape)
    logging.info('Model output shape: %s', model.output_shape)
    logging.info('Model number of weights: %s', model.count_params())
    return model

  # TODO(nband): debug, switch from keras.models.save to tf.train.Checkpoint
  checkpoint_filenames = utils.parse_keras_models(FLAGS.checkpoint_dir)
  if not checkpoint_filenames:
    raise Exception(
      f'Did not locate a Keras checkpoint in checkpoint directory '
      f'{FLAGS.checkpoint_dir}')

  # Load in models and wrap, to apply sigmoid on logits, use mixed precision,
  # and cast to NumPy array for use with Deferred Prediction API.
  if model_type in {'ensemble', 'dropoutensemble'}:
    estimator = []
    for checkpoint_file in checkpoint_filenames:
      loaded_model = load_keras_model(checkpoint=checkpoint_file)
      estimator.append(deferred_prediction.wrap_retinopathy_estimator(
        loaded_model, use_mixed_precision=FLAGS.use_bfloat16))
  else:
    latest_checkpoint_file = utils.get_latest_checkpoint(
      file_names=checkpoint_filenames)
    loaded_model = load_keras_model(checkpoint=latest_checkpoint_file)
    estimator = deferred_prediction.wrap_retinopathy_estimator(
      loaded_model, use_mixed_precision=FLAGS.use_bfloat16)

  # Uncertainty estimation arguments -- dependent on model_type
  estimator_args = {'uncertainty_type': uncertainty_type}

  if model_type in {
    'dropout', 'radial', 'variational_inference', 'dropoutensemble'}:
    # Argument for stochastic forward passes
    estimator_args['num_samples'] = num_mc_samples

  # Containers used for caching performance evaluation
  y_true = list()
  y_pred = list()
  y_uncertainty = list()

  test_iterator = iter(dataset_test)
  for step in range(steps_per_test_eval):
    if step % 100 == 0:
      logging.info(f'Evaluated {step}/{steps_per_test_eval} batches.')

    test_start_time = time.time()
    inputs = next(test_iterator)  # pytype: disable=attribute-error
    images = inputs['features']
    labels = inputs['labels']

    # Obtain the predictive mean and uncertainty of the estimator
    # Training setting = False to disable BatchNorm at evaluation time
    # We manually enable dropout at evaluation time (as desired) in the
    # model implementations;
    # e.g. see `apply_dropout` in models/resnet50_dropout.py.

    # Sample from probabilistic model
    batch_mean, batch_uncertainty = uncertainty_estimator_fn(
      images, estimator, training_setting=False,
      **estimator_args)

    # Cache predictions
    y_true.append(labels)
    y_pred.append(batch_mean)
    y_uncertainty.append(batch_uncertainty)

    ms_per_example = (time.time() - test_start_time) * 1e6 / eval_batch_size
    metrics['test/ms_per_example'].update_state(ms_per_example)

  # Use vectorized NumPy containers
  y_true = np.concatenate(y_true).flatten()
  y_pred = np.concatenate(y_pred).flatten()
  y_uncertainty = np.concatenate(y_uncertainty).flatten()

  # Evaluate and update metrics
  deferred_prediction.update_metrics_keras(
    y_true=y_true, y_pred=y_pred, y_uncertainty=y_uncertainty,
    metrics_dict=metrics, test_metric_fns=test_metric_fns,
    fractions=deferred_prediction_fractions)

  # Write evaluation metrics to summary
  total_results = {name: metric.result() for name, metric in metrics.items()}
  import pprint
  pprint.pprint(total_results)

  with summary_writer.as_default():
    for name, result in total_results.items():
      # Note that the step parameter must be set, but is meaningless here.
      tf.summary.scalar(name, result, step=0)

  logging.info(f'Wrote results to {summary_path}.')


if __name__ == '__main__':
  app.run(main)

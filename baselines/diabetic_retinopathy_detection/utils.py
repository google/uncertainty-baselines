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

"""Utilities for models for Diabetic Retinopathy Detection."""

import logging
import os
from functools import partial
from typing import List, Dict, Optional, Union

import robustness_metrics as rm
import tensorflow as tf
import tensorflow.keras.backend as K

from deferred_prediction import negative_log_likelihood_metric


# Distribution / parallelism.


def init_distribution_strategy(force_use_cpu: bool,
                               use_gpu: bool, tpu_name: str):
  """Initialize distribution/parallelization of training or inference.

  Args:
    force_use_cpu: bool, if True, force usage of CPU.
    use_gpu: bool, whether to run on GPU or otherwise TPU.
    tpu_name: str, name of the TPU. Only used if use_gpu is False.

  Returns:
    tf.distribute.Strategy
  """
  if force_use_cpu:
    logging.info('Use CPU')
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
  elif use_gpu:
    logging.info('Use GPU')

  if force_use_cpu or use_gpu:
    strategy = tf.distribute.MirroredStrategy()
  else:
    logging.info('Use TPU at %s', tpu_name if tpu_name is not None else 'local')
    resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu=tpu_name)
    tf.config.experimental_connect_to_cluster(resolver)
    tf.tpu.experimental.initialize_tpu_system(resolver)
    strategy = tf.distribute.TPUStrategy(resolver)

  return strategy


# Model initialization.


def load_input_shape(dataset_train: tf.data.Dataset):
  """Retrieve size of input to model using Shape tuple access.

  Depends on the number of distributed devices.

  Args:
    dataset_train: training dataset.

  Returns:
    list, input shape of model
  """
  try:
    shape_tuple = dataset_train.element_spec['features'].shape
  except AttributeError:  # Multiple TensorSpec in a (nested) PerReplicaSpec.
    tensor_spec_list = dataset_train.element_spec[  # pylint: disable=protected-access
        'features']._flat_tensor_specs
    shape_tuple = tensor_spec_list[0].shape

  return shape_tuple.as_list()[1:]


# Metrics.

def get_test_metrics(
  test_metric_classes: Dict,
  deferred_prediction_fractions: Optional[List[float]] = None):
  """Add a unique test metric for each fraction of data retained in deferred
    prediction.

    If deferred_prediction_fractions is unspecified (= None), we are simply
    evaluating on the full test set, and don't specify the suffix `retain_X`.

  Args:
    test_metric_classes: dict, keys are the name of the test metric and values
      are the classes, which we use for instantiation.
    deferred_prediction_fractions: List[float], used to create a copy of each
      metric for each of the specified fractions. Each metric will be evaluated
      at each deferral percentage, i.e., using an uncertainty score to defer
      prediction on (100 - X)% of the test set, and evaluating the metric on
      the remaining "retained" X% of points.
  Returns:
    dict, test_metrics
  """
  test_metrics = {}

  def update_test_metrics(retain_fraction):
    for metric_key, metric_class in test_metric_classes.items():
      if retain_fraction is None:
        retain_metric_key = f'test/{metric_key}'
      else:
        retain_metric_key = f'test_retain_{retain_fraction}/{metric_key}'

      test_metrics.update({retain_metric_key: metric_class()})

  if deferred_prediction_fractions is None:
    deferred_prediction_fractions = [None]

  # Update test metrics dict, for each retain fraction and metric
  for fraction in deferred_prediction_fractions:
    update_test_metrics(retain_fraction=fraction)

  return test_metrics


def get_diabetic_retinopathy_test_metric_fns(use_tpu: bool):
  """
  Construct dict with keys = metric name, and values that are either:
    None: if no additional preprocessing should be applied to the y_true
      and y_pred tensors before metric.update_state(y_true, y_pred) is called
      each epoch.
    metric_fn: `Callable[[tf.Tensor, tf.Tensor], tf.Tensor]`, which will be
      used for preprocessing the y_true and y_pred tensors, such that the
      output of the metric_fn will be used in metric.update_state(output)
      during each epoch.
  Args:
    use_tpu: bool, ECE is not yet supported on TPU.
  Returns:
    test_metric_fns, dict containing our test metrics and preprocessing methods
  """
  test_metric_fns = {
    'negative_log_likelihood': negative_log_likelihood_metric,
    'accuracy': None,
    'auprc': None,
    'auroc': None
  }

  if not use_tpu:
    test_metric_fns['ece'] = None

  return test_metric_fns


def get_diabetic_retinopathy_test_metric_classes(use_tpu: bool, num_bins: int):
  """Retrieve the relevant metric classes for test set evaluation.
  Do not yet instantiate the test metrics - this must be done for each
  fraction in deferred prediction.

  Args:
    use_tpu: bool, is run using TPU.
    num_bins: int, number of ECE bins.

  Returns:
    dict, test_metric_classes
  """
  test_metric_classes = {
    'negative_log_likelihood': tf.keras.metrics.Mean,
    'accuracy': tf.keras.metrics.BinaryAccuracy
  }

  if use_tpu:
    test_metric_classes.update({
      'auprc': partial(tf.keras.metrics.AUC, curve='PR'),
      'auroc': partial(tf.keras.metrics.AUC, curve='ROC')
    })
  else:
    test_metric_classes.update({
      'ece': partial(rm.metrics.ExpectedCalibrationError, num_bins=num_bins)
    })

  return test_metric_classes


def get_diabetic_retinopathy_base_test_metrics(
    use_tpu: bool, num_bins: int,
    deferred_prediction_fractions: Optional[List[float]] = None):
  """Initialize base test metrics for non-ensemble Diabetic Retinopathy
  predictors.

  Should be called within the distribution strategy scope (e.g. see
  eval_deferred_prediction.py script).

  Note:
    We disclude AUCs in non-TPU case, which must be defined and added to this
    dict outside the strategy scope. See the
    `get_diabetic_retinopathy_cpu_eval_metrics` method for initializing AUC.
    We disclude ECE in TPU case, which currently throws an XLA error on TPU.

  Args:
    use_tpu: bool, is run using TPU.
    num_bins: int, number of ECE bins.
    deferred_prediction_fractions: Optional[List[float]], create a copy of each
      metric for each of the specified fractions. Each metric will be evaluated
      at each deferral percentage, i.e., using an uncertainty score to defer
      prediction on (100 - X)% of the test set, and evaluating the metric on
      the remaining "retained" X% of points.
      If default (None), just create one copy of each metric for evaluation
        on full test set (no deferral).
  Returns:
    dict, test_metrics
  """
  test_metrics = {}

  # Retrieve unitialized test metric classes
  test_metric_classes = get_diabetic_retinopathy_test_metric_classes(
    use_tpu=use_tpu, num_bins=num_bins)

  # Create a test metric for each deferred prediction fraction
  test_metrics.update(get_test_metrics(
    test_metric_classes, deferred_prediction_fractions))

  return test_metrics


def get_diabetic_retinopathy_base_metrics(
    use_tpu: bool, num_bins: int, use_validation: bool = True,
        deferred_prediction_fractions: Optional[List[float]] = None):
  """Initialize base metrics for non-ensemble Diabetic Retinopathy predictors.

  Should be called within the distribution strategy scope (e.g. see
  deterministic.py script).

  Note:
    We disclude AUCs in non-TPU case, which must be defined and added to this
    dict outside the strategy scope. See the
    `get_diabetic_retinopathy_cpu_eval_metrics` method for initializing AUCs.
    We disclude ECE in TPU case, which currently throws an XLA error on TPU.

  Args:
    use_tpu: bool, is run using TPU.
    num_bins: int, number of ECE bins.
    use_validation: whether to use a validation split.
    deferred_prediction_fractions: Optional[List[float]], create a copy of each
      metric for each of the specified fractions. Each metric will be evaluated
      at each deferral percentage, i.e., using an uncertainty score to defer
      prediction on (100 - X)% of the test set, and evaluating the metric on
      the remaining "retained" X% of points.
      If default (None), just create one copy of each metric for evaluation
        on full test set (no deferral).

  Returns:
    dict, metrics
  """
  metrics = {
      'train/negative_log_likelihood': tf.keras.metrics.Mean(),
      'train/accuracy': tf.keras.metrics.BinaryAccuracy(),
      'train/loss': tf.keras.metrics.Mean(),  # NLL + L2
  }
  if use_validation:
    metrics.update({
        'validation/negative_log_likelihood': tf.keras.metrics.Mean(),
        'validation/accuracy': tf.keras.metrics.BinaryAccuracy(),
    })

  if use_tpu:
    # AUC does not yet work within GPU strategy scope, but does for TPU
    metrics.update({
      'train/auprc': tf.keras.metrics.AUC(curve='PR'),
      'train/auroc': tf.keras.metrics.AUC(curve='ROC'),
    })
    if use_validation:
      metrics.update({
        'validation/auprc': tf.keras.metrics.AUC(curve='PR'),
        'validation/auroc': tf.keras.metrics.AUC(curve='ROC')
      })
  else:
    # ECE does not yet work on TPU
    metrics.update({
        'train/ece': rm.metrics.ExpectedCalibrationError(num_bins=num_bins),
    })
    if use_validation:
      metrics.update({
          'validation/ece': rm.metrics.ExpectedCalibrationError(
              num_bins=num_bins)
      })

  # Retrieve unitialized test metric classes
  test_metric_classes = get_diabetic_retinopathy_test_metric_classes(
    use_tpu=use_tpu, num_bins=num_bins)

  # Create a test metric for each deferred prediction fraction
  metrics.update(get_test_metrics(
    test_metric_classes, deferred_prediction_fractions))

  return metrics


def get_diabetic_retinopathy_cpu_test_metrics(
      deferred_prediction_fractions: Optional[List[float]] = None):
  """Initialize test metrics for non-ensemble Diabetic Retinopathy predictors
  that must be initialized outside of the accelerator scope for CPU evaluation.

  Note that this method will cause an error on TPU.

  Args:
    deferred_prediction_fractions: Optional[List[float]], create a copy of each
      metric for each of the specified fractions. Each metric will be evaluated
      at each deferral percentage, i.e., using an uncertainty score to defer
      prediction on (100 - X)% of the test set, and evaluating the metric on
      the remaining "retained" X% of points.
      If default (None), just create one copy of each metric for evaluation
        on full test set (no deferral).

  Returns:
    dict, test_metrics
  """
  test_metrics = {}

  # Do not yet instantiate the test metrics -
  # this must be done for each fraction in deferred prediction
  test_metric_classes = {
    'auprc': partial(tf.keras.metrics.AUC, curve='PR'),
    'auroc': partial(tf.keras.metrics.AUC, curve='ROC')
  }

  # Create a test metric for each deferred prediction fraction
  test_metrics.update(get_test_metrics(
    test_metric_classes, deferred_prediction_fractions))

  return test_metrics


def get_diabetic_retinopathy_cpu_metrics(
  use_validation: bool = True,
  deferred_prediction_fractions: Optional[List[float]] = None):
  """Initialize metrics for non-ensemble Diabetic Retinopathy predictors
  that must be initialized outside of the accelerator scope for CPU evaluation.

  Note that this method will cause an error on TPU.

  Args:
    use_validation: whether to use a validation split.
    deferred_prediction_fractions: Optional[List[float]], create a copy of each
      metric for each of the specified fractions. Each metric will be evaluated
      at each deferral percentage, i.e., using an uncertainty score to defer
      prediction on (100 - X)% of the test set, and evaluating the metric on
      the remaining "retained" X% of points.
      If default (None), just create one copy of each metric for evaluation
        on full test set (no deferral).

  Returns:
    dict, metrics
  """
  metrics = {
    'train/auprc': tf.keras.metrics.AUC(curve='PR'),
    'train/auroc': tf.keras.metrics.AUC(curve='ROC')
  }

  if use_validation:
    metrics.update({
      'validation/auprc': tf.keras.metrics.AUC(curve='PR'),
      'validation/auroc': tf.keras.metrics.AUC(curve='ROC')
    })

  # Do not yet instantiate the test metrics -
  # this must be done for each fraction in deferred prediction
  test_metric_classes = {
    'auprc': partial(tf.keras.metrics.AUC, curve='PR'),
    'auroc': partial(tf.keras.metrics.AUC, curve='ROC')}

  # Create a test metric for each deferred prediction fraction
  metrics.update(get_test_metrics(
    test_metric_classes, deferred_prediction_fractions))

  return metrics


def log_epoch_metrics(metrics, use_tpu):
  """Log epoch metrics -- different metrics supported depending on TPU use.

  Args:
    metrics: dict, contains all train/test metrics evaluated for the run.
    use_tpu: bool, is run using TPU.
  """
  if use_tpu:
    logging.info(
      'Train Loss (NLL+L2): %.4f, Accuracy: %.2f%%, '
      'AUPRC: %.2f%%, AUROC: %.2f%%',
      metrics['train/loss'].result(),
      metrics['train/accuracy'].result() * 100,
      metrics['train/auprc'].result() * 100,
      metrics['train/auroc'].result() * 100)
    logging.info(
      'Test NLL: %.4f, Accuracy: %.2f%%, '
      'AUPRC: %.2f%%, AUROC: %.2f%%',
      metrics['test/negative_log_likelihood'].result(),
      metrics['test/accuracy'].result() * 100,
      metrics['test/auprc'].result() * 100,
      metrics['test/auroc'].result() * 100)
  else:
    logging.info(
      'Train Loss (NLL+L2): %.4f, Accuracy: %.2f%%, '
      'AUPRC: %.2f%%, AUROC: %.2f%%, ECE: %.2f%%',
      metrics['train/loss'].result(),
      metrics['train/accuracy'].result() * 100,
      metrics['train/auprc'].result() * 100,
      metrics['train/auroc'].result() * 100,
      metrics['train/ece'].result() * 100)
    logging.info(
      'Test NLL: %.4f, Accuracy: %.2f%%, '
      'AUPRC: %.2f%%, AUROC: %.2f%%, ECE: %.2f%%',
      metrics['test/negative_log_likelihood'].result(),
      metrics['test/accuracy'].result() * 100,
      metrics['test/auprc'].result() * 100,
      metrics['test/auroc'].result() * 100,
      metrics['test/ece'].result() * 100)


# Checkpoint write/load.


# TODO(nband): debug checkpoint issue with retinopathy models
#   (appears distribution strategy-related)
#   For now, we just reload from keras.models (and only use for inference)
#   using the method below (parse_keras_models)
def parse_checkpoint_dir(checkpoint_dir):
  """Parse directory of checkpoints.

  Intended for use with Deep Ensembles and ensembles of MC Dropout models.
  Currently not used, as per above bug.

  Args:
    checkpoint_dir: checkpoint dir.
  Returns:
    paths of checkpoints
  """
  paths = []
  subdirectories = tf.io.gfile.glob(checkpoint_dir)
  is_checkpoint = lambda f: ('checkpoint' in f and '.index' in f)
  for subdir in subdirectories:
    for path, _, files in tf.io.gfile.walk(subdir):
      if any(f for f in files if is_checkpoint(f)):
        latest_checkpoint_without_suffix = tf.train.latest_checkpoint(path)
        paths.append(os.path.join(path, latest_checkpoint_without_suffix))
        break

  return paths


def parse_keras_models(checkpoint_dir):
  """Parse directory of saved Keras models.

  Used for Deep Ensembles and ensembles of MC Dropout models.

  Args:
    checkpoint_dir: checkpoint dir.
  Returns:
    paths of saved Keras models
  """
  paths = []
  is_keras_model_dir = lambda dir_name: ('keras_model' in dir_name)
  for dir_name in tf.io.gfile.listdir(checkpoint_dir):
    dir_path = os.path.join(checkpoint_dir, dir_name)
    if tf.io.gfile.isdir(dir_path) and is_keras_model_dir(dir_name):
      paths.append(dir_path)

  return paths


def get_latest_checkpoint(file_names):
  """Get latest checkpoint from list of file names. Only necessary
  if manually saving/loading Keras models, i.e., not using the
  tf.train.Checkpoint API.

  Args:
    file_names: List[str], file names located with the `parse_keras_models`
      method.
  Returns:
    str, the file name with the most recent checkpoint
  """
  if not file_names:
    return None

  checkpoint_epoch_and_file_name = []
  for file_name in file_names:
    try:
      checkpoint_epoch = file_name.split('/')[-2].split('_')[-1]
    except ValueError:
      raise Exception('Expected Keras checkpoint directory path of format '
                      'gs://path_to_checkpoint/keras_model_{checkpoint_epoch}/')
    checkpoint_epoch = int(checkpoint_epoch)
    checkpoint_epoch_and_file_name.append((checkpoint_epoch, file_name))

  return sorted(checkpoint_epoch_and_file_name, reverse=True)[0][1]


# Model Training.

def get_diabetic_retinopathy_class_balance_weights(
      positive_empirical_prob: float = None) -> Dict[int, float]:
  """Class weights used for rebalancing the dataset, by skewing the `loss`
  accordingly.

  Diabetic Retinopathy positive class proportions are imbalanced:
    Train: 19.6%
    Val: 18.8%
    Test: 19.2%

  Here, we compute appropriate class weights such that the following
  loss reweighting can be done multiplicatively for each element.

  \mathcal{L}= -\frac{1}{K n} \sum_{i=1}^{n} \frac{\mathcal{L}_{\text{cross-entropy}}}{p(k)}
  where we have K = 2 classes, n images in a minibatch, and the p(k) is the
  empirical probability of class k in the training dataset.

  Therefore, we here compute weights
    w_k = \frac{1}{K} * \frac{1}{p(k)}
  in order to apply the reweighting with an elementwise multiply over the
  batch losses.

  We can also use the empirical probabilities for a particular minibatch,
  i.e. p(k)_{\text{minibatch}}.
  """
  if positive_empirical_prob is None:
    positive_empirical_prob = 0.196

  return {
    0: (1 / 2) * (1 / (1 - positive_empirical_prob)),
    1: (1 / 2) * (1 / positive_empirical_prob)
  }


def get_positive_empirical_prob(labels: tf.Tensor) -> float:
  """
  Given a set of binary labels, determine the empirical probability of a
  positive label (i.e., the proportion of ones).

  Args:
    labels: tf.Tensor, batch of labels

  Returns:
    empirical probability of a positive label
  """
  n_pos_labels = tf.math.count_nonzero(labels)
  total_n_labels = labels.get_shape()[0]
  return n_pos_labels / total_n_labels


def get_weighted_binary_cross_entropy(weights: Dict[int, float]):
  """
  Return a function for calculating weighted binary cross entropy with
  multi-hot encoded labels.

  Due to @menrfa
  (https://stackoverflow.com/questions/46009619/
    keras-weighted-binary-crossentropy)

  # Example
  >>> y_true = tf.convert_to_tensor([1, 0, 0, 0, 0, 0], dtype=tf.int64)
  >>> y_pred = tf.convert_to_tensor(
  ...            [0.6, 0.1, 0.1, 0.9, 0.1, 0.], dtype=tf.float32)
  >>> weights = {
  ...     0: 1.,
  ...     1: 2.
  ... }

  # With weights
  >>> loss_fn = get_weighted_binary_cross_entropy(
  ...             weights=weights, from_logits=False)
  >>> loss_fn(y_true, y_pred)
  <tf.Tensor(0.6067193, shape=(), dtype=tf.float32)>

  # Without weights
  >>> loss_fn = tf.keras.losses.binary_crossentropy
  >>> loss_fn(y_true, y_pred)
  <tf.Tensor(0.52158177, shape=(), dtype=tf.float32)>

  # Another example
  >>> y_true = tf.convert_to_tensor([[0., 1.], [0., 0.]], dtype=tf.float32)
  >>> y_pred = tf.convert_to_tensor([[0.6, 0.4], [0.4, 0.6]], dtype=tf.float32)
  >>> weights = {
  ...     0: 1.,
  ...     1: 2.
  ... }

  # With weights
  >>> loss_fn = get_weighted_binary_cross_entropy(weights=weights, from_logits=False)
  >>> loss_fn(y_true, y_pred)
  <tf.Tensor: shape=(2,), dtype=float32, numpy=array([1.3744358 , 0.71355796], dtype=float32)>

  # Without weights
  >>> loss_fn = tf.keras.losses.binary_crossentropy
  >>> loss_fn(y_true, y_pred)
  <tf.Tensor: shape=(2,), dtype=float32, numpy=array([0.9162905 , 0.71355796], dtype=float32)>

  Args:
    weights: dict, set weights for respective labels, e.g.,
      {
          0: 1.
          1: 8.
      }
      In this case, we aim to compensate for the true (1) label occurring
      less in the training dataset than the false (0) label. e.g.
      [0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0]
    from_logits: bool, if False, we apply a sigmoid to each logit.

  Returns:
    A function to calculate (weighted) binary cross entropy.
  """
  if 0 not in weights or 1 not in weights:
    raise NotImplementedError

  def weighted_cross_entropy_fn(y_true, y_pred, from_logits=False):
    tf_y_true = tf.cast(y_true, dtype=y_pred.dtype)
    tf_y_pred = tf.cast(y_pred, dtype=y_pred.dtype)

    weight_1 = tf.cast(weights[1], dtype=y_pred.dtype)
    weight_0 = tf.cast(weights[0], dtype=y_pred.dtype)
    weights_v = tf.where(tf.equal(tf_y_true, 1), weight_1, weight_0)
    ce = K.binary_crossentropy(
      tf_y_true, tf_y_pred, from_logits=from_logits)
    loss = K.mean(tf.multiply(ce, weights_v), axis=-1)
    return loss

  return weighted_cross_entropy_fn


def get_diabetic_retinopathy_loss_fn(
    class_reweight_mode: Union[str, None],
        class_weights: Union[Dict[int, float], None]):
  """
  Initialize loss function based on class reweighting setting. Return None
  for a minibatch loss, which must be defined per-minibatch, using the
  minibatch empirical label distribution.

  Args:
    class_reweight_mode: Union[str, None], None indicates no class reweighting,
      `constant` indicates reweighting with the training set empirical
      distribution, `minibatch` indicates reweighting with the minibatch
      empirical label distribution.
    class_weights: Union[Dict[int, float], None], class weights as produced by
       `get_diabetic_retinopathy_class_balance_weights`, should only be
       provided for the `constant` class_reweight_mode.

  Returns:
    None, or loss_fn
  """
  #
  if class_reweight_mode is None:
    loss_fn = tf.keras.losses.binary_crossentropy
  elif class_reweight_mode == 'constant':
    # Initialize a reweighted BCE using the empirical class distribution
    # of the training dataset.
    loss_fn = get_weighted_binary_cross_entropy(
      weights=class_weights)
  elif class_reweight_mode == 'minibatch':
    # This loss_fn must be reinitialized for each batch, using the
    # minibatch empirical class distribution.
    loss_fn = None
  else:
    raise NotImplementedError(
      f'Reweighting mode {class_reweight_mode} unsupported.')

  return loss_fn


def get_minibatch_reweighted_loss_fn(labels: tf.Tensor):
  """
  The minibatch-reweighted loss function can only be initialized using the
  labels of a particular minibatch.

  Args:
    labels: tf.Tensor, the labels of a minibatch

  Returns:
    loss_fn, for use in a particular minibatch
  """
  minibatch_positive_empirical_prob = get_positive_empirical_prob(labels=labels)
  minibatch_class_weights = (
    get_diabetic_retinopathy_class_balance_weights(
      positive_empirical_prob=minibatch_positive_empirical_prob))
  batch_loss_fn = get_weighted_binary_cross_entropy(
    weights=minibatch_class_weights)
  return batch_loss_fn

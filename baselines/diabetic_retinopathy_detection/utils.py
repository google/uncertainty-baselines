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

import tensorflow.compat.v1 as tf
import uncertainty_metrics as um
from tensorflow import distribute, config, tpu, io

"""Optimizer."""


class LearningRateSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
  """Learning rate schedule.

  It starts with a linear warmup to the initial learning rate over
  `warmup_epochs`. This is found to be helpful for large batch size training
  (Goyal et al., 2018). The learning rate's value then uses the initial
  learning rate, and decays by a multiplier at the start of each epoch in
  `decay_epochs`. The stepwise decaying schedule follows He et al. (2015).
  """

  def __init__(self, steps_per_epoch, initial_learning_rate, decay_ratio,
               decay_epochs, warmup_epochs):
    super(LearningRateSchedule, self).__init__()
    self.steps_per_epoch = steps_per_epoch
    self.initial_learning_rate = initial_learning_rate
    self.decay_ratio = decay_ratio
    self.decay_epochs = decay_epochs
    self.warmup_epochs = warmup_epochs

  def __call__(self, step):
    lr_epoch = tf.cast(step, tf.float32) / self.steps_per_epoch
    learning_rate = self.initial_learning_rate
    if self.warmup_epochs >= 1:
      learning_rate *= lr_epoch / self.warmup_epochs
    decay_epochs = [self.warmup_epochs] + self.decay_epochs
    for index, start_epoch in enumerate(decay_epochs):
      learning_rate = tf.where(
          lr_epoch >= start_epoch,
          self.initial_learning_rate * self.decay_ratio**index, learning_rate)
    return learning_rate


"""Distribution / parallelism."""


def init_distribution_strategy(force_use_cpu, use_gpu, tpu_name):
  """Initialize distribution/parallelization of training or inference.

  Args:
    force_use_cpu: bool, if True, force usage of CPU.
    use_gpu: bool, whether to run on GPU or otherwise TPU.
    tpu: str, name of the TPU. Only used if use_gpu is False.
  Returns:
    tf.distribute.Strategy
  """
  if force_use_cpu:
    logging.info('Use CPU')
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
  elif use_gpu:
    logging.info('Use GPU')

  if force_use_cpu or use_gpu:
    strategy = distribute.MirroredStrategy()
  else:
    logging.info('Use TPU at %s', tpu_name if tpu_name is not None else 'local')
    resolver = distribute.cluster_resolver.TPUClusterResolver(tpu=tpu_name)
    config.experimental_connect_to_cluster(resolver)
    tpu.experimental.initialize_tpu_system(resolver)
    strategy = distribute.TPUStrategy(resolver)

  return strategy


"""Model initialization."""


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


"""Metrics."""


def get_diabetic_retinopathy_base_metrics(use_tpu, num_bins):
  """
  Initialize base metrics for non-ensemble Diabetic Retinopathy predictors.
  Should be called within the distribution strategy scope
  (e.g. see deterministic.py script).

  :param use_tpu: bool, is run using TPU
  :return: dict, metrics
    Note:
      We disclude AUC in non-TPU case, which must be defined and added to this
      dict outside the strategy scope.
      We disclude ECE in TPU case, which currently throws an XLA error on TPU.
  """
  metrics = {
    'train/negative_log_likelihood': tf.keras.metrics.Mean(),
    'train/accuracy': tf.keras.metrics.BinaryAccuracy(),
    'train/loss': tf.keras.metrics.Mean(),  # NLL + L2
    'test/negative_log_likelihood': tf.keras.metrics.Mean(),
    'test/accuracy': tf.keras.metrics.BinaryAccuracy()}

  if use_tpu:
    # AUC does not yet work within GPU strategy scope, but does for TPU
    metrics.update({
      'train/auc': tf.keras.metrics.AUC(),
      'test/auc': tf.keras.metrics.AUC()})
  else:
    # ECE does not yet work on TPU
    metrics.update({
      'train/ece': um.ExpectedCalibrationError(num_bins=num_bins),
      'test/ece': um.ExpectedCalibrationError(num_bins=num_bins)})

  return metrics


def log_epoch_metrics(metrics, use_tpu):
  """
  Log epoch metrics -- different metrics supported depending on TPU use.
  :param metrics: dict, contains all train/test metrics evaluated for the run
  :param use_tpu: bool, is run using TPU
  """
  if use_tpu:
    logging.info(
      f'Train Loss (NLL+L2): %.4f, Accuracy: %.2f%%, '
      f'AUC: %.2f%%',
      metrics['train/loss'].result(),
      metrics['train/accuracy'].result() * 100,
      metrics['train/auc'].result() * 100)
    logging.info(
      'Test NLL: %.4f, Accuracy: %.2f%%, AUC: %.2f%%',
      metrics['test/negative_log_likelihood'].result(),
      metrics['test/accuracy'].result() * 100,
      metrics['test/auc'].result() * 100)
  else:
    logging.info(
      f'Train Loss (NLL+L2): %.4f, Accuracy: %.2f%%, '
      f'AUC: %.2f%%, ECE: %.2f%%',
      metrics['train/loss'].result(),
      metrics['train/accuracy'].result() * 100,
      metrics['train/auc'].result() * 100,
      metrics['train/ece'].result() * 100)
    logging.info(
      'Test NLL: %.4f, Accuracy: %.2f%%, AUC: %.2f%%, ECE: %.2f%%',
      metrics['test/negative_log_likelihood'].result(),
      metrics['test/accuracy'].result() * 100,
      metrics['test/auc'].result() * 100,
      metrics['test/ece'].result() * 100)


"""Checkpoint write/load."""


# TODO @nband: debug checkpoint issue with retinopathy models
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
  subdirectories = io.gfile.glob(checkpoint_dir)
  is_checkpoint = lambda f: ('checkpoint' in f and '.index' in f)
  for subdir in subdirectories:
    for path, _, files in io.gfile.walk(subdir):
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
  for dir_name in io.gfile.listdir(checkpoint_dir):
    dir_path = os.path.join(checkpoint_dir, dir_name)
    if tf.io.gfile.isdir(dir_path) and is_keras_model_dir(dir_name):
      paths.append(dir_path)

  return paths

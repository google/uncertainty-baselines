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

"""Utilities for models for Diabetic Retinopathy Detection."""

import logging
import os

import tensorflow.compat.v1 as tf


class LearningRateSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
  """Learning rate schedule.

  It starts with a linear warmup to the initial learning rate over
  `warmup_epochs`. This is found to be helpful for large batch size training
  (Goyal et al., 2018). The learning rate's value then uses the initial
  learning rate, and decays by a multiplier at the start of each epoch in
  `decay_epochs`. The stepwise decaying schedule follows He et al. (2015).
  """

  def __init__(self,
               steps_per_epoch,
               initial_learning_rate,
               decay_ratio,
               decay_epochs,
               warmup_epochs):
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
          self.initial_learning_rate * self.decay_ratio**index,
          learning_rate)
    return learning_rate


def init_distribution_strategy(force_use_cpu, use_gpu, tpu):
  """Initialize distribution/parallelization of training or inference.
  :param force_use_cpu: bool, if True, force usage of CPU.
  :param use_gpu: bool, whether to run on GPU or otherwise TPU.
  :param tpu: str, name of the TPU. Only used if use_gpu is False.
  :return: tf.distribute.Strategy
  """
  if force_use_cpu:
    logging.info('Use CPU')
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
  elif use_gpu:
    logging.info('Use GPU')

  if force_use_cpu or use_gpu:
    strategy = tf.distribute.MirroredStrategy()
  else:
    logging.info('Use TPU at %s', tpu if tpu is not None else 'local')
    resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu=tpu)
    tf.config.experimental_connect_to_cluster(resolver)
    tf.tpu.experimental.initialize_tpu_system(resolver)
    strategy = tf.distribute.TPUStrategy(resolver)

  return strategy


def load_input_shape(dataset_train: tf.data.Dataset):
  """
  Retrieve size of input to model using
  Shape tuple access depends on the number of distributed devices.

  :return: list, input shape of model
  """
  try:
    shape_tuple = dataset_train.element_spec['features'].shape
  except AttributeError:  # Multiple TensorSpec in a (nested) PerReplicaSpec.
    tensor_spec_list = dataset_train.element_spec[  # pylint: disable=protected-access
      'features']._flat_tensor_specs
    shape_tuple = tensor_spec_list[0].shape

  return shape_tuple.as_list()[1:]


# TODO @nband: debug checkpoint issue with retinopathy models
#   (appears distribution strategy-related)
#   For now, we just reload from keras.models (and only use for inference)
#   using the method below (parse_keras_models)
def parse_checkpoint_dir(checkpoint_dir):
  """
  Parse directory of checkpoints; intended for use with Deep Ensembles and
  ensembles of MC Dropout models. Currently not used, as per above bug.
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
  """
  Parse directory of saved Keras models; used for Deep Ensembles and
  ensembles of MC Dropout models.
  """
  is_keras_model_dir = lambda f: ('keras_model' in f)

  return [
    f.path for f in os.scandir(checkpoint_dir)
    if f.is_dir() and is_keras_model_dir(f.path)]

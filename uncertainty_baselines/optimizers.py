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

# Lint as: python3
"""Utilities related to optimizers for Uncertainty Baselines."""

import functools

from typing import Any, Dict, Optional, Union
from absl import logging
import tensorflow.compat.v2 as tf
import tensorflow_addons as tfa

LearningRateSchedule = tf.keras.optimizers.schedules.LearningRateSchedule


def _maybe_get_items(d, keys):
  return {k: d[k] for k in keys if k in d}


def get(
    optimizer_name: str,
    learning_rate: float,
    weight_decay: Optional[float] = None,
    learning_rate_schedule: Union[None, str, LearningRateSchedule] = None,
    steps_per_epoch: Optional[int] = None,
    model: tf.keras.Model = None,
    **passed_optimizer_kwargs: Dict[str, Any]) -> tf.keras.optimizers.Optimizer:
  """Builds a tf.keras.optimizers.Optimizer.

  Args:
    optimizer_name: the name of the optimizer to use.
    learning_rate: the base learning rate to use in a possible the learning rate
      schedule.
    weight_decay: an optional weight decay coefficient, applied using decoupled
      weight decay as per
      https://www.tensorflow.org/addons/api_docs/python/tfa/optimizers/extend_with_decoupled_weight_decay.
    learning_rate_schedule: if None, then a constant learning rate is used. Else
      a string name can be passed and a schedule created via
      get_learning_rate_schedule(). Additionally, a custom
      tf.keras.optimizers.schedules.LearningRateSchedule can be passed.
    steps_per_epoch: the number of steps per one epoch of training data.
    model: the Keras model being optimized.
    **passed_optimizer_kwargs: additional kwargs passed to the optimizer.

  Returns:
    A Keras optimizer.
  """
  if isinstance(learning_rate_schedule, str):
    learning_rate = get_learning_rate_schedule(
        schedule_name=learning_rate_schedule,
        base_learning_rate=learning_rate,
        steps_per_epoch=steps_per_epoch,
        **passed_optimizer_kwargs)
  optimizer_kwargs = {'learning_rate': learning_rate}

  optimizer_name = optimizer_name.lower()
  if optimizer_name == 'adam':
    optimizer_class = tf.keras.optimizers.Adam
    optimizer_kwargs.update(
        _maybe_get_items(
            passed_optimizer_kwargs,
            ['learning_rate', 'beta_1', 'beta_2', 'epsilon', 'amsgrad']))
  elif optimizer_name == 'nadam':
    optimizer_class = tf.keras.optimizers.Nadam
    optimizer_kwargs.update(
        _maybe_get_items(
            passed_optimizer_kwargs,
            ['learning_rate', 'beta_1', 'beta_2', 'epsilon']))
  elif optimizer_name == 'rmsprop':
    optimizer_class = tf.keras.optimizers.RMSprop
    optimizer_kwargs.update(
        _maybe_get_items(
            passed_optimizer_kwargs,
            ['learning_rate', 'rho', 'momentum', 'epsilon', 'centered']))
  elif optimizer_name in ['momentum', 'nesterov']:
    optimizer_kwargs.update(
        _maybe_get_items(
            passed_optimizer_kwargs, ['learning_rate', 'momentum', 'nesterov']))
    if optimizer_name == 'nesterov':
      optimizer_kwargs['nesterov'] = True
    optimizer_class = tf.keras.optimizers.SGD
  else:
    raise ValueError('Unrecognized optimizer name: {}'.format(optimizer_name))

  if weight_decay is not None and weight_decay > 0.0:
    optimizer_class = tfa.optimizers.extend_with_decoupled_weight_decay(
        optimizer_class)
    optimizer_kwargs['weight_decay'] = weight_decay

  optimizer = optimizer_class(**optimizer_kwargs)
  if weight_decay is not None and weight_decay > 0.0 and model:
    decay_var_list = []
    skipped_variables = []
    for var in model.trainable_variables:
      if 'kernel' in var.name or 'bias' in var.name:
        decay_var_list.append(var)
      else:
        skipped_variables.append(var)
    logging.info(
        'Not applying weight decay to the following variables:\n%s',
        '\n'.join([var.name for var in skipped_variables]))
    optimizer.apply_gradients = functools.partial(
        optimizer.apply_gradients, decay_var_list=decay_var_list)
  return optimizer


def resnet50_learning_rate_schedule(base_learning_rate, steps_per_epoch):
  """Creates a piecewise LR schedule for the common ResNet-50 ImageNet setup."""
  boundaries = [
      int(40 * steps_per_epoch),
      int(60 * steps_per_epoch),
      int(80 * steps_per_epoch),
      int(90 * steps_per_epoch),
  ]
  values = [
      base_learning_rate,
      base_learning_rate * 1e-1,
      base_learning_rate * 1e-2,
      base_learning_rate * 1e-3,
      base_learning_rate * 5e-4,
  ]
  return tf.keras.optimizers.schedules.PiecewiseConstantDecay(
      boundaries=boundaries,
      values=values)


class LinearWarmupLearningRateSchedule(LearningRateSchedule):
  """Warm up learning rate schedule.

  It starts with a linear warmup to the initial learning rate over
  `warmup_epochs`. This is found to be helpful for large batch size training
  (Goyal et al., 2018). The learning rate's value then uses the initial
  learning rate, and decays by a multiplier at the start of each epoch in
  `decay_epochs`. The stepwise decaying schedule follows He et al. (2015).
  """

  def __init__(self,
               base_learning_rate: float,
               steps_per_epoch: int,
               decay_ratio: float,
               decay_epochs: int,
               warmup_epochs: int):
    super(LinearWarmupLearningRateSchedule, self).__init__()
    self.steps_per_epoch = steps_per_epoch
    self.initial_learning_rate = base_learning_rate
    self.decay_ratio = decay_ratio
    self.decay_epochs = decay_epochs
    self.warmup_epochs = warmup_epochs

  def __call__(self, step: int):
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


def get_learning_rate_schedule(
    schedule_name: str,
    base_learning_rate: float,
    steps_per_epoch: int,
    **schedule_kwargs: Dict[str, Any]) -> Union[float, LearningRateSchedule]:
  """Builds a step-based learning rate schedule."""
  schedule_name = schedule_name.lower()
  if schedule_name == 'constant':
    return base_learning_rate
  elif schedule_name == 'resnet50':
    return resnet50_learning_rate_schedule(base_learning_rate, steps_per_epoch)
  elif schedule_name == 'linear_warmup':
    print('schedule_kwargs : ', schedule_kwargs)
    warmup_hparams = ['decay_ratio', 'decay_epochs', 'warmup_epochs']
    if not all(elem in schedule_kwargs.keys() for elem in warmup_hparams):
      raise ValueError('schedule_kwargs must contain "decay_ratio", '
                       '"decay_epochs" and "warmup_epochs" hyperparameters, '
                       'but only contains ', schedule_kwargs.keys())

    decay_epochs = [int(x) for x in schedule_kwargs['decay_epochs']]
    return LinearWarmupLearningRateSchedule(
        base_learning_rate, steps_per_epoch,
        decay_ratio=schedule_kwargs['decay_ratio'],
        decay_epochs=decay_epochs,
        warmup_epochs=schedule_kwargs['warmup_epochs'])
  else:
    raise ValueError('Unrecognized schedule name: {}'.format(schedule_name))

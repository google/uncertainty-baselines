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

from typing import Any, Dict, Optional, Union
import tensorflow.compat.v2 as tf
import tensorflow_addons as tfa

LearningRateSchedule = tf.keras.optimizers.schedules.LearningRateSchedule


def get(
    optimizer_name: str,
    learning_rate: float,
    weight_decay: Optional[float] = None,
    learning_rate_schedule: Union[None, str, LearningRateSchedule] = None,
    steps_per_epoch: Optional[int] = None,
    **optimizer_kwargs: Dict[str, Any]) -> tf.keras.optimizers.Optimizer:
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
    **optimizer_kwargs: additional kwargs passed to the optimizer.

  Returns:
    A Keras optimizer.
  """
  if isinstance(learning_rate_schedule, str):
    learning_rate = get_learning_rate_schedule(
        schedule_name=learning_rate_schedule,
        base_learning_rate=learning_rate,
        steps_per_epoch=steps_per_epoch)
  optimizer_kwargs['learning_rate'] = learning_rate

  optimizer_name = optimizer_name.lower()
  if optimizer_name == 'adam':
    optimizer_class = tf.keras.optimizers.Adam
  elif optimizer_name == 'nadam':
    optimizer_class = tf.keras.optimizers.Nadam
  elif optimizer_name == 'rmsprop':
    optimizer_class = tf.keras.optimizers.RMSprop
  elif optimizer_name in ['momentum', 'nesterov']:
    if optimizer_name == 'nesterov':
      optimizer_kwargs['nesterov'] = True
    optimizer_class = tf.keras.optimizers.SGD
  else:
    raise ValueError('Unrecognized optimizer name: {}'.format(optimizer_name))

  if weight_decay is not None and weight_decay > 0.0:
    optimizer_class = tfa.optimizers.extend_with_decoupled_weight_decay(
        optimizer_class)
    optimizer_kwargs['weight_decay'] = weight_decay

  return optimizer_class(**optimizer_kwargs)


def renset50_learning_rate_schedule(base_learning_rate, steps_per_epoch):
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


def get_learning_rate_schedule(
    schedule_name: str,
    **schedule_kwargs: Dict[str, Any]) -> Union[float, LearningRateSchedule]:
  """Builds a step-based learning rate schedule."""
  schedule_name = schedule_name.lower()
  if schedule_name == 'constant':
    return schedule_kwargs['base_learning_rate']
  elif schedule_name == 'resnet50':
    return renset50_learning_rate_schedule(**schedule_kwargs)
  else:
    raise ValueError('Unrecognized schedule name: {}'.format(schedule_name))

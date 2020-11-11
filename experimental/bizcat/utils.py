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

"""Utilities for BizCat."""

import math as m
import tensorflow.compat.v1 as tf


class CosineLearningRateSchedule(
    tf.keras.optimizers.schedules.LearningRateSchedule):
  """Cosine learning rate schedule."""

  def __init__(self, base_learning_rate: float, train_steps: int):
    super(CosineLearningRateSchedule, self).__init__()
    self.base_learning_rate = base_learning_rate
    self.train_steps = train_steps

  def __call__(self, step: int):
    decay_factor = (
        1 + tf.math.cos(step / self.train_steps * tf.constant(m.pi))) * 0.5
    return self.base_learning_rate * decay_factor


class LearningRateSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
  """Resnet learning rate schedule."""

  def __init__(self, steps_per_epoch, initial_learning_rate, num_epochs,
               schedule):
    super(LearningRateSchedule, self).__init__()
    self.num_epochs = num_epochs
    self.steps_per_epoch = steps_per_epoch
    self.initial_learning_rate = initial_learning_rate
    self.schedule = schedule

  def __call__(self, step):
    lr_epoch = tf.cast(step, tf.float32) / self.steps_per_epoch
    warmup_lr_multiplier, warmup_end_epoch = self.schedule[0]
    # Scale learning rate schedule by total epochs at vanilla settings.
    warmup_end_epoch = (warmup_end_epoch * self.num_epochs) // 90
    learning_rate = (
        self.initial_learning_rate * warmup_lr_multiplier * lr_epoch /
        warmup_end_epoch)
    for mult, start_epoch in self.schedule:
      start_epoch = (start_epoch * self.num_epochs) // 90
      learning_rate = tf.where(lr_epoch >= start_epoch,
                               self.initial_learning_rate * mult, learning_rate)
    return learning_rate

  def get_config(self):
    return {
        'steps_per_epoch': self.steps_per_epoch,
        'initial_learning_rate': self.initial_learning_rate,
        'num_epochs': self.num_epochs,
        'schedule': self.schedule,
    }

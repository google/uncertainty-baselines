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

  def __init__(self,
               base_learning_rate: float,
               steps_per_epoch: int,
               num_epochs: int,
               init_step=23424):
    super(CosineLearningRateSchedule, self).__init__()
    self.base_learning_rate = base_learning_rate
    self.steps_per_epoch = steps_per_epoch
    self.num_epochs = num_epochs
    self.epoch = 0
    self.init_step = init_step
    self.decay_factor = 0

  def __call__(self, step: int):
    self.epoch = (step - self.init_step) // self.steps_per_epoch
    decay_factor = (
        1 + tf.math.cos(self.epoch / self.num_epochs * tf.constant(m.pi))) * 0.5
    self.decay_factor = decay_factor
    self.learning_rate = self.base_learning_rate * decay_factor
    return self.learning_rate

  def get_lr(self):
    return self.learning_rate

  def get_epoch(self):
    return self.epoch

  def get_decay_factor(self):
    return self.decay_factor


class LearningRateSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
  """Resnet learning rate schedule."""

  def __init__(self, steps_per_epoch, initial_learning_rate, num_epochs,
               schedule):
    super(LearningRateSchedule, self).__init__()
    self.num_epochs = num_epochs
    self.steps_per_epoch = steps_per_epoch
    self.initial_learning_rate = initial_learning_rate
    self.schedule = schedule
    self.learning_rate = initial_learning_rate
    self.epoch = 0
    self.decay_factor = 0

  def __call__(self, step):
    self.epoch = step // self.steps_per_epoch
    lr_epoch = tf.cast(step, tf.float32) / self.steps_per_epoch
    warmup_lr_multiplier, warmup_end_epoch = self.schedule[0]
    # Scale learning rate schedule by total epochs at vanilla settings.
    warmup_end_epoch = (warmup_end_epoch * self.num_epochs) // 90
    learning_rate = (
        self.initial_learning_rate * warmup_lr_multiplier * lr_epoch /
        warmup_end_epoch)
    for mult, start_epoch in self.schedule:
      start_epoch = (start_epoch * self.num_epochs) // 90
      self.learning_rate = tf.where(lr_epoch >= start_epoch,
                                    self.initial_learning_rate * mult,
                                    learning_rate)
    return self.learning_rate

  def get_config(self):
    return {
        'steps_per_epoch': self.steps_per_epoch,
        'initial_learning_rate': self.initial_learning_rate,
        'num_epochs': self.num_epochs,
        'schedule': self.schedule,
    }

  def get_lr(self):
    return self.learning_rate

  def get_epoch(self):
    return self.epoch

  def get_decay_factor(self):
    return self.decay_factor

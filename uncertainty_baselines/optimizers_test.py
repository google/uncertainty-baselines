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

"""Tests for uncertainty_baselines.optimizers."""

import tensorflow as tf
import uncertainty_baselines as ub


class OptimizersTest(tf.test.TestCase):

  def testAdamW(self):
    weight_decay = 1e-3
    optimizer = ub.optimizers.get(
        optimizer_name='adam',
        learning_rate=0.1,
        weight_decay=weight_decay,
        learning_rate_schedule='constant',
        beta_1=0.9,
        epsilon=1e-1)
    shape = (7,)
    initial_value = tf.ones(shape)
    test_var = tf.Variable(name='test_var', initial_value=initial_value)
    zeros_update = [(tf.zeros(shape), test_var)]
    optimizer.apply_gradients(zeros_update)
    # Because we gave a gradient of 0, the only update to the variable should be
    # from the weight decay.
    self.assertAllClose(initial_value - weight_decay, test_var)

  def testResNet50LearningRateSchedule(self):
    base_learning_rate = 1.0
    steps_per_epoch = 2
    schedule = ub.optimizers.get_learning_rate_schedule(
        schedule_name='resnet50',
        base_learning_rate=base_learning_rate,
        steps_per_epoch=steps_per_epoch)
    actual_learning_rates = [schedule(step) for step in range(200)]
    expected_learning_rates = (
        [base_learning_rate * 1e0] * (40 * steps_per_epoch + 1) +
        [base_learning_rate * 1e-1] * 20 * steps_per_epoch +
        [base_learning_rate * 1e-2] * 20 * steps_per_epoch +
        [base_learning_rate * 1e-3] * 10 * steps_per_epoch +
        [base_learning_rate * 5e-4] * (10 * steps_per_epoch - 1))
    self.assertAllClose(actual_learning_rates, expected_learning_rates)


if __name__ == '__main__':
  tf.test.main()

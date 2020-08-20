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
"""Tests for ResNet50."""

import tensorflow as tf
import uncertainty_baselines as ub


class ResNet50Test(tf.test.TestCase):

  def testCreateModel(self):
    batch_size = 31
    model = ub.models.ResNet50Builder(batch_size)
    logits = model(tf.random.uniform((batch_size, 224, 224, 3)))
    self.assertEqual(logits.shape, (batch_size, 1000))


if __name__ == '__main__':
  tf.test.main()

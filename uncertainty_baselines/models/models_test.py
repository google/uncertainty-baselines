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
"""Tests for ub.models.get()."""

import tensorflow as tf
import uncertainty_baselines as ub


class ModelsTest(tf.test.TestCase):

  def testGetModel(self):
    model = ub.models.get('resnet50', batch_size=13, batch_norm_epsilon=1e-2)
    self.assertEqual(176, len(model.layers))


if __name__ == '__main__':
  tf.test.main()

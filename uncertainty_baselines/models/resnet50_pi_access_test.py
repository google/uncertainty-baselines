# coding=utf-8
# Copyright 2023 The Uncertainty Baselines Authors.
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

"""Tests for ResNet-50 with PI access."""

import tensorflow as tf
import uncertainty_baselines as ub


class Resnet50PIAccessTest(tf.test.TestCase):

  def testCreateModel(self):

    model = ub.models.resnet50_pi_access(
        input_shape=(32, 32, 1),
        pi_input_shape=(5, 12),
        num_classes=3,
        width_multiplier=0.3,
        omit_last_layer=False,
        name='resnet50_pi_access_test')

    self.assertEqual(model.input_shape, ((None, 32, 32, 1), (None, 5, 12)))
    self.assertEqual(model.output_shape, (None, 5, 3))


if __name__ == '__main__':
  tf.test.main()

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

"""Tests for uncertainty_baselines.models.textcnn."""

import tensorflow as tf
import uncertainty_baselines as ub


class TextCNNTest(tf.test.TestCase):

  def testCreateModel(self):
    batch_size = 31
    num_classes = 150
    feature_size = 32
    vocab_size = 10000

    model = ub.models.textcnn(batch_size=batch_size,
                              num_classes=num_classes,
                              feature_size=feature_size,
                              vocab_size=vocab_size)
    logits = model(tf.random.uniform((batch_size, feature_size)))
    self.assertEqual(logits.shape, (batch_size, num_classes))


if __name__ == '__main__':
  tf.test.main()

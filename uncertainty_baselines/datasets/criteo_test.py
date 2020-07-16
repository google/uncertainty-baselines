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
"""Tests for Criteo."""

from absl.testing import parameterized
import tensorflow.compat.v2 as tf
import uncertainty_baselines as ub
from uncertainty_baselines.datasets import base


class CriteoDatasetTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      ('Train', base.Split.TRAIN),
      ('Validation', base.Split.VAL),
      ('Test', base.Split.TEST))
  def testDatasetSize(self, split):
    batch_size = 9
    eval_batch_size = 5
    dataset_builder = ub.datasets.CriteoDataset(
        batch_size=batch_size,
        eval_batch_size=eval_batch_size,
        shuffle_buffer_size=20)
    dataset = dataset_builder.build(split).take(1)
    element = next(iter(dataset))
    features = element['features']
    labels = element['labels']

    expected_batch_size = (
        batch_size if split == base.Split.TRAIN else eval_batch_size)
    features_length = len(features)
    feature_shape = features[list(features)[0]].shape
    labels_shape = labels.shape
    self.assertEqual(feature_shape, (expected_batch_size,))
    self.assertEqual(features_length, 39)
    self.assertEqual(labels_shape, (expected_batch_size,))


if __name__ == '__main__':
  tf.test.main()

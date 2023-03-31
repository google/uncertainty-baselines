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

"""Tests for CIFAR-100-C."""

from absl.testing import parameterized
import tensorflow as tf
import uncertainty_baselines as ub


class Cifar100CorruptedDatasetTest(parameterized.TestCase):

  def testCifar100CorruptedDatasetShape(self):
    batch_size_splits = {'test': 5}
    for split, bs in batch_size_splits.items():
      dataset_builder = ub.datasets.Cifar100CorruptedDataset(
          split=split,
          corruption_type='brightness',
          severity=1)
      dataset = dataset_builder.load(batch_size=bs).take(1)
      element = next(iter(dataset))
      features = element['features']
      labels = element['labels']

      features_shape = features.shape
      labels_shape = labels.shape
      self.assertEqual(features_shape, (bs, 32, 32, 3))
      self.assertEqual(labels_shape, (bs,))


if __name__ == '__main__':
  tf.test.main()

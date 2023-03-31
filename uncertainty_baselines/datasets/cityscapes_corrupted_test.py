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

"""Tests for Cityscapes Corrupted."""

from absl.testing import parameterized
import tensorflow as tf
import uncertainty_baselines as ub


class CityscapesCorruptedDatasetTest(ub.datasets.DatasetTest,
                                     parameterized.TestCase):

  def testCityscapesCorruptedDatasetShape(self):
    batch_size_splits = {'test': 5}
    for split, bs in batch_size_splits.items():
      dataset_builder = ub.datasets.CityscapesCorruptedDataset(
          split=split, corruption_type='gaussian_noise', severity=1)
      dataset = dataset_builder.load(batch_size=bs).take(1)
      self.assertEqual(
          list(dataset.element_spec.keys()), ['features', 'labels'])
      element = next(iter(dataset))
      features = element['features']
      labels = element['labels']
      features_shape = features.shape
      labels_shape = labels.shape
      self.assertEqual(features_shape, (bs, 1024, 2048, 3))
      self.assertEqual(labels_shape, (bs, 1024, 2048, 1))


if __name__ == '__main__':
  tf.test.main()

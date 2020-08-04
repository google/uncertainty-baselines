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
"""Tests for Genomics_OOD."""

import tensorflow.compat.v2 as tf
import uncertainty_baselines as ub
from uncertainty_baselines.datasets import base


class GenomicsOodDatasetTest(tf.test.TestCase):
  """Utility class for testing dataset construction."""

  def testDatasetSize(self):
    seq_size = 250
    batch_size = 9
    eval_batch_size = 5
    dataset_builder = ub.datasets.GenomicsOodDataset(
        batch_size=batch_size,
        eval_batch_size=eval_batch_size,
        num_validation_examples=10,
        num_test_examples=10,
        shuffle_buffer_size=20)
    for split in base.Split:
      dataset = dataset_builder.build(split).take(1)
      element = next(iter(dataset))
      features = element['features']
      labels = element['labels']

      features_shape = features.shape
      labels_shape = labels.shape
      if split == base.Split.TRAIN:
        expected_bs = batch_size
      else:
        expected_bs = eval_batch_size
      self.assertEqual(features_shape, (expected_bs, seq_size))
      self.assertEqual(labels_shape, (expected_bs,))


if __name__ == '__main__':
  tf.test.main()

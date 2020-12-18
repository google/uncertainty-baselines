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

import tensorflow as tf
import tensorflow_datasets as tfds
import uncertainty_baselines as ub


class GenomicsOodDatasetTest(tf.test.TestCase):
  """Utility class for testing dataset construction."""

  def testDatasetSize(self):
    seq_size = 250
    for data_mode in ['ind', 'ood']:
      for split in [tfds.Split.TRAIN, tfds.Split.VALIDATION, tfds.Split.TEST]:
        if data_mode == 'ood' and split == tfds.Split.TRAIN:
          continue
        dataset_builder = ub.datasets.GenomicsOodDataset(
            split=split, data_mode=data_mode, shuffle_buffer_size=20)
        batch_size = 9 if split == tfds.Split.TRAIN else 5
        dataset = dataset_builder.load(batch_size=batch_size).take(1)
        element = next(iter(dataset))
        features = element['features']
        labels = element['labels']

        features_shape = features.shape
        labels_shape = labels.shape
        self.assertEqual(features_shape, (batch_size, seq_size))
        self.assertEqual(labels_shape, (batch_size,))


if __name__ == '__main__':
  tf.test.main()

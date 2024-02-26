# coding=utf-8
# Copyright 2024 The Uncertainty Baselines Authors.
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

"""Tests for SMCalflowDataset."""

from absl.testing import parameterized
import tensorflow as tf
import tensorflow_datasets as tfds
from uncertainty_baselines.datasets import smcalflow


class SMCalflowDatasetTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(('Train', tfds.Split.TRAIN),
                                  ('Validation', tfds.Split.VALIDATION),
                                  ('Test', tfds.Split.TEST))
  def testDatasetSize(self, split):
    for name, dataset_builder_cls in zip(
        ['smcalflow', 'multiwoz'],
        [smcalflow.SMCalflowDataset, smcalflow.MultiWoZDataset]):
      if split == tfds.Split.TEST and not smcalflow._has_test_split(name):
        continue
      dataset_builder = dataset_builder_cls(split=split, shuffle_buffer_size=20)
      expected_sizes = smcalflow._get_num_examples(name)
      self.assertEqual(dataset_builder.num_examples, expected_sizes[split])

  @parameterized.named_parameters(('Train', tfds.Split.TRAIN),
                                  ('Validation', tfds.Split.VALIDATION),
                                  ('Test', tfds.Split.TEST))
  def testDatasetShape(self, split):
    batch_size = 5
    max_seq_length = 32
    for name, dataset_builder_cls in zip(
        ['smcalflow', 'multiwoz'],
        [smcalflow.SMCalflowDataset, smcalflow.MultiWoZDataset]):
      if split == tfds.Split.TEST and not smcalflow._has_test_split(name):
        continue
      dataset_builder = dataset_builder_cls(
          split=split, shuffle_buffer_size=20, max_seq_length=max_seq_length)

      dataset = dataset_builder.load(batch_size=batch_size).take(1)
      element = next(iter(dataset))
      for feature in smcalflow._FEATURES:
        self.assertEqual(element[feature].shape, (batch_size, max_seq_length))

if __name__ == '__main__':
  tf.test.main()

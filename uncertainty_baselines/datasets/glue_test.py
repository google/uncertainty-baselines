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

"""Tests for uncertainty_baselines.datasets.glue."""

from absl.testing import parameterized

import tensorflow as tf
import tensorflow_datasets as tfds

from uncertainty_baselines.datasets import glue


class GlueTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(('Train', tfds.Split.TRAIN),
                                  ('Validation', tfds.Split.VALIDATION),
                                  ('Test', tfds.Split.TEST))
  def testDatasetSize(self, split):
    batch_size = 9 if split == tfds.Split.TRAIN else 5
    for _, dataset_class in glue.GlueDatasets.items():
      dataset_builder = dataset_class(
          split=split,
          shuffle_buffer_size=20)
      dataset = dataset_builder.load(batch_size=batch_size).take(1)
      element = next(iter(dataset))
      text_a = element['text_a']
      labels = element['labels']

      feature_shape = text_a.shape[0]
      labels_shape = labels.shape[0]

      self.assertEqual(feature_shape, batch_size)
      self.assertEqual(labels_shape, batch_size)

  @parameterized.named_parameters(('Train', tfds.Split.TRAIN),
                                  ('Validation', tfds.Split.VALIDATION),
                                  ('Test', tfds.Split.TEST))
  def testTextbIsNone(self, split):
    batch_size = 9 if split == tfds.Split.TRAIN else 5
    for dataset_name, dataset_class in glue.GlueDatasets.items():
      dataset_builder = dataset_class(
          split=split,
          shuffle_buffer_size=20)
      dataset = dataset_builder.load(batch_size=batch_size).take(1)
      element = next(iter(dataset))
      text_b = element['text_b']

      if dataset_name in ('glue/cola', 'glue/sst2'):
        self.assertIsNone(text_b)
      else:
        self.assertIsNotNone(text_b)


if __name__ == '__main__':
  tf.test.main()

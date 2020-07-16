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

"""Tests for uncertainty_baselines.datasets.glue."""

from absl.testing import parameterized

import tensorflow as tf

from uncertainty_baselines.datasets import base
from uncertainty_baselines.datasets import glue


class GlueTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(('Train', base.Split.TRAIN),
                                  ('Validation', base.Split.VAL),
                                  ('Test', base.Split.TEST))
  def testDatasetSize(self, split):
    batch_size = 9
    eval_batch_size = 5
    for _, dataset_class in glue.GlueDatasets.items():
      dataset_builder = dataset_class(
          batch_size=batch_size,
          eval_batch_size=eval_batch_size,
          shuffle_buffer_size=20)
      dataset = dataset_builder.build(split).take(1)
      element = next(iter(dataset))
      text_a = element['text_a']
      labels = element['labels']

      expected_batch_size = (
          batch_size if split == base.Split.TRAIN else eval_batch_size)
      feature_shape = text_a.shape[0]
      labels_shape = labels.shape[0]

      self.assertEqual(feature_shape, expected_batch_size)
      self.assertEqual(labels_shape, expected_batch_size)

  @parameterized.named_parameters(('Train', base.Split.TRAIN),
                                  ('Validation', base.Split.VAL),
                                  ('Test', base.Split.TEST))
  def testTextbIsNone(self, split):
    batch_size = 9
    eval_batch_size = 5
    for dataset_name, dataset_class in glue.GlueDatasets.items():
      dataset_builder = dataset_class(
          batch_size=batch_size,
          eval_batch_size=eval_batch_size,
          shuffle_buffer_size=20)
      dataset = dataset_builder.build(split).take(1)
      element = next(iter(dataset))
      text_b = element['text_b']

      if dataset_name in ('glue/cola', 'glue/sst2'):
        self.assertIsNone(text_b)
      else:
        self.assertIsNotNone(text_b)


if __name__ == '__main__':
  tf.test.main()

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

"""Tests for uncertainty_baselines.datasets.mnli."""

from absl.testing import parameterized

import tensorflow as tf

from uncertainty_baselines.datasets import base
from uncertainty_baselines.datasets import mnli


class MnliTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(('ind,train', 'matched', base.Split.TRAIN),
                                  ('ind,valid', 'matched', base.Split.VAL),
                                  ('ind,test', 'matched', base.Split.TEST),
                                  ('ood,valid', 'mismatched', base.Split.VAL),
                                  ('ood,test', 'mismatched', base.Split.TEST))
  def testDatasetSize(self, mode, split):
    batch_size = 9
    eval_batch_size = 5
    dataset_builder = mnli.MnliDataset(
        mode=mode,
        batch_size=batch_size,
        eval_batch_size=eval_batch_size,
        shuffle_buffer_size=20)
    dataset = dataset_builder.build(split).take(1)
    element = next(iter(dataset))
    text_a = element['text_a']
    text_b = element['text_b']
    labels = element['labels']

    expected_batch_size = (
        batch_size if split == base.Split.TRAIN else eval_batch_size)
    feature_a_shape = text_a.shape[0]
    feature_b_shape = text_b.shape[0]
    labels_shape = labels.shape[0]

    self.assertEqual(feature_a_shape, expected_batch_size)
    self.assertEqual(feature_b_shape, expected_batch_size)
    self.assertEqual(labels_shape, expected_batch_size)

  def testTrainSplitError(self):
    """Tests for ValueError when calling train split for mismatched data."""
    batch_size = 9
    eval_batch_size = 5
    dataset_builder = mnli.MnliDataset(
        mode='mismatched',
        batch_size=batch_size,
        eval_batch_size=eval_batch_size,
        shuffle_buffer_size=20)

    with self.assertRaises(Exception):
      dataset_builder.build(base.Split.TRAIN)


if __name__ == '__main__':
  tf.test.main()

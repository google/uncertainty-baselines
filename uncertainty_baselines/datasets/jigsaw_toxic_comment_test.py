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
"""Tests for uncertainty_baselines.datasets.jigsaw_toxic_comment."""

from absl.testing import parameterized
import tensorflow.compat.v2 as tf

import uncertainty_baselines as ub
from uncertainty_baselines.datasets import base
from uncertainty_baselines.datasets import jigsaw_toxic_comment

UbSplit = ub.datasets.base.Split


class JigsawToxicCommentTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(('Train', UbSplit.TRAIN),
                                  ('Test', UbSplit.TEST))
  def testDatasetSize(self, split):
    batch_size = 9
    eval_batch_size = 5
    dataset_builder = ub.datasets.JigsawToxicCommentDataset(
        batch_size=batch_size,
        eval_batch_size=eval_batch_size,
        shuffle_buffer_size=20)
    dataset = dataset_builder.build(split).take(1)
    element = next(iter(dataset))
    features = element['features']
    labels = element['labels']

    expected_batch_size = (
        batch_size if split == UbSplit.TRAIN else eval_batch_size)
    expected_feature_length = jigsaw_toxic_comment._FEATURE_LENGTH
    expected_label_length = jigsaw_toxic_comment._LABEL_LENGTH

    feature_batch_size, feature_length = features.shape
    label_batch_size, label_length = labels.shape
    self.assertEqual(feature_batch_size, expected_batch_size)
    self.assertEqual(label_batch_size, expected_batch_size)
    self.assertEqual(feature_length, expected_feature_length)
    self.assertEqual(label_length, expected_label_length)

  def testValidSplitError(self):
    val_split = base.Split.VAL
    zero_val_percent = 0.

    dataset_builder = ub.datasets.JigsawToxicCommentDataset(
        batch_size=32, eval_batch_size=32,
        validation_percent=zero_val_percent,
        shuffle_buffer_size=32)

    with self.assertRaises(ValueError):
      dataset_builder.build(val_split)


if __name__ == '__main__':
  tf.test.main()

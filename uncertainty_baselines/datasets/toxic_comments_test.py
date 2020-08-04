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
"""Tests for toxicity classification datasets."""

from absl.testing import parameterized

import tensorflow as tf

from uncertainty_baselines.datasets import base
from uncertainty_baselines.datasets import toxic_comments


CCDataClass = toxic_comments.CivilCommentsDataset
CCIdentitiesDataClass = toxic_comments.CivilCommentsIdentitiesDataset
WTDataClass = toxic_comments.WikipediaToxicityDataset


class ToxicCommentsDatasetTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      ('civil_train', base.Split.TRAIN, CCDataClass),
      ('civil_valid', base.Split.VAL, CCDataClass),
      ('civil_test', base.Split.TEST, CCDataClass),
      ('civil_identities_train', base.Split.TRAIN, CCIdentitiesDataClass),
      ('civil_identities_valid', base.Split.VAL, CCIdentitiesDataClass),
      ('civil_identities_test', base.Split.TEST, CCIdentitiesDataClass),
      ('wiki_train', base.Split.TRAIN, WTDataClass),
      ('wiki_valid', base.Split.VAL, WTDataClass),
      ('wiki_test', base.Split.TEST, WTDataClass))
  def testDatasetSize(self, split, dataset_class):
    batch_size = 9
    eval_batch_size = 5
    dataset_builder = dataset_class(
        batch_size=batch_size,
        eval_batch_size=eval_batch_size,
        shuffle_buffer_size=20)
    dataset = dataset_builder.build(split).take(1)
    element = next(iter(dataset))
    features = element['features']
    labels = element['labels']

    expected_batch_size = (
        batch_size if split == base.Split.TRAIN else eval_batch_size)
    feature_shape = features.shape[0]
    labels_shape = labels.shape[0]

    self.assertEqual(feature_shape, expected_batch_size)
    self.assertEqual(labels_shape, expected_batch_size)

  @parameterized.named_parameters(
      ('civil_comments', CCDataClass),
      ('civil_comments_identities', CCIdentitiesDataClass),
      ('wikipedia_toxicity', WTDataClass))
  def testSubType(self, dataset_class):
    """Test if toxicity subtype is available from the example."""
    batch_size = 9
    eval_batch_size = 5
    dataset_builder = dataset_class(
        batch_size=batch_size,
        eval_batch_size=eval_batch_size,
        shuffle_buffer_size=20)
    dataset = dataset_builder.build(base.Split.TRAIN).take(1)
    element = next(iter(dataset))

    for subtype_name in dataset_builder.additional_labels:
      self.assertEqual(element[subtype_name].shape[0], batch_size)
      self.assertEqual(element[subtype_name].dtype, tf.float32)


if __name__ == '__main__':
  tf.test.main()

# coding=utf-8
# Copyright 2022 The Uncertainty Baselines Authors.
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

"""Tests for toxicity classification datasets."""

from absl.testing import parameterized

import tensorflow as tf
import tensorflow_datasets as tfds

from uncertainty_baselines.datasets import toxic_comments

HUB_PREPROCESS_URL = 'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3'

CCDataClass = toxic_comments.CivilCommentsDataset
CCIdentitiesDataClass = toxic_comments.CivilCommentsIdentitiesDataset
WTDataClass = toxic_comments.WikipediaToxicityDataset


class ToxicCommentsDatasetTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      ('civil_train', tfds.Split.TRAIN, CCDataClass),
      ('civil_valid', tfds.Split.VALIDATION, CCDataClass),
      ('civil_test', tfds.Split.TEST, CCDataClass),
      ('civil_identities_train', tfds.Split.TRAIN, CCIdentitiesDataClass),
      ('civil_identities_valid', tfds.Split.VALIDATION, CCIdentitiesDataClass),
      ('civil_identities_test', tfds.Split.TEST, CCIdentitiesDataClass),
      ('wiki_train', tfds.Split.TRAIN, WTDataClass),
      ('wiki_valid', tfds.Split.VALIDATION, WTDataClass),
      ('wiki_test', tfds.Split.TEST, WTDataClass))
  def testDatasetSize(self, split, dataset_class):
    batch_size = 9 if split == tfds.Split.TRAIN else 5
    dataset_builder = dataset_class(
        split=split,
        shuffle_buffer_size=20)
    dataset = dataset_builder.load(batch_size=batch_size).take(1)
    element = next(iter(dataset))

    features = element['features']
    labels = element['labels']
    ids = element['id']

    self.assertEqual(features.shape[0], batch_size)
    self.assertEqual(labels.shape[0], batch_size)
    self.assertEqual(ids.shape[0], batch_size)

  @parameterized.named_parameters(
      ('civil_train', tfds.Split.TRAIN, CCDataClass),
      ('civil_valid', tfds.Split.VALIDATION, CCDataClass),
      ('civil_test', tfds.Split.TEST, CCDataClass),
      ('civil_identities_train', tfds.Split.TRAIN, CCIdentitiesDataClass),
      ('civil_identities_valid', tfds.Split.VALIDATION, CCIdentitiesDataClass),
      ('civil_identities_test', tfds.Split.TEST, CCIdentitiesDataClass),
      ('wiki_train', tfds.Split.TRAIN, WTDataClass),
      ('wiki_valid', tfds.Split.VALIDATION, WTDataClass),
      ('wiki_test', tfds.Split.TEST, WTDataClass))
  def testTFHubProcessor(self, split, dataset_class):
    batch_size = 9 if split == tfds.Split.TRAIN else 5
    dataset_builder = dataset_class(
        split=split,
        shuffle_buffer_size=20,
        tf_hub_preprocessor_url=HUB_PREPROCESS_URL)
    dataset = dataset_builder.load(batch_size=batch_size).take(1)
    element = next(iter(dataset))

    input_ids = element['input_ids']
    input_mask = element['input_mask']
    segment_ids = element['segment_ids']

    self.assertEqual(input_ids.shape[0], batch_size)
    self.assertEqual(input_mask.shape[0], batch_size)
    self.assertEqual(segment_ids.shape[0], batch_size)

  @parameterized.named_parameters(
      ('civil_comments', CCDataClass),
      ('civil_comments_identities', CCIdentitiesDataClass),
      ('wikipedia_toxicity', WTDataClass))
  def testSubType(self, dataset_class):
    """Test if toxicity subtype is available from the example."""
    batch_size = 9
    dataset_builder = dataset_class(
        split=tfds.Split.TRAIN,
        shuffle_buffer_size=20)
    dataset = dataset_builder.load(batch_size=batch_size).take(1)
    element = next(iter(dataset))

    for subtype_name in dataset_builder.additional_labels:
      self.assertEqual(element[subtype_name].shape[0], batch_size)
      self.assertEqual(element[subtype_name].dtype, tf.float32)


if __name__ == '__main__':
  tf.test.main()

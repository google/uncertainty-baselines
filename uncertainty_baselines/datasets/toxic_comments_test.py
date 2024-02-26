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

"""Tests for toxicity classification datasets."""

from absl.testing import parameterized
import pandas as pd
import tensorflow as tf
import tensorflow_datasets as tfds
from uncertainty_baselines.datasets import toxic_comments

HUB_PREPROCESS_URL = 'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3'

CCDataClass = toxic_comments.CivilCommentsDataset
CCIdentitiesDataClass = toxic_comments.CivilCommentsIdentitiesDataset
WTDataClass = toxic_comments.WikipediaToxicityDataset


def _create_fake_signals(dataset_name, is_train_signals):
  is_train_signals = is_train_signals[:5]
  is_train_signals += [0] * (5 - len(is_train_signals))
  fake_signals = {
      'civil_comments_identities': pd.DataFrame({
          'id': [b'876617', b'688889', b'5769682', b'4997434', b'5489660'],
          'is_train': is_train_signals,
      }).set_index('id'),
      'civil_comments': pd.DataFrame({
          'id': [b'634903', b'5977874', b'5390534', b'871483', b'825427'],
          'is_train': is_train_signals,
      }).set_index('id'),
      'wikipedia_toxicity': pd.DataFrame({
          'id': [
              b'ee9697785fe41ff8',
              b'29fec512f2ee929e',
              b'88944b29dde50648',
              b'c7bf1f59096102f3',
              b'7d71ee0e8ea0794a',
          ],
          'is_train': is_train_signals,
      }).set_index('id'),
  }

  return fake_signals[dataset_name]


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
        split=split, dataset_type='tfds', shuffle_buffer_size=20)
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
        dataset_type='tfds',
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
        split=tfds.Split.TRAIN, dataset_type='tfds', shuffle_buffer_size=20)
    dataset = dataset_builder.load(batch_size=batch_size).take(1)
    element = next(iter(dataset))

    for subtype_name in dataset_builder.additional_labels:
      self.assertEqual(element[subtype_name].shape[0], batch_size)
      self.assertEqual(element[subtype_name].dtype, tf.float32)

  @parameterized.named_parameters(
      ('civil_comments', 'civil_comments', CCDataClass),
      ('civil_comments_identities', 'civil_comments_identities',
       CCIdentitiesDataClass),
      ('wikipedia_toxicity', 'wikipedia_toxicity', WTDataClass))
  def testAppendSignals(self, dataset_name, dataset_class):
    """Test if toxicity subtype is available from the example."""
    batch_size = 5
    is_train_signals = [1, 0, 0, 1, 0]
    signals = _create_fake_signals(dataset_name, is_train_signals)

    dataset_builder = dataset_class(
        split=tfds.Split.TRAIN,
        dataset_type='tfds',
        is_training=False,  # Fix the order.
        signals=signals,
        shuffle_buffer_size=20)
    dataset = dataset_builder.load(batch_size=batch_size).take(1)
    element = next(iter(dataset))

    self.assertEqual(element['is_train'].numpy().tolist(), is_train_signals)

  @parameterized.named_parameters(
      ('civil_comments', 'civil_comments', CCDataClass),
      ('civil_comments_identities', 'civil_comments_identities',
       CCIdentitiesDataClass),
      ('wikipedia_toxicity', 'wikipedia_toxicity', WTDataClass))
  def testOnlyKeepTrainExamples(self, dataset_name, dataset_class):
    """Test if toxicity subtype is available from the example."""
    batch_size = 3
    is_train_signals = [1, 1, 0, 1, 1]
    signals = _create_fake_signals(dataset_name, is_train_signals)

    dataset_builder = dataset_class(
        split=tfds.Split.TRAIN,
        dataset_type='tfds',
        is_training=False,  # Fix the order.
        signals=signals,
        only_keep_train_examples=True,
        shuffle_buffer_size=20)
    dataset = dataset_builder.load(batch_size=batch_size).take(1)
    element = next(iter(dataset))

    self.assertEqual(dataset_builder.num_examples, sum(is_train_signals))
    expected_is_train_ids = signals[signals['is_train'] == 1].index.tolist()
    self.assertEqual(element['id'].numpy().tolist(),
                     expected_is_train_ids[:batch_size])

  @parameterized.named_parameters(
      ('civil_comments', CCDataClass),
      (
          'civil_comments_identities',
          CCIdentitiesDataClass,
      ),
      ('wikipedia_toxicity', WTDataClass),
  )
  def testFilterByCreatedDateRegex(self, dataset_class):
    """Test if toxicity subtype is available from the example."""
    batch_size = 9
    created_date_regex = r'.*-01-.*'

    dataset_builder = dataset_class(
        split=tfds.Split.TRAIN,
        dataset_type='tfds',
        shuffle_buffer_size=20,
        created_date_regex=created_date_regex,
    )
    dataset = dataset_builder.load(batch_size=batch_size).take(1)
    element = next(iter(dataset))
    if 'created_date' in element:
      for created_date in element['created_date'].numpy().tolist():
        self.assertRegex(created_date.decode('ascii'), created_date_regex)


if __name__ == '__main__':
  tf.test.main()

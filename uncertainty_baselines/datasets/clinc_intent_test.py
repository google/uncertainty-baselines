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
"""Tests for ClincIntentDetectionDataset."""

from absl.testing import parameterized
import numpy as np
import tensorflow.compat.v2 as tf
import uncertainty_baselines as ub
from uncertainty_baselines.datasets import base
from uncertainty_baselines.datasets import clinc_intent


class ClincIntentDetectionDatasetTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(('Train', base.Split.TRAIN),
                                  ('Validation', base.Split.VAL),
                                  ('Test', base.Split.TEST))
  def testDatasetSize(self, split):
    batch_size = 9
    eval_batch_size = 5
    dataset_builder = ub.datasets.ClincIntentDetectionDataset(
        batch_size=batch_size,
        eval_batch_size=eval_batch_size,
        shuffle_buffer_size=20)
    dataset = dataset_builder.build(split).take(1)
    element = next(iter(dataset))
    features = element['features']
    labels = element['labels']

    expected_batch_size = (
        batch_size if split == base.Split.TRAIN else eval_batch_size)
    feature_shape, _ = features.shape
    labels_shape = labels.shape
    self.assertEqual(feature_shape, expected_batch_size)
    self.assertEqual(labels_shape, (expected_batch_size,))

  @parameterized.named_parameters(('IND', 'ind', clinc_intent._NUM_TRAIN_IND),
                                  ('OOD', 'ood', clinc_intent._NUM_TRAIN_OOD),
                                  ('All', 'all', clinc_intent._NUM_TRAIN_ALL))
  def testDataMode(self, data_mode, num_train_examples_expected):
    """Tests if all data modes can be loaded correctly."""
    batch_size = 9
    eval_batch_size = 5
    split = base.Split.TRAIN

    dataset_builder = ub.datasets.ClincIntentDetectionDataset(
        batch_size=batch_size,
        eval_batch_size=eval_batch_size,
        shuffle_buffer_size=20,
        data_mode=data_mode)

    num_train_examples = dataset_builder.info['num_train_examples']
    dataset = dataset_builder.build(split).take(1)
    element = next(iter(dataset))
    features = element['features']

    _, features_length = features.shape

    self.assertEqual(features_length, clinc_intent._FEATURE_LENGTH)
    self.assertEqual(num_train_examples, num_train_examples_expected)

  def testTokenizer(self):
    """Tests if tokenizer is loaded correctly."""
    dataset_builder = ub.datasets.ClincIntentDetectionDataset(
        batch_size=9,
        eval_batch_size=5,
        shuffle_buffer_size=20,
    )

    # The number of valid tokens.
    vocab_size = dataset_builder.tokenizer.num_words

    self.assertEqual(vocab_size, 7291)

  def testNumTokens(self):
    """Tests if num_tokens field is loaded correctly."""
    batch_size = 9
    eval_batch_size = 5
    split = base.Split.TRAIN

    dataset_builder = ub.datasets.ClincIntentDetectionDataset(
        batch_size=batch_size,
        eval_batch_size=eval_batch_size,
        shuffle_buffer_size=20,)

    dataset = dataset_builder.build(split).take(1)
    element = next(iter(dataset))
    features = element['features']
    num_tokens = element['num_tokens']

    # compute number of tokens expected
    num_tokens_expected = np.sum(features.numpy() != 0, axis=-1)
    num_tokens_loaded = num_tokens.numpy()

    np.testing.assert_array_equal(num_tokens_loaded, num_tokens_expected)

if __name__ == '__main__':
  tf.test.main()

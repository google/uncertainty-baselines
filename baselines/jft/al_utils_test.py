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

"""Tests for al_utils."""

import os
import pathlib

import tensorflow as tf
import tensorflow_datasets as tfds
import al_utils  # local file import from baselines.jft
import test_utils  # local file import from baselines.jft


class AlUtilsTest(tf.test.TestCase):

  def setUp(self):
    super().setUp()
    baseline_root_dir = pathlib.Path(__file__).parents[1]
    self.data_dir = os.path.join(baseline_root_dir, 'testing_data')

  def test_mnist_subset_has_ids(self):
    data_dir = self.data_dir

    num_examples = 50
    with tfds.testing.mock_data(num_examples=num_examples, data_dir=data_dir):
      # NOTE: MNIST has no id field, so this tests the enumerate code path.
      mnist_builder = tfds.builder('mnist')
      print(mnist_builder.info.splits)
      dataset_builder = al_utils.SubsetDatasetBuilder(
          mnist_builder, subset_ids=None)
      dataset = dataset_builder.as_dataset(split='train').batch(10)

      all_ids = []
      n = 0
      for example in dataset.as_numpy_iterator():
        n += example['image'].shape[0]
        all_ids.extend(example['id'])
      all_ids.sort()

    self.assertEqual(n, num_examples)
    self.assertEqual(all_ids, list(range(num_examples)))

  def test_cifar10_subset_has_ids(self):
    data_dir = self.data_dir

    num_examples = 50
    with tfds.testing.mock_data(num_examples=num_examples, data_dir=data_dir):
      # NOTE: cifar10 has an id field.
      cifar10_builder = tfds.builder('cifar10')
      dataset_builder = al_utils.SubsetDatasetBuilder(
          cifar10_builder, subset_ids=None)
      dataset = dataset_builder.as_dataset(split='train').batch(10)

      all_ids = []
      n = 0
      for example in dataset.as_numpy_iterator():
        n += example['image'].shape[0]
        all_ids.extend(example['id'])
      all_ids.sort()

    self.assertEqual(n, num_examples)
    self.assertEqual(all_ids, list(range(num_examples)))

  def test_cifar10_subset_filtering(self):
    data_dir = self.data_dir

    train_ids = [0, 1, 2]
    test_ids = [3, 4, 5]

    num_examples = 50
    with tfds.testing.mock_data(num_examples=num_examples, data_dir=data_dir):
      cifar10_builder = tfds.builder('cifar10')

      for split, ids in zip(['train', 'test'], [train_ids, test_ids]):
        dataset_builder = al_utils.SubsetDatasetBuilder(
            cifar10_builder, subset_ids=ids)
        dataset = dataset_builder.as_dataset(split=split).batch(1)

        ds_ids = []
        for example in dataset.as_numpy_iterator():
          ds_ids.extend(example['id'])
        ds_ids.sort()

        self.assertEqual(ds_ids, ids)


if __name__ == '__main__':
  tf.test.main()

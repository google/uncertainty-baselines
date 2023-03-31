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

"""Tests for al_utils.py."""

import tensorflow as tf
import tensorflow_datasets as tfds
# pylint: disable=unused-import # to register Cifar10Subset as dataset
import al_utils  # local file import from baselines.jft


class AlUtilsTest(tf.test.TestCase):

  def test_mnist_subset_has_ids(self):
    # NOTE: MNIST has no id field, so this tests the enumerate code path.
    mnist_builder = tfds.builder('mnist')
    mnist_builder.download_and_prepare()

    print(mnist_builder.as_dataset('train'))

    dataset_builder = al_utils.SubsetDatasetBuilder(
        mnist_builder, subset_ids=None)

    dataset = dataset_builder.as_dataset(split='train').batch(10000)

    all_ids = []
    n = 0
    for example in dataset.as_numpy_iterator():
      n += example['image'].shape[0]
      all_ids.extend(example['id'])

    all_ids.sort()

    self.assertEqual(n, 60000)
    self.assertEqual(all_ids, list(range(60000)))

  def test_cifar10_subset_has_ids(self):
    # NOTE: cifar10 has an id field.
    cifar10_builder = tfds.builder('cifar10')
    cifar10_builder.download_and_prepare()

    dataset_builder = al_utils.SubsetDatasetBuilder(
        cifar10_builder, subset_ids=None)

    dataset = dataset_builder.as_dataset(split='train').batch(10000)

    all_ids = []
    n = 0
    for example in dataset.as_numpy_iterator():
      n += example['image'].shape[0]
      all_ids.extend(example['id'])

    all_ids.sort()

    self.assertEqual(n, 50000)
    self.assertEqual(all_ids, list(range(50000)))

  def test_cifar10_subset_filtering(self):
    train_ids = [0, 1, 2]
    test_ids = [3, 4, 5]

    cifar10_builder = tfds.builder('cifar10')
    cifar10_builder.download_and_prepare()

    for split, ids in zip(['train', 'test'], [train_ids, test_ids]):
      dataset_builder = al_utils.SubsetDatasetBuilder(
          cifar10_builder, subset_ids=ids)
      dataset = dataset_builder.as_dataset(split=split).batch(1)

      ds_ids = []
      for example in dataset.as_numpy_iterator():
        ds_ids.extend(example['id'])

      ds_ids.sort()

      self.assertEqual(ds_ids, ids)

  def test_sample_class_balanced_ids(self):
    cifar10_builder = tfds.builder('cifar10')
    num_classes = 10
    cifar10_builder.download_and_prepare()
    data_builder = al_utils.SubsetDatasetBuilder(
        cifar10_builder, subset_ids=None)
    dataset = data_builder.as_dataset(split='train')
    ids = al_utils.sample_class_balanced_ids(2 * num_classes + 1, dataset,
                                             num_classes)
    for i, example_id in enumerate(ids):
      # pylint: disable=cell-var-from-loop
      examples = dataset.filter(lambda x: x['id'] == example_id)
      examples = examples.as_numpy_iterator()
      self.assertLen(examples, 1)
      self.assertEqual(examples.next(), i%num_classes)


if __name__ == '__main__':
  tf.test.main()

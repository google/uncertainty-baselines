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

import tensorflow as tf
import tensorflow_datasets as tfds
# pylint: disable=unused-import # to register Cifar10Subset as dataset
import al_utils  # local file import from baselines.jft.active_learning


class AlUtilsTest(tf.test.TestCase):

  def test_cifar10_subset_has_ids(self):
    dataset_builder = tfds.builder(
        'cifar10_subset', subset_ids={
            'train': None,
            'test': None
        })

    split = tfds.even_splits('train', 1)
    dataset = dataset_builder.as_dataset(split)[0].batch(500)

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

    dataset_builder = tfds.builder(
        'cifar10_subset', subset_ids={
            'train': train_ids,
            'test': test_ids
        })

    for split, ids in zip(['train', 'test'], [train_ids, test_ids]):
      split = tfds.even_splits(split, 1)
      dataset = dataset_builder.as_dataset(split)[0].batch(1)

      ds_ids = []
      for example in dataset.as_numpy_iterator():
        ds_ids.extend(example['id'])

      ds_ids.sort()

      self.assertEqual(ds_ids, ids)


if __name__ == '__main__':
  tf.test.main()

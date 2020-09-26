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
from uncertainty_baselines.datasets import movielens


class MovieLensDatasetTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(('Train', base.Split.TRAIN),
                                  ('Validation', base.Split.VAL),
                                  ('Test', base.Split.TEST))
  def testDatasetSize(self, split):
    batch_size = 9
    eval_batch_size = 5

    dataset_builder = movielens.MovieLensDataset(
        batch_size=batch_size,
        eval_batch_size=eval_batch_size,
        validation_percent=0.1,
        test_percent=0.2,
        shuffle_buffer_size=20)
    dataset = dataset_builder.build(split).take(1)
    element = next(iter(dataset))

    movie_id = element['movie_id']
    user_id = element['user_id']
    labels = element['labels']

    expected_batch_size = (
        batch_size if split == base.Split.TRAIN else eval_batch_size)

    self.assertEqual(movie_id.shape[0], expected_batch_size)
    self.assertEqual(user_id.shape[0], expected_batch_size)
    self.assertEqual(labels.shape[0], expected_batch_size)

  @parameterized.named_parameters(('Train', base.Split.TRAIN),
                                  ('Validation', base.Split.VAL),
                                  ('Test', base.Split.TEST))
  def testDatasetSizeNoneValidation(self, split):
    batch_size = 9
    eval_batch_size = 5

    dataset_builder = movielens.MovieLensDataset(
        batch_size=batch_size,
        eval_batch_size=eval_batch_size,
        validation_percent=0,
        test_percent=0.2,
        shuffle_buffer_size=20)

    if split == base.Split.VAL:
      with self.assertRaises(ValueError):
        dataset = dataset_builder.build(split).take(1)
    else:
      dataset = dataset_builder.build(split).take(1)
      element = next(iter(dataset))

      movie_id = element['movie_id']
      user_id = element['user_id']
      labels = element['labels']

      expected_batch_size = (
          batch_size if split == base.Split.TRAIN else eval_batch_size)

      self.assertEqual(movie_id.shape[0], expected_batch_size)
      self.assertEqual(user_id.shape[0], expected_batch_size)
      self.assertEqual(labels.shape[0], expected_batch_size)


if __name__ == '__main__':
  tf.test.main()

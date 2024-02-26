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

"""Tests for uncertainty_baselines.datasets.glue."""

from absl.testing import parameterized

import tensorflow as tf
import tensorflow_datasets as tfds

from uncertainty_baselines.datasets import movielens


class MovieLensDatasetTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(('Train', tfds.Split.TRAIN),
                                  ('Validation', tfds.Split.VALIDATION),
                                  ('Test', tfds.Split.TEST))
  def testDatasetSize(self, split):
    batch_size = 9 if split == tfds.Split.TRAIN else 5

    dataset_builder = movielens.MovieLensDataset(
        split=split,
        validation_percent=0.1,
        test_percent=0.2,
        shuffle_buffer_size=20)
    dataset = dataset_builder.load(batch_size=batch_size).take(1)
    element = next(iter(dataset))

    feature_name_arr = ['timestamp', 'movie_id', 'movie_title',
                        'user_id', 'user_gender', 'bucketized_user_age',
                        'user_occupation_label', 'user_occupation_text',
                        'user_zip_code', 'labels']

    for name in feature_name_arr:
      feature = element[name]
      self.assertEqual(feature.shape[0], batch_size)

  @parameterized.named_parameters(('Train', tfds.Split.TRAIN),
                                  ('Validation', tfds.Split.VALIDATION),
                                  ('Test', tfds.Split.TEST))
  def testDatasetSizeNoneValidation(self, split):
    batch_size = 9 if split == tfds.Split.TRAIN else 5
    if split == tfds.Split.VALIDATION:
      with self.assertRaises(ValueError):
        dataset_builder = movielens.MovieLensDataset(
            split,
            validation_percent=0,
            test_percent=0.2,
            shuffle_buffer_size=20)
    else:
      dataset_builder = movielens.MovieLensDataset(
          split,
          validation_percent=0,
          test_percent=0.2,
          shuffle_buffer_size=20)
      dataset = dataset_builder.load(batch_size=batch_size).take(1)
      element = next(iter(dataset))

      feature_name_arr = ['timestamp', 'movie_id', 'movie_title',
                          'user_id', 'user_gender', 'bucketized_user_age',
                          'user_occupation_label', 'user_occupation_text',
                          'user_zip_code', 'labels']

      for name in feature_name_arr:
        feature = element[name]
        self.assertEqual(feature.shape[0], batch_size)


if __name__ == '__main__':
  tf.test.main()

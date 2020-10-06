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
"""Data loader for the MovieLens dataset."""

from typing import Any, Dict, Optional
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds
from uncertainty_baselines.datasets import base


class MovieLensDataset(base.BaseDataset):
  """MovieLens dataset builder class."""

  def __init__(
      self,
      batch_size: int,
      eval_batch_size: int,
      validation_percent: float = 0.1,
      test_percent: float = 0.2,
      shuffle_buffer_size: int = None,
      num_parallel_parser_calls: int = 64,
      data_dir: Optional[str] = None,
      **unused_kwargs: Dict[str, Any]):
    """Create a Criteo tf.data.Dataset builder.

    Args:
      batch_size: the training batch size.
      eval_batch_size: the validation and test batch size.
      validation_percent: the percent of the data to use as a validation set.
      test_percent: the percent of the data to use as a test set.
      shuffle_buffer_size: the number of example to use in the shuffle buffer
        for tf.data.Dataset.shuffle(). Usually shuffle_buffer_size shall be
        much larger than batch_size.
      num_parallel_parser_calls: the number of parallel threads to use while
        preprocessing in tf.data.Dataset.map().
      data_dir: optional dir to save TFDS data to. If none then the local
        filesystem is used. Required for using TPUs on Cloud.
    """
    # The total example size and detailed info on MovieLens-1M can be found at:
    # https://www.tensorflow.org/datasets/catalog/movie_lens#movie_lens1m-ratings
    num_total_examples = 1000209

    num_train_examples = int(
        num_total_examples * (1.0 - test_percent - validation_percent))
    num_validation_examples = int(num_total_examples * validation_percent)
    num_test_examples = int(num_total_examples * test_percent)

    super(MovieLensDataset, self).__init__(
        name='movielens',
        num_train_examples=num_train_examples,
        num_validation_examples=num_validation_examples,
        num_test_examples=num_test_examples,
        batch_size=batch_size,
        eval_batch_size=eval_batch_size,
        shuffle_buffer_size=shuffle_buffer_size,
        num_parallel_parser_calls=num_parallel_parser_calls,
        data_dir=data_dir)

  def _read_examples(self, split: base.Split) -> tf.data.Dataset:
    if split == base.Split.TRAIN:
      train_split = tfds.core.ReadInstruction(
          'train', to=self._num_train_examples, unit='abs')
      return tfds.load(
          'movie_lens/1m-ratings', split=train_split, **self._tfds_kwargs)
    if split == base.Split.VAL:
      if self._num_validation_examples == 0:
        raise ValueError(
            'No validation set provided. Set `validation_percent > 0.0` to '
            'take a subset of the training set as validation.')
      val_split = tfds.core.ReadInstruction(
          'train',
          from_=self._num_train_examples,
          to=-self._num_test_examples, unit='abs')
      return tfds.load(
          'movie_lens/1m-ratings', split=val_split, **self._tfds_kwargs)
    elif split == base.Split.TEST:
      test_split = tfds.core.ReadInstruction(
          'train', from_=-self._num_test_examples, unit='abs')
      return tfds.load(
          'movie_lens/1m-ratings', split=test_split, **self._tfds_kwargs)

  def _create_process_example_fn(self, split: base.Split) -> base.PreProcessFn:
    del split

    def _example_parser(example: Dict[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
      """A pre-process function to return features and labels."""
      # TODO(chenzhe): To add movie_genres feature. This feature has different
      # lengths for different input examples.
      parsed_example = {
          'timestamp': example['timestamp'],
          # movie features
          'movie_id': example['movie_id'],
          'movie_title': example['movie_title'],
          # user features
          'user_id': example['user_id'],
          'user_gender': example['user_gender'],
          'bucketized_user_age': example['bucketized_user_age'],
          'user_occupation_label': example['user_occupation_label'],
          'user_occupation_text': example['user_occupation_text'],
          'user_zip_code': example['user_zip_code'],
          # label
          'labels': example['user_rating'],
      }
      return parsed_example

    return _example_parser



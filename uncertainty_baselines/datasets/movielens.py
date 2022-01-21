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

"""Data loader for the MovieLens dataset."""

from typing import Dict, Optional
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds
from uncertainty_baselines.datasets import base


class MovieLensDataset(base.BaseDataset):
  """MovieLens dataset builder class."""

  def __init__(self,
               split: str,
               validation_percent: float = 0.0,
               test_percent: float = 0.2,
               shuffle_buffer_size: Optional[int] = None,
               num_parallel_parser_calls: int = 64,
               normalize: bool = True,
               try_gcs: bool = False,
               download_data: bool = False,
               data_dir: Optional[str] = None,
               is_training: Optional[bool] = None):
    """Create a MovieLens tf.data.Dataset builder.

    Args:
      split: a dataset split, either a custom tfds.Split or one of the
        tfds.Split enums [TRAIN, VALIDAITON, TEST] or their lowercase string
        names.
      validation_percent: the percent of the training set to use as a validation
        set.
      test_percent: the percent of the data to use as a test set.
      shuffle_buffer_size: the number of example to use in the shuffle buffer
        for tf.data.Dataset.shuffle().
      num_parallel_parser_calls: the number of parallel threads to use while
        preprocessing in tf.data.Dataset.map().
      normalize: whether or not to normalize each image by the CIFAR dataset
        mean and stddev.
      try_gcs: Whether or not to try to use the GCS stored versions of dataset
        files.
      download_data: Whether or not to download data before loading.
      data_dir: Directory to read/write data, that is passed to the
              tfds dataset_builder as a data_dir parameter.
      is_training: Whether or not the given `split` is the training split. Only
        required when the passed split is not one of ['train', 'validation',
        'test', tfds.Split.TRAIN, tfds.Split.VALIDATION, tfds.Split.TEST].
    """
    # The total example size and detailed info on MovieLens-1M can be found at:
    # https://www.tensorflow.org/datasets/catalog/movie_lens#movie_lens1m-ratings
    num_total_examples = 1000209

    if is_training is None:
      is_training = split in ['train', tfds.Split.TRAIN]

    if validation_percent < 0.0 or validation_percent >= 1.0:
      raise ValueError(
          'validation_percent must be in [0, 1), received {}.'.format(
              validation_percent))

    num_train_examples = int(
        num_total_examples * (1.0 - test_percent - validation_percent))
    num_validation_examples = int(num_total_examples * validation_percent)
    num_test_examples = int(num_total_examples * test_percent)

    if split == tfds.Split.TRAIN:
      split = tfds.core.ReadInstruction(
          'train', to=num_train_examples, unit='abs')
    if split == tfds.Split.VALIDATION:
      if num_validation_examples == 0:
        raise ValueError(
            'No validation set provided. Set `validation_percent > 0.0` to '
            'take a subset of the training set as validation.')
      split = tfds.core.ReadInstruction(
          'train',
          from_=num_train_examples,
          to=-num_test_examples,
          unit='abs')
    if split == tfds.Split.TEST:
      split = tfds.core.ReadInstruction(
          'train', from_=-num_test_examples, unit='abs')

    name = 'movie_lens/1m-ratings'
    dataset_builder = tfds.builder(name, try_gcs=try_gcs, data_dir=data_dir)
    super().__init__(
        name=name,
        dataset_builder=dataset_builder,
        split=split,
        is_training=is_training,
        shuffle_buffer_size=shuffle_buffer_size,
        num_parallel_parser_calls=num_parallel_parser_calls,
        download_data=download_data)

  def _create_process_example_fn(self) -> base.PreProcessFn:

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

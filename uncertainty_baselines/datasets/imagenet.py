# coding=utf-8
# Copyright 2021 The Uncertainty Baselines Authors.
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
"""ImageNet dataset builder.

We have an option to use a percent of the training dataset as a validation set,
and treat the original validation set as the test set. This is similar to what
is also done in the NeurIPS uncertainty benchmark paper
https://arxiv.org/abs/1906.02530 (which used (100 / 1024)% as a validation set).
"""

from typing import Any, Dict, Optional
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds
from uncertainty_baselines.datasets import base
from uncertainty_baselines.datasets import inception_preprocessing


class ImageNetDataset(base.BaseDataset):
  """ImageNet dataset builder class."""

  def __init__(
      self,
      split: str,
      validation_percent: float = 0.0,
      shuffle_buffer_size: Optional[int] = None,
      num_parallel_parser_calls: int = 64,
      try_gcs: bool = False,
      download_data: bool = False,
      **unused_kwargs: Dict[str, Any]):
    """Create an ImageNet tf.data.Dataset builder.

    Args:
      split: a dataset split, either a custom tfds.Split or one of the
        tfds.Split enums [TRAIN, VALIDAITON, TEST] or their lowercase string
        names.
      validation_percent: the percent of the training set to use as a validation
        set.
      shuffle_buffer_size: the number of example to use in the shuffle buffer
        for tf.data.Dataset.shuffle().
      num_parallel_parser_calls: the number of parallel threads to use while
        preprocessing in tf.data.Dataset.map().
      try_gcs: Whether or not to try to use the GCS stored versions of dataset
        files.
      download_data: Whether or not to download data before loading.
    """
    name = 'imagenet2012'
    dataset_builder = tfds.builder(name, try_gcs=try_gcs)
    split = base.get_validation_percent_split(
        dataset_builder,
        validation_percent,
        split,
        test_split=tfds.Split.VALIDATION)
    super(ImageNetDataset, self).__init__(
        name=name,
        dataset_builder=dataset_builder,
        split=split,
        shuffle_buffer_size=shuffle_buffer_size,
        num_parallel_parser_calls=num_parallel_parser_calls,
        fingerprint_key='file_name',
        download_data=download_data)

  def _create_process_example_fn(self) -> base.PreProcessFn:
    """Create a pre-process function to return images in [0, 1]."""

    def _example_parser(example: Dict[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
      """Preprocesses ImageNet image Tensors using inception_preprocessing."""
      # `preprocess_image` returns images in [-1, 1].
      image = inception_preprocessing.preprocess_image(
          example['image'],
          height=224,
          width=224,
          is_training=self._is_training)
      # Rescale to [0, 1].
      image = (image + 1.0) / 2.0

      label = tf.cast(example['label'], tf.int32)
      parsed_example = {
          'features': image,
          'labels': label,
      }
      if 'file_name' in example:
        parsed_example['file_name'] = example['file_name']
      return parsed_example

    return _example_parser

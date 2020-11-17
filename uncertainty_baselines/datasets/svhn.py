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
"""SVHN dataset builder."""

from typing import Any, Dict, Optional

import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds
from uncertainty_baselines.datasets import base


class SvhnDataset(base.BaseDataset):
  """SVHN dataset builder class."""

  def __init__(
      self,
      split: str,
      validation_percent: float = 0.0,
      shuffle_buffer_size: Optional[int] = None,
      num_parallel_parser_calls: int = 64,
      try_gcs: bool = False,
      download_data: bool = False,
      normalize_by_cifar: bool = False,
      **unused_kwargs: Dict[str, Any]):
    """Create an SVHN tf.data.Dataset builder.

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
      normalize_by_cifar: whether or not to normalize each image by the CIFAR
        dataset mean and stddev.
    """
    self._normalize_by_cifar = normalize_by_cifar
    name = 'svhn_cropped'
    dataset_builder = tfds.builder(name, try_gcs=try_gcs)
    split = base.get_validation_percent_split(
        dataset_builder, validation_percent, split)
    super(SvhnDataset, self).__init__(
        name=name,
        dataset_builder=dataset_builder,
        split=split,
        shuffle_buffer_size=shuffle_buffer_size,
        num_parallel_parser_calls=num_parallel_parser_calls,
        download_data=download_data)

  def _create_process_example_fn(self) -> base.PreProcessFn:

    def _example_parser(example: Dict[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
      """A pre-process function to return images in [0, 1]."""
      image = example['image']
      image = tf.image.convert_image_dtype(image, tf.float32)
      label = tf.cast(example['label'], tf.int32)
      if self._normalize_by_cifar:
        mean = tf.constant([0.4914, 0.4822, 0.4465])
        std = tf.constant([0.2023, 0.1994, 0.2010])
        image = (image - mean) / std
      parsed_example = {
          'features': image,
          'labels': label,
      }
      return parsed_example

    return _example_parser

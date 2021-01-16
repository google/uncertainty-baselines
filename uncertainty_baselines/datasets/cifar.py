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
"""CIFAR{10,100} dataset builders."""

from typing import Any, Dict

from robustness_metrics.common import types
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds
from uncertainty_baselines.datasets import base


class _CifarDataset(base.BaseDataset):
  """CIFAR dataset builder abstract class."""

  def __init__(
      self,
      name: str,
      fingerprint_key: str,
      split: str,
      validation_percent: float = 0.0,
      shuffle_buffer_size: int = None,
      num_parallel_parser_calls: int = 64,
      drop_remainder: bool = True,
      normalize: bool = True,
      try_gcs: bool = False,
      download_data: bool = False,
      **unused_kwargs: Dict[str, Any]):
    """Create a CIFAR10 or CIFAR100 tf.data.Dataset builder.

    Args:
      name: the name of this dataset, either 'cifar10' or 'cifar100'.
      fingerprint_key: The name of the feature holding a string that will be
        used to create an element id using a fingerprinting function. If None,
        then `ds.enumerate()` is added before the `ds.map(preprocessing_fn)` is
        called and an `id` field is added to the example Dict.
      split: a dataset split, either a custom tfds.Split or one of the
        tfds.Split enums [TRAIN, VALIDAITON, TEST] or their lowercase string
        names.
      validation_percent: the percent of the training set to use as a validation
        set.
      shuffle_buffer_size: the number of example to use in the shuffle buffer
        for tf.data.Dataset.shuffle().
      num_parallel_parser_calls: the number of parallel threads to use while
        preprocessing in tf.data.Dataset.map().
      drop_remainder: whether or not to drop the last batch of data if the
        number of points is not exactly equal to the batch size. This option
        needs to be True for running on TPUs.
      normalize: whether or not to normalize each image by the CIFAR dataset
        mean and stddev.
      try_gcs: Whether or not to try to use the GCS stored versions of dataset
        files.
      download_data: Whether or not to download data before loading.
    """
    self._normalize = normalize
    dataset_builder = tfds.builder(name, try_gcs=try_gcs)
    split = base.get_validation_percent_split(
        dataset_builder, validation_percent, split)
    super(_CifarDataset, self).__init__(
        name=name,
        dataset_builder=dataset_builder,
        split=split,
        shuffle_buffer_size=shuffle_buffer_size,
        num_parallel_parser_calls=num_parallel_parser_calls,
        drop_remainder=drop_remainder,
        fingerprint_key=fingerprint_key,
        download_data=download_data)

  def _create_process_example_fn(self) -> base.PreProcessFn:

    def _example_parser(example: types.Features) -> types.Features:
      """A pre-process function to return images in [0, 1]."""
      image = example['image']
      label = example['label']
      if self._is_training:
        # Expand the image by 2 pixels, then crop back down to 32x32.
        image = tf.image.resize_with_crop_or_pad(image, 32 + 4, 32 + 4)
        image = tf.image.random_crop(image, [32, 32, 3])
        image = tf.image.random_flip_left_right(image)

      # Normalize the values of the image to be in [0, 1].
      if image.dtype != tf.uint8:
        raise ValueError(
            'Images need to be type uint8 to use tf.image.convert_image_dtype.')
      # The image has values in the range [0, 1].
      image = tf.image.convert_image_dtype(image, tf.float32)
      # Optionally normalize by the dataset statistics.
      if self._normalize:
        mean = tf.constant([0.4914, 0.4822, 0.4465])
        std = tf.constant([0.2023, 0.1994, 0.2010])
        image = (image - mean) / std
      parsed_example = example.copy()
      parsed_example['features'] = image
      parsed_example['labels'] = tf.cast(label, tf.int32)
      del parsed_example['image']
      del parsed_example['label']
      return parsed_example

    return _example_parser


class Cifar10Dataset(_CifarDataset):
  """CIFAR10 dataset builder class."""

  def __init__(self, **kwargs):
    super(Cifar10Dataset, self).__init__(
        name='cifar10',
        fingerprint_key='id',
        **kwargs)


class Cifar100Dataset(_CifarDataset):
  """CIFAR100 dataset builder class."""

  def __init__(self, **kwargs):
    super(Cifar100Dataset, self).__init__(
        name='cifar100',
        fingerprint_key='id',
        **kwargs)


class Cifar10CorruptedDataset(_CifarDataset):
  """CIFAR10-C dataset builder class."""

  def __init__(
      self,
      corruption_type: str,
      severity: int,
      **kwargs):
    """Create a CIFAR10-C tf.data.Dataset builder.

    Args:
      corruption_type: Corruption name.
      severity: Corruption severity, an integer between 1 and 5.
      **kwargs: Additional keyword arguments.
    """
    super(Cifar10CorruptedDataset, self).__init__(
        name=f'cifar10_corrupted/{corruption_type}_{severity}',
        fingerprint_key=None,
        **kwargs)

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

"""CIFAR{10,100} dataset builders."""

import functools
from typing import Any, Dict, Optional

from robustness_metrics.common import types
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds
from uncertainty_baselines.datasets import augment_utils
from uncertainty_baselines.datasets import augmix
from uncertainty_baselines.datasets import base


def normalize_convert_image(input_image, dtype):
  input_image = tf.image.convert_image_dtype(input_image, dtype)
  mean = tf.constant([0.4914, 0.4822, 0.4465], dtype=dtype)
  std = tf.constant([0.2023, 0.1994, 0.2010], dtype=dtype)
  return (input_image - mean) / std


def _augmix(image, params, augmenter, dtype):
  """Apply augmix augmentation to image."""
  depth = params['augmix_depth']
  width = params['augmix_width']
  prob_coeff = params['augmix_prob_coeff']
  count = params['aug_count']

  augmented = [
      augmix.augment_and_mix(image, depth, width, prob_coeff, augmenter, dtype)
      for _ in range(count)
  ]
  image = normalize_convert_image(image, dtype)
  return tf.stack([image] + augmented, 0)


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
      use_bfloat16: bool = False,
      aug_params: Dict[str, Any] = None,
      data_dir: str = None,
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
      use_bfloat16: Whether or not to load the data in bfloat16 or float32.
      aug_params: hyperparameters for the data augmentation pre-processing.
      data_dir: Directory to read/write data, that is passed to the
        tfds dataset_builder as a data_dir parameter .
    """
    self._normalize = normalize
    dataset_builder = tfds.builder(
        name, try_gcs=try_gcs,
        data_dir=data_dir)
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

    self._use_bfloat16 = use_bfloat16
    if aug_params is None:
      aug_params = {}
    self._adaptive_mixup = aug_params.get('adaptive_mixup', False)
    ensemble_size = aug_params.get('ensemble_size', 1)
    if self._adaptive_mixup and 'mixup_coeff' not in aug_params:
      # Hard target in the first epoch!
      aug_params['mixup_coeff'] = tf.ones([ensemble_size, 10])
    self._aug_params = aug_params

  def _create_process_example_fn(self) -> base.PreProcessFn:

    def _example_parser(example: types.Features) -> types.Features:
      """A pre-process function to return images in [0, 1]."""
      image = example['image']
      image_dtype = tf.bloat16 if self._use_bfloat16 else tf.float32
      if self._is_training:
        image_shape = tf.shape(image)
        # Expand the image by 2 pixels, then crop back down to 32x32.
        image = tf.image.resize_with_crop_or_pad(
            image, image_shape[0] + 4, image_shape[1] + 4)
        image = tf.image.random_crop(image, [image_shape[0], image_shape[0], 3])
        image = tf.image.random_flip_left_right(image)

        # Only random augment for now.
        random_augment = self._aug_params.get('random_augment', False)
        if random_augment:
          count = self._aug_params['aug_count']
          augmenter = augment_utils.RandAugment()
          augmented = [augmenter.distort(image) for _ in range(count)]
          image = tf.stack(augmented)

      if self._is_training and self._aug_params.get('augmix', False):
        augmenter = augment_utils.RandAugment()
        image = _augmix(image, self._aug_params, augmenter, image_dtype)

      # Normalize the values of the image to be in [0, 1].
      if image.dtype != tf.uint8:
        raise ValueError(
            'Images need to be type uint8 to use tf.image.convert_image_dtype.')
      # The image has values in the range [0, 1].
      # Optionally normalize by the dataset statistics.
      if self._normalize:
        image = normalize_convert_image(image, image_dtype)
      else:
        image = tf.image.convert_image_dtype(image, image_dtype)
      parsed_example = example.copy()
      parsed_example['features'] = image

      parsed_example['labels'] = tf.cast(example['label'], tf.int32)
      mixup_alpha = self._aug_params.get('mixup_alpha', 0)
      label_smoothing = self._aug_params.get('label_smoothing', 0.)
      should_onehot = mixup_alpha > 0 or label_smoothing > 0
      if should_onehot:
        parsed_example['labels'] = tf.one_hot(parsed_example['labels'], 10)

      del parsed_example['image']
      del parsed_example['label']
      return parsed_example

    return _example_parser

  def _create_process_batch_fn(
      self,
      batch_size: int) -> Optional[base.PreProcessFn]:
    if self._is_training and self._aug_params.get('mixup_alpha', 0) > 0:
      if self._adaptive_mixup:
        return functools.partial(
            augmix.adaptive_mixup, batch_size, self._aug_params)
      else:
        return functools.partial(augmix.mixup, batch_size, self._aug_params)
    return None


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

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

from typing import Any, Dict, Optional, Union

import numpy as np
from robustness_metrics.common import types
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds
from uncertainty_baselines.datasets import augment_utils
from uncertainty_baselines.datasets import augmix
from uncertainty_baselines.datasets import base


# We use the convention of using mean = np.mean(train_images, axis=(0,1,2))
# and std = np.std(train_images, axis=(0,1,2)).
CIFAR10_MEAN = np.array([0.4914, 0.4822, 0.4465])
CIFAR10_STD = np.array([0.2470, 0.2435, 0.2616])
# Previously we used std = np.mean(np.std(train_images, axis=(1, 2)), axis=0)
# which gave std = tf.constant([0.2023, 0.1994, 0.2010], dtype=dtype), however
# we change convention to use the std over the entire training set instead.


def _tuple_dict_fn_converter(fn, *args):

  def dict_fn(batch_dict):
    images, labels = fn(*args, batch_dict['features'], batch_dict['labels'])
    return {'features': images, 'labels': labels}

  return dict_fn




class _CifarDataset(base.BaseDataset):
  """CIFAR dataset builder abstract class."""

  def __init__(
      self,
      name: str,
      fingerprint_key: str,
      split: str,
      seed: Optional[Union[int, tf.Tensor]] = None,
      validation_percent: float = 0.0,
      shuffle_buffer_size: Optional[int] = None,
      num_parallel_parser_calls: int = 64,
      drop_remainder: bool = False,
      normalize: bool = True,
      try_gcs: bool = False,
      download_data: bool = False,
      use_bfloat16: bool = False,
      aug_params: Optional[Dict[str, Any]] = None,
      data_dir: Optional[str] = None,
      is_training: Optional[bool] = None,
      is_cifar10h: Optional[bool] = False):
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
      seed: the seed used as a source of randomness.
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
        tfds dataset_builder as a data_dir parameter.
      is_training: Whether or not the given `split` is the training split. Only
        required when the passed split is not one of ['train', 'validation',
        'test', tfds.Split.TRAIN, tfds.Split.VALIDATION, tfds.Split.TEST].
      is_cifar10h: whether or not to load the Cifar10H labels for Cifar10.
    """
    self._normalize = normalize
    dataset_builder = tfds.builder(
        name, try_gcs=try_gcs,
        data_dir=data_dir)
    if is_training is None:
      is_training = split in ['train', tfds.Split.TRAIN]
    new_split = base.get_validation_percent_split(
        dataset_builder, validation_percent, split)
    super().__init__(
        name=name,
        dataset_builder=dataset_builder,
        split=new_split,
        seed=seed,
        is_training=is_training,
        shuffle_buffer_size=shuffle_buffer_size,
        num_parallel_parser_calls=num_parallel_parser_calls,
        drop_remainder=drop_remainder,
        fingerprint_key=fingerprint_key,
        download_data=download_data,
        cache=True)

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
      image_dtype = tf.bfloat16 if self._use_bfloat16 else tf.float32
      use_augmix = self._aug_params.get('augmix', False)
      if self._is_training:
        image_shape = tf.shape(image)
        # Expand the image by 2 pixels, then crop back down to 32x32.
        image = tf.image.resize_with_crop_or_pad(
            image, image_shape[0] + 4, image_shape[1] + 4)
        # Note that self._seed will already be shape (2,), as is required for
        # stateless random ops, and so will per_example_step_seed.
        per_example_step_seed = tf.random.experimental.stateless_fold_in(
            self._seed, example[self._enumerate_id_key])
        # per_example_step_seeds will be of size (num, 3).
        # First for random_crop, second for flip, third optionally for
        # RandAugment, and foruth optionally for Augmix.
        per_example_step_seeds = tf.random.experimental.stateless_split(
            per_example_step_seed, num=4)
        image = tf.image.stateless_random_crop(
            image,
            (image_shape[0], image_shape[0], 3),
            seed=per_example_step_seeds[0])
        image = tf.image.stateless_random_flip_left_right(
            image,
            seed=per_example_step_seeds[1])

        # Only random augment for now.
        if self._aug_params.get('random_augment', False):
          count = self._aug_params['aug_count']
          augment_seeds = tf.random.experimental.stateless_split(
              per_example_step_seeds[2], num=count)
          augmenter = augment_utils.RandAugment()
          augmented = [
              augmenter.distort(image, seed=augment_seeds[c])
              for c in range(count)
          ]
          image = tf.stack(augmented)

        if use_augmix:
          augmenter = augment_utils.RandAugment()
          image = augmix.do_augmix(
              image, self._aug_params, augmenter, image_dtype,
              mean=CIFAR10_MEAN, std=CIFAR10_STD,
              seed=per_example_step_seeds[3])

      # The image has values in the range [0, 1].
      # Optionally normalize by the dataset statistics.
      if not use_augmix:
        if self._normalize:
          image = augmix.normalize_convert_image(
              image, image_dtype, mean=CIFAR10_MEAN, std=CIFAR10_STD)
        else:
          image = tf.image.convert_image_dtype(image, image_dtype)
      parsed_example = example.copy()
      parsed_example['features'] = image

      # Note that labels are always float32, even when images are bfloat16.
      mixup_alpha = self._aug_params.get('mixup_alpha', 0)
      label_smoothing = self._aug_params.get('label_smoothing', 0.)
      should_onehot = mixup_alpha > 0 or label_smoothing > 0

      labels = example['label']

      if should_onehot:
        num_classes = 100 if self.name == 'cifar100' else 10
        parsed_example['labels'] = tf.one_hot(
            labels, num_classes, dtype=tf.float32)
      else:
        parsed_example['labels'] = tf.cast(labels, tf.float32)


      del parsed_example['image']
      del parsed_example['label']
      return parsed_example

    return _example_parser

  def _create_process_batch_fn(
      self,
      batch_size: int) -> Optional[base.PreProcessFn]:
    if self._is_training and self._aug_params.get('mixup_alpha', 0) > 0:
      if self._adaptive_mixup:
        return _tuple_dict_fn_converter(
            augmix.adaptive_mixup, batch_size, self._aug_params)
      else:
        return _tuple_dict_fn_converter(
            augmix.mixup, batch_size, self._aug_params)
    return None


class Cifar10Dataset(_CifarDataset):
  """CIFAR10 dataset builder class."""

  def __init__(self, **kwargs):
    super().__init__(
        name='cifar10',
        fingerprint_key='id',
        **kwargs)


class Cifar100Dataset(_CifarDataset):
  """CIFAR100 dataset builder class."""

  def __init__(self, **kwargs):
    super().__init__(
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
    super().__init__(
        name=f'cifar10_corrupted/{corruption_type}_{severity}',
        fingerprint_key=None,
        **kwargs)



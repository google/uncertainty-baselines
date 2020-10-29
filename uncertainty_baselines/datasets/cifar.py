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
"""CIFAR{10,100} dataset builders."""

from typing import Any, Dict, Optional

import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds
from uncertainty_baselines.datasets import base


class _CifarDataset(base.BaseDataset):
  """CIFAR dataset builder abstract class."""

  def __init__(
      self,
      name: str,
      batch_size: int,
      eval_batch_size: int,
      validation_percent: float = 0.0,
      shuffle_buffer_size: int = None,
      num_parallel_parser_calls: int = 64,
      data_dir: Optional[str] = None,
      normalize: bool = True,
      **unused_kwargs: Dict[str, Any]):
    """Create a CIFAR10 or CIFAR100 tf.data.Dataset builder.

    Args:
      name: the name of this dataset, either 'cifar10' or 'cifar100'.
      batch_size: the training batch size.
      eval_batch_size: the validation and test batch size.
      validation_percent: the percent of the training set to use as a validation
        set.
      shuffle_buffer_size: the number of example to use in the shuffle buffer
        for tf.data.Dataset.shuffle().
      num_parallel_parser_calls: the number of parallel threads to use while
        preprocessing in tf.data.Dataset.map().
      data_dir: optional dir to save TFDS data to. If none then the local
        filesystem is used. Required for using TPUs on Cloud.
      normalize: whether or not to normalize each image by the CIFAR dataset
        mean and stddev.
    """
    if validation_percent < 0.0 or validation_percent >= 1.0:
      raise ValueError(
          'validation_percent must be in [0, 1), received {}.'.format(
              validation_percent))
    self._normalize = normalize
    num_train_examples = 50000
    num_validation_examples = int(num_train_examples * validation_percent)
    num_train_examples -= num_validation_examples
    super(_CifarDataset, self).__init__(
        name=name,
        num_train_examples=num_train_examples,
        num_validation_examples=num_validation_examples,
        num_test_examples=10000,
        batch_size=batch_size,
        eval_batch_size=eval_batch_size,
        shuffle_buffer_size=shuffle_buffer_size,
        num_parallel_parser_calls=num_parallel_parser_calls,
        data_dir=data_dir)

  def _read_examples(self, split: base.Split) -> tf.data.Dataset:
    """Creates a dataset to be processed by _create_process_example_fn."""
    if split == base.Split.TEST:
      return tfds.load(
          self.name + ':3.*.*',
          split='test',
          **self._tfds_kwargs)

    if split == base.Split.TRAIN:
      if self._num_validation_examples == 0:
        train_split = 'train'
      else:
        train_split = tfds.core.ReadInstruction(
            'train', to=-self._num_validation_examples, unit='abs')
      return tfds.load(
          self.name + ':3.*.*',
          split=train_split,
          **self._tfds_kwargs)
    if split == base.Split.VAL:
      if self._num_validation_examples == 0:
        raise ValueError(
            'No validation set provided. Set `validation_percent > 0.0` to '
            'take a subset of the training set as validation.')
      val_split = tfds.core.ReadInstruction(
          'train', from_=-self._num_validation_examples, unit='abs')
      return tfds.load(
          self.name + ':3.*.*',
          split=val_split,
          **self._tfds_kwargs)

  def _create_process_example_fn(self, split: base.Split) -> base.PreProcessFn:

    def _example_parser(example: Dict[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
      """A pre-process function to return images in [0, 1]."""
      image = example['image']
      label = example['label']
      if self._is_training(split):
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
      return {'features': image, 'labels': tf.cast(label, tf.int32)}

    return _example_parser


class Cifar10Dataset(_CifarDataset):
  """CIFAR10 dataset builder class."""

  def __init__(self, **kwargs):
    super(Cifar10Dataset, self).__init__(name='cifar10', **kwargs)


class Cifar100Dataset(_CifarDataset):
  """CIFAR100 dataset builder class."""

  def __init__(self, **kwargs):
    super(Cifar100Dataset, self).__init__(name='cifar100', **kwargs)

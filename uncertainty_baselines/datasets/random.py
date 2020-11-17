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
"""Random noise dataset builder."""

from typing import Any, Dict, Iterable

import numpy as np
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds

from uncertainty_baselines.datasets import base


class _RandomDatasetBuilder(tfds.core.DatasetBuilder):
  """Minimal TFDS DatasetBuilder for a random dataset."""
  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
  }

  def __init__(self, image_shape, **kwargs):
    self._image_shape = image_shape
    self._num_train_examples = 50000
    self._num_validation_examples = 10000
    self._num_test_examples = 10000
    super(_RandomDatasetBuilder, self).__init__(**kwargs)

  def _download_and_prepare(self, dl_manager, download_config=None):
    """Downloads and prepares dataset for reading."""
    raise NotImplementedError(
        'Must provide a data_dir with the files already downloaded to.')

  def _as_dataset(
      self,
      split: tfds.Split,
      decoders=None,
      read_config=None,
      shuffle_files=False) -> tf.data.Dataset:
    raise NotImplementedError

  # Note that we override `as_dataset` instead of `_as_dataset` to avoid any
  # `data_dir` reading logic.
  def as_dataset(
      self,
      split: tfds.Split,
      *,
      batch_size=None,
      decoders=None,
      read_config=None,
      shuffle_files=False,
      as_supervised=False) -> tf.data.Dataset:
    """Constructs a `tf.data.Dataset`."""
    del batch_size
    del decoders
    del read_config
    del shuffle_files
    del as_supervised
    if split == tfds.Split.TRAIN:
      return tf.data.Dataset.range(self._num_train_examples)
    if split == tfds.Split.VALIDATION:
      return tf.data.Dataset.range(self._num_validation_examples)
    if split == tfds.Split.TEST:
      return tf.data.Dataset.range(self._num_test_examples)
    raise ValueError('Unsupported split given: {}.'.format(split))

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the `tfds.core.DatasetInfo` object."""
    features = {
        'labels': tfds.features.ClassLabel(num_classes=2),
        'features': tfds.features.Tensor(
            shape=self._image_shape, dtype=tf.float32),
    }
    info = tfds.core.DatasetInfo(
        builder=self,
        description='Random noise dataset.',
        features=tfds.features.FeaturesDict(features),
        metadata=None)
    split_dict = tfds.core.SplitDict('criteo')
    # Instead of having a single element shard_lengths, we should really have a
    # list of the number of elements in each file shard in each split.
    split_dict.add(tfds.core.SplitInfo(
        name=tfds.Split.TRAIN,
        shard_lengths=[self._num_train_examples]))
    split_dict.add(tfds.core.SplitInfo(
        name=tfds.Split.VALIDATION,
        shard_lengths=[self._num_validation_examples]))
    split_dict.add(tfds.core.SplitInfo(
        name=tfds.Split.TEST,
        shard_lengths=[self._num_test_examples]))
    info.update_splits_if_different(split_dict)
    return info


class _RandomNoiseDataset(base.BaseDataset):
  """Random Image dataset builder abstract class."""

  def __init__(
      self,
      name: str,
      split: str,
      image_shape: Iterable[int] = (32, 32, 3),
      shuffle_buffer_size: int = None,
      num_parallel_parser_calls: int = 64,
      eval_filter_class_id: int = -1,
      data_mode: str = 'ind',
      data_dir: str = None,
      try_gcs: bool = False,
      download_data: bool = False,
      **unused_kwargs: Dict[str, Any]):
    """Create a Random Image tf.data.Dataset builder.

    Args:
      name: the name of this dataset, either 'random_gaussian' or
        'random_rademacher'.
      split: a dataset split, either a custom tfds.Split or one of the
        tfds.Split enums [TRAIN, VALIDAITON, TEST] or their lowercase string
        names.
      image_shape: the image shape for random images to be generated. By
        default, images are generated in the shape (32, 32, 3).
      shuffle_buffer_size: the number of example to use in the shuffle buffer
        for tf.data.Dataset.shuffle().
      num_parallel_parser_calls: the number of parallel threads to use while
        preprocessing in tf.data.Dataset.map().
      eval_filter_class_id: evalulate inputs from a particular class only.
      data_mode: either 'ind' or 'ood' to decide whether to read in-distribution
        data or out-of-domain data.
      data_dir: path to a directory containing the Genomics OOD dataset, with
        filenames train-*-of-*', 'validate.tfr', 'test.tfr'.
      try_gcs: Whether or not to try to use the GCS stored versions of dataset
        files. Currently unsupported.
      download_data: Whether or not to download data before loading. Currently
        unsupported.
    """
    self._image_shape = image_shape
    self._split_seed = {
        tfds.Split.TRAIN: 0,
        tfds.Split.VALIDATION: 1,
        tfds.Split.TEST: 2,
    }
    super(_RandomNoiseDataset, self).__init__(
        name=name,
        dataset_builder=_RandomDatasetBuilder(image_shape=image_shape),
        split=split,
        shuffle_buffer_size=shuffle_buffer_size,
        num_parallel_parser_calls=num_parallel_parser_calls,
        download_data=False)


class RandomGaussianImageDataset(_RandomNoiseDataset):
  """Random Gaussian Image dataset builder abstract class."""

  def __init__(self, **kwargs):
    super(RandomGaussianImageDataset, self).__init__(
        name='random_gaussian', **kwargs)

  def _create_process_example_fn(self) -> base.PreProcessFn:

    def _example_parser(range_val: int) -> Dict[str, tf.Tensor]:
      """Parses a single range integer into stateless image Tensors."""
      seed = [
          self._split_seed[self._split],
          self._split_seed[self._split] + range_val
      ]
      image = tf.random.stateless_normal(
          self._image_shape,
          seed=seed,
          dtype=tf.float32)
      image_min = tf.reduce_min(image)
      image_max = tf.reduce_max(image)
      # Normalize the values of the image to be in [-1, 1].
      image = 2.0 * (image - image_min) / (image_max - image_min) - 1.0
      label = tf.zeros([], tf.int32)
      return {'features': image, 'labels': label}

    return _example_parser


class RandomRademacherImageDataset(_RandomNoiseDataset):
  """Random Rademacher Image dataset builder abstract class."""

  def __init__(self, **kwargs):
    super(RandomRademacherImageDataset, self).__init__(
        name='random_rademacher', **kwargs)

  def _create_process_example_fn(self) -> base.PreProcessFn:

    def _example_parser(range_val: int) -> Dict[str, tf.Tensor]:
      """Parses a single range integer into stateless image Tensors."""
      seed = [
          self._split_seed[self._split],
          self._split_seed[self._split] + range_val
      ]
      image = tf.random.stateless_categorical(
          tf.math.log([[0.5, 0.5]]),
          np.prod(self._image_shape),
          seed=seed,
          dtype=tf.int32)
      image = tf.reshape(tf.cast(image, tf.float32), self._image_shape)
      image = 2.0 * (image - 0.5)
      label = tf.zeros([], tf.int32)
      return {'features': image, 'labels': label}

    return _example_parser

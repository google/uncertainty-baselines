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

from typing import Any, Dict, Iterable

import numpy as np
import tensorflow.compat.v2 as tf

from uncertainty_baselines.datasets import base


class _RandomNoiseDataset(base.BaseDataset):
  """Random Image dataset builder abstract class."""

  def __init__(
      self,
      name: str,
      batch_size: int,
      eval_batch_size: int,
      image_shape: Iterable[int] = (32, 32, 3),
      num_train_examples: int = 50000,
      num_validation_examples: int = 10000,
      num_test_examples: int = 10000,
      shuffle_buffer_size: int = None,
      num_parallel_parser_calls: int = 64,
      **unused_kwargs: Dict[str, Any]):
    """Create a Random Image tf.data.Dataset builder.

    Args:
      name: the name of this dataset, either 'random_gaussian' or
        'random_rademacher'.
      batch_size: the training batch size.
      eval_batch_size: the validation and test batch size.
      image_shape: the image shape for random images to be generated. By default
       , images are generated in the shape (32, 32, 3).
      num_train_examples: the number of training images to generate.
      num_validation_examples: the number of validation images to generate.
      num_test_examples: the number of test images to generate.
      shuffle_buffer_size: the number of example to use in the shuffle buffer
        for tf.data.Dataset.shuffle().
      num_parallel_parser_calls: the number of parallel threads to use while
        preprocessing in tf.data.Dataset.map().
    """
    self._image_shape = image_shape
    self._split_seed = {
        base.Split.TRAIN: 0,
        base.Split.VAL: 1,
        base.Split.TEST: 2,
    }
    super(_RandomNoiseDataset, self).__init__(
        name=name,
        num_train_examples=num_train_examples,
        num_validation_examples=num_validation_examples,
        num_test_examples=num_test_examples,
        batch_size=batch_size,
        eval_batch_size=eval_batch_size,
        shuffle_buffer_size=shuffle_buffer_size,
        num_parallel_parser_calls=num_parallel_parser_calls)

  def _read_examples(self, split: base.Split) -> tf.data.Dataset:
    """Creates a dataset to be processed by _create_process_example_fn."""
    if split == base.Split.TEST:
      return tf.data.Dataset.range(self._num_test_examples)
    if split == base.Split.TRAIN:
      return tf.data.Dataset.range(self._num_train_examples)
    if split == base.Split.VAL:
      return tf.data.Dataset.range(self._num_validation_examples)


class RandomGaussianImageDataset(_RandomNoiseDataset):
  """Random Gaussian Image dataset builder abstract class."""

  def __init__(self, **kwargs):
    super(RandomGaussianImageDataset, self).__init__(
        name="random_gaussian", **kwargs)

  def _create_process_example_fn(self, split: base.Split) -> base.PreProcessFn:

    def _example_parser(range_val: int) -> Dict[str, tf.Tensor]:
      """Parses a single range integer into stateless image Tensors."""
      image = tf.random.stateless_normal(
          self._image_shape,
          [self._split_seed[split], self._split_seed[split] + range_val],
          dtype=tf.float32)
      image_min = tf.reduce_min(image)
      image_max = tf.reduce_max(image)
      # Normalize the values of the image to be in [-1, 1].
      image = 2.0 * (image - image_min) / (image_max - image_min) - 1.0
      label = tf.zeros([], tf.int32)
      return {"features": image, "labels": label}

    return _example_parser


class RandomRademacherImageDataset(_RandomNoiseDataset):
  """Random Rademacher Image dataset builder abstract class."""

  def __init__(self, **kwargs):
    super(RandomRademacherImageDataset, self).__init__(
        name="random_rademacher", **kwargs)

  def _create_process_example_fn(self, split: base.Split) -> base.PreProcessFn:

    def _example_parser(range_val: int) -> Dict[str, tf.Tensor]:
      """Parses a single range integer into stateless image Tensors."""
      image = tf.random.stateless_categorical(
          tf.math.log([[0.5, 0.5]]), np.prod(self._image_shape),
          [self._split_seed[split], self._split_seed[split] + range_val],
          dtype=tf.int32)
      image = tf.reshape(tf.cast(image, tf.float32), self._image_shape)
      image = 2.0 * (image - 0.5)
      label = tf.zeros([], tf.int32)
      return {"features": image, "labels": label}

    return _example_parser

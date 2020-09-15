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
"""MNIST dataset builder."""

from typing import Any, Dict

import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds
from uncertainty_baselines.datasets import base


class MnistDataset(base.BaseDataset):
  """MNIST dataset builder class."""

  def __init__(
      self,
      batch_size: int,
      eval_batch_size: int,
      shuffle_buffer_size: int = None,
      num_parallel_parser_calls: int = 64,
      **unused_kwargs: Dict[str, Any]):
    """Create an MNIST tf.data.Dataset builder.

    Args:
      batch_size: the training batch size.
      eval_batch_size: the validation and test batch size.
      shuffle_buffer_size: the number of example to use in the shuffle buffer
        for tf.data.Dataset.shuffle().
      num_parallel_parser_calls: the number of parallel threads to use while
        preprocessing in tf.data.Dataset.map().
    """
    super(MnistDataset, self).__init__(
        name='mnist',
        num_train_examples=50000,
        num_validation_examples=10000,
        num_test_examples=10000,
        batch_size=batch_size,
        eval_batch_size=eval_batch_size,
        shuffle_buffer_size=shuffle_buffer_size,
        num_parallel_parser_calls=num_parallel_parser_calls)

  def _read_examples(self, split: base.Split) -> tf.data.Dataset:
    """We use the original 'validation' set as test."""
    if split == base.Split.TRAIN:
      train_split = tfds.core.ReadInstruction(
          'train', to=-self._num_validation_examples, unit='abs')
      return tfds.load('mnist', split=train_split)
    elif split == base.Split.VAL:
      val_split = tfds.core.ReadInstruction(
          'train', from_=-self._num_validation_examples, unit='abs')
      return tfds.load('mnist', split=val_split)
    elif split == base.Split.TEST:
      return tfds.load('mnist', split='test')
    else:
      raise ValueError(
          'Invalid dataset split in _read_examples: {}'.format(split))

  def _create_process_example_fn(self, split: base.Split) -> base.PreProcessFn:
    del split

    def _example_parser(example: Dict[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
      """A pre-process function to return images in [0, 1]."""
      image = example['image']
      image = tf.image.convert_image_dtype(image, tf.float32)
      label = tf.cast(example['label'], tf.int32)
      parsed_example = {
          'features': image,
          'labels': label,
      }
      return parsed_example

    return _example_parser

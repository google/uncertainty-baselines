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
      batch_size: int,
      eval_batch_size: int,
      validation_percent: float = 0.0,
      shuffle_buffer_size: int = None,
      num_parallel_parser_calls: int = 64,
      data_dir: Optional[str] = None,
      **unused_kwargs: Dict[str, Any]):
    """Create an ImageNet tf.data.Dataset builder.

    Args:
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
    """
    num_train_examples = 1281167
    num_validation_examples = int(num_train_examples * validation_percent)
    num_train_examples -= num_validation_examples
    super(ImageNetDataset, self).__init__(
        name='imagenet',
        num_train_examples=num_train_examples,
        num_validation_examples=num_validation_examples,
        num_test_examples=50000,
        batch_size=batch_size,
        eval_batch_size=eval_batch_size,
        shuffle_buffer_size=shuffle_buffer_size,
        num_parallel_parser_calls=num_parallel_parser_calls,
        data_dir=data_dir)

  def _read_examples(self, split: base.Split) -> tf.data.Dataset:
    """Read ImageNet examples. We use the original 'validation' set as test."""
    # TODO(znado): rewrite this to use tfrecords and/or tar.gz files, see
    # cl/295280383 and https://github.com/tensorflow/datasets/issues/1739.
    if split == base.Split.TRAIN:
      if self._num_validation_examples == 0:
        train_split = 'train'
      else:
        train_split = tfds.core.ReadInstruction(
            'train', to=-self._num_validation_examples, unit='abs')
      return tfds.load(
          'imagenet2012:5.0.0',
          split=train_split,
          try_gcs=True,
          data_dir=self._data_dir)
    elif split == base.Split.VAL:
      if self._num_validation_examples == 0:
        raise ValueError(
            'No validation set provided. Set `validation_percent > 0.0` to '
            'take a subset of the training set as validation.')
      val_split = tfds.core.ReadInstruction(
          'train', from_=-self._num_validation_examples, unit='abs')
      return tfds.load(
          'imagenet2012:5.0.0',
          split=val_split,
          try_gcs=True,
          data_dir=self._data_dir)
    elif split == base.Split.TEST:
      return tfds.load(
          'imagenet2012',
          split='validation',
          try_gcs=True,
          data_dir=self._data_dir)

  def _create_process_example_fn(self, split: base.Split) -> base.PreProcessFn:
    """Create a pre-process function to return images in [0, 1]."""

    def _example_parser(example: Dict[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
      """Preprocesses ImageNet image Tensors using inception_preprocessing."""
      # `preprocess_image` returns images in [-1, 1].
      image = inception_preprocessing.preprocess_image(
          example['image'],
          height=224,
          width=224,
          is_training=self._is_training(split))
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

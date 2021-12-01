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

"""Cityscapes dataset builder.

We have an option to use a percent of the training dataset as a validation set,
and treat the original validation set as the test set. This is because the test
set in cityscapes only includes a subset of the classes per image.
"""
from typing import Dict, Optional

import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds
from uncertainty_baselines.datasets import base


class CityscapesDataset(base.BaseDataset):
  """Cityscapes dataset builder class."""

  def __init__(
      self,
      split: str,
      validation_percent: float = 0.0,
      shuffle_buffer_size: Optional[int] = 1,
      num_parallel_parser_calls: int = 1,
      try_gcs: bool = False,
      download_data: bool = False,
      data_dir: Optional[str] = None,
      is_training: Optional[bool] = None,
      use_bfloat16: bool = False,
      normalize_input: bool = False,
      image_height: int = 1024,
      image_width: int = 2048,
      one_hot: bool = False,
      include_file_name: bool = False,
  ):
    """Create an Cityscapes tf.data.Dataset builder.

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
      data_dir: Directory to read/write data, that is passed to the
        tfds dataset_builder as a data_dir parameter.
      is_training: Whether or not the given `split` is the training split. Only
        required when the passed split is not one of ['train', 'validation',
        'test', tfds.Split.TRAIN, tfds.Split.VALIDATION, tfds.Split.TEST].
      use_bfloat16: Whether or not to use bfloat16 or float32 images.
      normalize_input: Whether or not to normalize images by the ImageNet mean
        and stddev.
      image_height: The height of the image in pixels.
      image_width: The height of the image in pixels.
      one_hot: whether or not to use one-hot labels.
      include_file_name: Whether or not to include a string file_name field in
        each example. Since this field is a string, it is not compatible with
        TPUs.
    """
    name = 'cityscapes'
    dataset_builder = tfds.builder(name, try_gcs=try_gcs, data_dir=data_dir)
    if is_training is None:
      is_training = split in ['train', tfds.Split.TRAIN]
    new_split = base.get_validation_percent_split(
        dataset_builder,
        validation_percent,
        split,
        test_split=tfds.Split.VALIDATION)

    super().__init__(
        name=name,
        dataset_builder=dataset_builder,
        split=new_split,
        is_training=is_training,
        shuffle_buffer_size=shuffle_buffer_size,
        num_parallel_parser_calls=num_parallel_parser_calls,
        download_data=download_data)

    self._use_bfloat16 = use_bfloat16
    self._normalize_input = normalize_input
    self._image_height = image_height
    self._image_width = image_width
    self._one_hot = one_hot
    self._include_file_name = include_file_name

  def _create_process_example_fn(self) -> base.PreProcessFn:
    """Create a pre-process function to return images in [0, 1]."""

    def _example_parser(example: Dict[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
      """Preprocesses Cityscapes image Tensors."""
      image = example['image_left']

      if self._normalize_input:
        image = tf.cast(image, tf.float32) / 255.
      if self._use_bfloat16:
        image = tf.cast(image, tf.bfloat16)

      if self._one_hot:
        label = tf.one_hot(example['segmentation_label'], 34, dtype=tf.int32)
        label = tf.reshape(
            label, shape=[self._image_height, self._image_width, 34])

      else:
        label = tf.cast(example['segmentation_label'], tf.int32)

      parsed_example = {
          'features': image,
          'labels': label,
      }
      if self._include_file_name and 'file_name' in example:
        parsed_example['file_name'] = example['file_name']
      return parsed_example

    return _example_parser

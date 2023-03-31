# coding=utf-8
# Copyright 2023 The Uncertainty Baselines Authors.
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

"""SVHN dataset builder."""

from typing import Dict, Optional

import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds
from uncertainty_baselines.datasets import base


class DtdDataset(base.BaseDataset):
  """DTD (Describable Textures Dataset) dataset builder class."""

  def __init__(self,
               split: str,
               validation_percent: float = 0.0,
               shuffle_buffer_size: Optional[int] = None,
               num_parallel_parser_calls: int = 64,
               drop_remainder: bool = True,
               try_gcs: bool = False,
               download_data: bool = False,
               data_dir: Optional[str] = None,
               normalize_by_cifar: bool = False,
               is_training: Optional[bool] = None):
    """Create an DTD tf.data.Dataset builder.

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
      drop_remainder: whether or not to drop the last batch of data if the
        number of points is not exactly equal to the batch size. This option
        needs to be True for running on TPUs.
      try_gcs: Whether or not to try to use the GCS stored versions of dataset
        files.
      download_data: Whether or not to download data before loading.
      data_dir: Directory to read/write data, that is passed to the tfds
        dataset_builder as a data_dir parameter.
      normalize_by_cifar: whether or not to normalize each image by the CIFAR
        dataset mean and stddev.
      is_training: Whether or not the given `split` is the training split. Only
        required when the passed split is not one of ['train', 'validation',
        'test', tfds.Split.TRAIN, tfds.Split.VALIDATION, tfds.Split.TEST].
    """
    self._normalize_by_cifar = normalize_by_cifar
    name = 'dtd'
    dataset_builder = tfds.builder(name, try_gcs=try_gcs, data_dir=data_dir)
    if is_training is None:
      is_training = split in ['train', tfds.Split.TRAIN]
    new_split = base.get_validation_percent_split(
        dataset_builder, validation_percent, split)
    super().__init__(
        name=name,
        dataset_builder=dataset_builder,
        split=new_split,
        is_training=is_training,
        shuffle_buffer_size=shuffle_buffer_size,
        num_parallel_parser_calls=num_parallel_parser_calls,
        drop_remainder=drop_remainder,
        download_data=download_data)

  def _create_process_example_fn(self) -> base.PreProcessFn:

    def _example_parser(example: Dict[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
      """A pre-process function to return images in [0, 1]."""
      image = example['image']
      image = tf.image.convert_image_dtype(image, tf.float32)
      label = tf.cast(example['label'], tf.float32)
      if self._normalize_by_cifar:
        # Follow the same preprocess functions as in
        # https://github.com/hendrycks/outlier-exposure/blob/e6ede98a5474a0620d9befa50b38eaf584df4401/SVHN/test.py#L237
        image = tf.image.resize(image, size=(32, 32), method='bilinear')
        image = tf.image.central_crop(image, central_fraction=1.0)
        # We use the convention of mean = np.mean(train_images, axis=(0,1,2))
        # and std = np.std(train_images, axis=(0,1,2)).
        mean = tf.constant([0.4914, 0.4822, 0.4465])
        std = tf.constant([0.2470, 0.2435, 0.2616])
        # Previously, std = np.mean(np.std(train_images, axis=(1, 2)), axis=0)
        # which gave std = tf.constant([0.2023, 0.1994, 0.2010], dtype=dtype).
        # However, we change convention to use the std over the entire training
        # set instead.
        image = (image - mean) / std
      parsed_example = {
          'features': image,
          'labels': label,
      }
      return parsed_example

    return _example_parser

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
"""Kaggle diabetic retinopathy detection dataset builder."""

from typing import Any, Dict, Optional

import tensorflow as tf
import tensorflow_datasets as tfds
from uncertainty_baselines.datasets import base


class DiabeticRetinopathyDetectionDataset(base.BaseDataset):
  """Kaggle diabetic retinopathy detection dataset builder class."""

  def __init__(
      self,
      split: str,
      shuffle_buffer_size: Optional[int] = None,
      num_parallel_parser_calls: int = 64,
      data_dir: Optional[str] = None,
      download_data: bool = False,
      is_training: Optional[bool] = None,
      **unused_kwargs: Dict[str, Any]):
    """Create a Kaggle diabetic retinopathy detection tf.data.Dataset builder.

    Args:
      split: a dataset split, either a custom tfds.Split or one of the
        tfds.Split enums [TRAIN, VALIDAITON, TEST] or their lowercase string
        names.
      shuffle_buffer_size: the number of example to use in the shuffle buffer
        for tf.data.Dataset.shuffle().
      num_parallel_parser_calls: the number of parallel threads to use while
        preprocessing in tf.data.Dataset.map().
      data_dir: optional dir to save TFDS data to. If none then the local
        filesystem is used. Required for using TPUs on Cloud.
      download_data: Whether or not to download data before loading.
      is_training: Whether or not the given `split` is the training split. Only
        required when the passed split is not one of ['train', 'validation',
        'test', tfds.Split.TRAIN, tfds.Split.VALIDATION, tfds.Split.TEST].
    """
    if is_training is None:
      is_training = split in ['train', tfds.Split.TRAIN]
    dataset_builder = tfds.builder(
        'diabetic_retinopathy_detection/btgraham-300', data_dir=data_dir)
    super(DiabeticRetinopathyDetectionDataset, self).__init__(
        name='diabetic_retinopathy_detection',
        dataset_builder=dataset_builder,
        split=split,
        is_training=is_training,
        shuffle_buffer_size=shuffle_buffer_size,
        num_parallel_parser_calls=num_parallel_parser_calls,
        download_data=download_data)

  def _create_process_example_fn(self) -> base.PreProcessFn:

    def _example_parser(example: Dict[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
      """A pre-process function to return images in [0, 1]."""
      image = example['image']
      image = tf.image.convert_image_dtype(image, tf.float32)
      image = tf.image.resize(image, size=(512, 512), method='bilinear')
      label = tf.cast(example['label'] > 1, tf.int32)  # Binarise task.
      parsed_example = {
          'features': image,
          'labels': label,
          'name': example['name'],
      }
      return parsed_example

    return _example_parser

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

"""Kaggle diabetic retinopathy detection dataset builder."""

from typing import Dict, Optional

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
      decision_threshold: Optional[str] = 'moderate'
  ):
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
      decision_threshold: specifies where to binarize the labels {0, 1, 2, 3, 4}
        to create the binary classification task.
        'mild': classify {0} vs {1, 2, 3, 4}, i.e., mild DR or worse?
        'moderate': classify {0, 1} vs {2, 3, 4}, i.e., moderate DR or worse?
    """
    if is_training is None:
      is_training = split in ['train', tfds.Split.TRAIN]
    dataset_builder = tfds.builder(
        'diabetic_retinopathy_detection/btgraham-300', data_dir=data_dir)
    super().__init__(
        name='diabetic_retinopathy_detection',
        dataset_builder=dataset_builder,
        split=split,
        is_training=is_training,
        shuffle_buffer_size=shuffle_buffer_size,
        num_parallel_parser_calls=num_parallel_parser_calls,
        download_data=download_data)
    self.decision_threshold = decision_threshold
    print(f'Building Kaggle DR dataset with decision threshold: '
          f'{decision_threshold}.')

  def _create_process_example_fn(self) -> base.PreProcessFn:

    def _example_parser(example: Dict[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
      """A pre-process function to return images in [0, 1]."""
      image = example['image']
      image = tf.image.convert_image_dtype(image, tf.float32)
      image = tf.image.resize(image, size=(512, 512), method='bilinear')

      if self.decision_threshold == 'mild':
        highest_negative_class = 0
      elif self.decision_threshold == 'moderate':
        highest_negative_class = 1
      else:
        raise NotImplementedError

      # Binarize task.
      label = tf.cast(example['label'] > highest_negative_class, tf.int32)

      parsed_example = {
          'features': image,
          'labels': label,
          'name': example['name'],
      }
      return parsed_example

    return _example_parser

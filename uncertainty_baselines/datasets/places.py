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
"""Places-365 dataset builder."""

from typing import Any, Dict, Optional, Union

import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds
from uncertainty_baselines.datasets import base
from uncertainty_baselines.datasets import inception_preprocessing


class Places365Dataset(base.BaseDataset):
  """Places365 dataset builder class."""

  def __init__(
      self,
      split: str,
      seed: Optional[Union[int, tf.Tensor]] = None,
      validation_percent: float = 0.0,
      shuffle_buffer_size: Optional[int] = None,
      num_parallel_parser_calls: int = 64,
      try_gcs: bool = False,
      download_data: bool = False,
      is_training: Optional[bool] = None,
      **unused_kwargs: Dict[str, Any]):
    """Create an Places-365 tf.data.Dataset builder.

    Args:
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
      try_gcs: Whether or not to try to use the GCS stored versions of dataset
        files.
      download_data: Whether or not to download data before loading.
      is_training: Whether or not the given `split` is the training split. Only
        required when the passed split is not one of ['train', 'validation',
        'test', tfds.Split.TRAIN, tfds.Split.VALIDATION, tfds.Split.TEST].
    """
    name = 'places365_small'
    dataset_builder = tfds.builder(name, try_gcs=try_gcs)
    if is_training is None:
      is_training = split in ['train', tfds.Split.TRAIN]
    new_split = base.get_validation_percent_split(
        dataset_builder,
        validation_percent,
        split,
        test_split=tfds.Split.VALIDATION)
    super(Places365Dataset, self).__init__(
        name=name,
        dataset_builder=dataset_builder,
        split=new_split,
        seed=seed,
        is_training=is_training,
        shuffle_buffer_size=shuffle_buffer_size,
        num_parallel_parser_calls=num_parallel_parser_calls,
        download_data=download_data)

  def _create_process_example_fn(self) -> base.PreProcessFn:
    """Create a pre-process function to return images in [0, 1]."""

    def _example_parser(example: Dict[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
      """Preprocesses Places-365 image Tensors using inception_preprocessing."""
      per_example_step_seed = tf.random.experimental.stateless_fold_in(
          self._seed, example[self._enumerate_id_key])
      # `preprocess_image` returns images in [-1, 1].
      image = inception_preprocessing.preprocess_image(
          example['image'],
          height=224,
          width=224,
          seed=per_example_step_seed,
          is_training=self._is_training)
      # Rescale to [0, 1].
      image = (image + 1.0) / 2.0

      label = tf.cast(example['label'], tf.int32)
      return {'features': image, 'labels': label}

    return _example_parser

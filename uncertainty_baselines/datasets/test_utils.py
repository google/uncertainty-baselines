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

"""Utils for testing datasets."""

from typing import Any, Dict, Sequence, Type, TypeVar, Union, Optional

import tensorflow as tf
import tensorflow_datasets as tfds
from uncertainty_baselines.datasets import base


_SPLITS = (tfds.Split.TRAIN, tfds.Split.VALIDATION, tfds.Split.TEST)


class DatasetTest(tf.test.TestCase):
  """Utility class for testing dataset construction."""

  def _testDatasetSize(
      self,
      dataset_class: Type[TypeVar('B', bound=base.BaseDataset)],
      image_size: Sequence[int],
      splits: Sequence[Union[float, str, tfds.Split]] = _SPLITS,
      label_size: Optional[Sequence[int]] = None,
      **kwargs: Dict[str, Any]):
    batch_size_splits = {}
    for split in splits:
      if split in ['train', tfds.Split.TRAIN]:
        batch_size_splits[split] = 9
      else:
        batch_size_splits[split] = 5
    for split, bs in batch_size_splits.items():
      dataset_builder = dataset_class(
          split=split,
          shuffle_buffer_size=20,
          **kwargs)
      dataset = dataset_builder.load(batch_size=bs).take(1)
      element = next(iter(dataset))
      features = element['features']
      labels = element['labels']

      features_shape = features.shape
      labels_shape = labels.shape
      self.assertEqual(features_shape, tuple([bs] + list(image_size)))

      if label_size is None:
        self.assertEqual(labels_shape, (bs,))
      else:
        self.assertEqual(labels_shape, tuple([bs] + list(label_size)))


if __name__ == '__main__':
  tf.test.main()

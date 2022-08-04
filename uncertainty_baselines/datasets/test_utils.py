# coding=utf-8
# Copyright 2022 The Uncertainty Baselines Authors.
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

from typing import Any, Dict, Sequence, Type, TypeVar, Union, Optional, Iterable

import tensorflow as tf
import tensorflow_datasets as tfds
from uncertainty_baselines.datasets import base

_SPLITS = (tfds.Split.TRAIN, tfds.Split.VALIDATION, tfds.Split.TEST)


class DatasetTest(tf.test.TestCase):
  """Utility class for testing dataset construction."""

  def _testDatasetSize(self,
                       dataset_class: Type[TypeVar('B',
                                                   bound=base.BaseDataset)],
                       image_size: Sequence[int],
                       splits: Sequence[Union[float, str,
                                              tfds.Split]] = _SPLITS,
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
          split=split, shuffle_buffer_size=20, **kwargs)
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


class DummyDatasetBuilder(tfds.core.DatasetBuilder):
  """Minimal TFDS DatasetBuilder for a dummy dataset."""
  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
  }

  def __init__(self, image_shape, num_train_examples, num_validation_examples,
               num_test_examples, **kwargs):
    self._image_shape = image_shape
    self._num_train_examples = num_train_examples
    self._num_validation_examples = num_validation_examples
    self._num_test_examples = num_test_examples
    super().__init__(**kwargs)

  def _download_and_prepare(self, dl_manager, download_config=None):
    """Downloads and prepares dataset for reading."""
    raise NotImplementedError(
        'Must provide a data_dir with the files already downloaded to.')

  def _as_dataset(self,
                  split: tfds.Split,
                  decoders=None,
                  read_config=None,
                  shuffle_files=False) -> tf.data.Dataset:
    raise NotImplementedError

  # Note that we override `as_dataset` instead of `_as_dataset` to avoid any
  # `data_dir` reading logic.
  def as_dataset(self,
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
        'labels':
            tfds.features.ClassLabel(num_classes=2),
        'features':
            tfds.features.Tensor(shape=self._image_shape, dtype=tf.float32),
    }
    info = tfds.core.DatasetInfo(
        builder=self,
        description='Dummy dataset.',
        features=tfds.features.FeaturesDict(features),
        metadata=None)

    # Instead of having a single element shard_lengths, we should really have a
    # list of the number of elements in each file shard in each split.
    split_infos = [
        tfds.core.SplitInfo(
            name=tfds.Split.VALIDATION,
            shard_lengths=[self._num_validation_examples],
            num_bytes=0,
        ),
        tfds.core.SplitInfo(
            name=tfds.Split.TEST,
            shard_lengths=[self._num_test_examples],
            num_bytes=0,
        ),
        tfds.core.SplitInfo(
            name=tfds.Split.TRAIN,
            shard_lengths=[self._num_train_examples],
            num_bytes=0,
        ),
    ]
    split_dict = tfds.core.SplitDict(
        split_infos, dataset_name='__dummy_dataset_builder')
    info.set_splits(split_dict)
    return info


class DummyDataset(base.BaseDataset):
  """Dummy dataset builder abstract class."""

  def __init__(self,
               split: str,
               image_shape: Iterable[int] = (32, 32, 3),
               num_train_examples: int = 5,
               num_validation_examples: int = 5,
               num_test_examples: int = 5):
    """Create a dummy tf.data.Dataset builder.

    Args:
      split: a dataset split, either a custom tfds.Split or one of the
        tfds.Split enums [TRAIN, VALIDAITON, TEST] or their lowercase string
        names.
      image_shape: the image shape for random images to be generated. By
        default, images are generated in the shape (32, 32, 3).
      num_train_examples: number of examples in training split.
      num_validation_examples: number of examples in validation split.
      num_test_examples: number of examples in test split.
    """
    self._image_shape = image_shape
    self._split_seed = {
        tfds.Split.TRAIN: 0,
        tfds.Split.VALIDATION: 1,
        tfds.Split.TEST: 2,
    }
    super().__init__(
        name='dummy_dataset',
        dataset_builder=DummyDatasetBuilder(
            image_shape=image_shape,
            num_train_examples=num_train_examples,
            num_validation_examples=num_validation_examples,
            num_test_examples=num_test_examples),
        split=split)

  def _create_process_example_fn(self) -> base.PreProcessFn:

    def _example_parser(
        range_val: Dict[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
      """Parses a single range integer into stateless image Tensors."""
      seed = [
          self._split_seed[self._split],
          self._split_seed[self._split] + range_val['features'],
      ]
      features = tf.random.stateless_normal(
          self._image_shape, seed=seed, dtype=tf.float32)
      label = tf.zeros([], tf.int32)
      return {'features': features, 'labels': label}

    return _example_parser


if __name__ == '__main__':
  tf.test.main()

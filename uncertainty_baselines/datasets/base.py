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
"""Abstract base classes which defines interfaces for datasets."""

from typing import Callable, Optional, Sequence, Type, TypeVar, Union
from robustness_metrics.common import ops
from robustness_metrics.common import types
from robustness_metrics.datasets import tfds as robustness_metrics_base
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds
from functools import partial


# For datasets like UCI, the tf.data.Dataset returned by _read_examples will
# have elements that are Sequence[tf.Tensor], for TFDS datasets they will be
# Dict[Text, tf.Tensor] (types.Features), for Criteo they are a tf.Tensor.
PreProcessFn = Callable[
    [Union[int, tf.Tensor, Sequence[tf.Tensor], types.Features]],
    types.Features]


def _absolute_split_len(absolute_split, dataset_splits):
  if absolute_split.from_ is None:
    start = 0
  else:
    start = absolute_split.from_
  if absolute_split.to is None:
    end = dataset_splits[absolute_split.splitname].num_examples
  else:
    end = absolute_split.to
  return end - start


def get_validation_percent_split(
    dataset_builder,
    validation_percent,
    split,
    test_split=tfds.Split.TEST):
  """Calculate a validation set from a provided validation_percent in [0, 1]."""
  if validation_percent < 0.0 or validation_percent >= 1.0:
    raise ValueError(
        'validation_percent must be in [0, 1), received {}.'.format(
            validation_percent))
  num_train_examples = dataset_builder.info.splits['train'].num_examples
  num_validation_examples = int(num_train_examples * validation_percent)
  if num_validation_examples == 0:
    train_split = tfds.Split.TRAIN
    # We cannot use None here because that will return all the splits if passed
    # to builder.as_dataset().
    validation_split = tfds.Split.VALIDATION
  else:
    train_split = tfds.core.ReadInstruction(
        'train', to=-num_validation_examples, unit='abs')
    validation_split = tfds.core.ReadInstruction(
        'train', from_=-num_validation_examples, unit='abs')

  if split in ['train', tfds.Split.TRAIN]:
    new_split = train_split
  elif split in ['validation', tfds.Split.VALIDATION]:
    new_split = validation_split
  elif split in ['test', tfds.Split.TEST]:
    new_split = test_split
  elif isinstance(split, str):
    # For Python 3 this should be save to check for the Text type, see
    # https://github.com/google/pytype/blob/a4d56ef763eb4a29b5db03a2013c3373f9f46146/pytype/pytd/builtins/2and3/typing.pytd#L31.
    raise ValueError(
        'Invalid string name for split, must be one of ["train", "validation"'
        ', "test"], received {}.'.format(split))
  else:
    new_split = split
  return new_split


class BaseDataset(robustness_metrics_base.TFDSDataset):
  """Abstract base dataset class.

  Requires subclasses to override _read_examples, _create_process_example_fn.
  """

  def __init__(
      self,
      name: str,
      dataset_builder: tfds.core.DatasetBuilder,
      split: Union[float, str, tfds.Split],
      shuffle_buffer_size: int = None,
      num_parallel_parser_calls: int = 64,
      drop_remainder: bool = True,
      fingerprint_key: Optional[str] = None,
      download_data: bool = False):
    """Create a tf.data.Dataset builder.

    Args:
      name: the name of this dataset.
      dataset_builder: the TFDS dataset builder used to read examples given a
        split.
      split: a dataset split, either a custom tfds.Split or one of the
        tfds.Split enums [TRAIN, VALIDAITON, TEST] or their lowercase string
        names. For Criteo it can also be a float to represent the level of data
        augmentation. For speech commands it can be a tuple of a string and
        float for shifted data splits.
      shuffle_buffer_size: the number of example to use in the shuffle buffer
        for tf.data.Dataset.shuffle().
      num_parallel_parser_calls: the number of parallel threads to use while
        preprocessing in tf.data.Dataset.map().
      drop_remainder: whether or not to drop the last batch of data if the
        number of points is not exactly equal to the batch size. This option
        needs to be True for running on TPUs.
      fingerprint_key: The name of the feature holding a string that will be
        used to create an element id using a fingerprinting function. If None,
        then `ds.enumerate()` is added before the `ds.map(preprocessing_fn)` is
        called and an `id` field is added to the example Dict.
      download_data: Whether or not to download data before loading.
    """
    self.name = name
    self._split = split
    self._num_parallel_parser_calls = num_parallel_parser_calls
    self._drop_remainder = drop_remainder
    self._download_data = download_data

    self._is_training = split in ['train', tfds.Split.TRAIN]
    num_train_examples = dataset_builder.info.splits['train'].num_examples
    # TODO(znado): properly parse the number of train/validation/test examples
    # from the provided split, see `make_file_instructions(...)` in
    # tensorflow_datasets/core/tfrecords_reader.py.
    if shuffle_buffer_size is None:
      self._shuffle_buffer_size = num_train_examples
    else:
      self._shuffle_buffer_size = shuffle_buffer_size
    super(BaseDataset, self).__init__(
        dataset_builder=dataset_builder,
        fingerprint_key=fingerprint_key,
        split=self._split,
        label_key='label')
    self._add_enumerate = False
    if self._fingerprint_key is None:
      self._fingerprint_key = '_enumerate_added_id'
      self._add_enumerate = True

  # This method can be overridden to add custom info via info.metadata.
  @property
  def tfds_info(self) -> tfds.core.DatasetInfo:
    return self._dataset_builder.info

  @property
  def split(self):
    return self._split

  @property
  def num_examples(self):
    if isinstance(self._split, tfds.core.ReadInstruction):
      absolute_split = self._split.to_absolute(
          {
              name: self.tfds_info.splits[name].num_examples
              for name in self.tfds_info.splits.keys()
          })[0]
      return _absolute_split_len(absolute_split, self.tfds_info.splits)
    return self.tfds_info.splits[self._split].num_examples

  def _create_process_example_fn(self) -> Optional[PreProcessFn]:
    """Create a function to perform dataset pre-processing, if needed.

    Returns:
      None if no processing is necessary. Otherwise, a function which takes as
      inputs a single element of the dataset (passed from dataset.map()), and
      returns a dict with keys 'features' and 'labels' and their corresponding
      Tensor values.
    """
    raise NotImplementedError(
        'Must override dataset _create_process_example_fn!')

  def _create_element_id(self, features: types.Features) -> types.Features:
    """Hash the element id to compute a unique id."""
    if 'element_id' in features:
      raise ValueError(
          '`element_id` should not be already present in the feature set.')
    fingerprint_feature = features[self._fingerprint_key]
    features['element_id'] = ops.fingerprint_int64(fingerprint_feature)
    return features

  def _create_enumerate_preprocess_fn(
      self,
      preprocess_fn: PreProcessFn):

    def enumerated_preprocess_fn(example_id: int, x) -> types.Features:
      features = preprocess_fn(x)
      features[self._fingerprint_key] = example_id
      return features
    return enumerated_preprocess_fn

  # TODO(znado): rename to as_dataset.
  def load(self,
           *,
           preprocess_fn: PreProcessFn = None,
           batch_size: int = -1) -> tf.data.Dataset:
    """Transforms the dataset from builder.as_dataset() to batch, repeat, etc.

    Note that we do not handle replication/sharding here, because the
    DistributionStrategy experimental_distribute_dataset() will shard the input
    files for us.

    Args:
      preprocess_fn: an optional preprocessing function, if not provided then a
        subclass must define _create_process_example_fn() which will be used to
        preprocess the data.
      batch_size: the batch size to use.

    Returns:
      A tf.data.Dataset of elements that are a dict with keys 'features' and
      'labels' and their corresponding Tensor values.
    """
    if batch_size <= 0:
      raise ValueError(
          'Must provide a positive batch size, received {}.'.format(batch_size))

    if self._download_data:
      self._dataset_builder.download_and_prepare(
          download_dir=self._dataset_builder.data_dir)
    dataset = self._dataset_builder.as_dataset(self._split)

    # Map the parser over the dataset.
    if preprocess_fn is None:
      preprocess_fn = self._create_process_example_fn()
    if self._add_enumerate:
      # If necessary, enumerate the dataset to generate a unique per-example id,
      # that is then added to the feature dict in
      # `self._create_enumerate_preprocess_fn` with key `self._fingerprint_key`.
      dataset = dataset.enumerate()
      enum_preprocess_fn = self._create_enumerate_preprocess_fn(preprocess_fn)

      # Compose function will not work with >1 arguments
      preprocess_fn = lambda id, x: self._create_element_id(enum_preprocess_fn(id, x))
    else:
      preprocess_fn = ops.compose(preprocess_fn, self._create_element_id)

    dataset = dataset.map(
        preprocess_fn,
        num_parallel_calls=self._num_parallel_parser_calls)

    # Shuffle and repeat only for the training split.
    if self._is_training:
      dataset = dataset.shuffle(self._shuffle_buffer_size)
      dataset = dataset.repeat()

    # Note that unless the default value of `drop_remainder=True` is overriden
    # in `__init__`, we always drop the last batch when the batch size does not
    # evenly divide the number of examples.
    # TODO(znado): add padding to last partial eval batch.
    dataset = dataset.batch(batch_size, drop_remainder=self._drop_remainder)

    dataset = dataset.prefetch(-1)

    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = (
        tf.data.experimental.AutoShardPolicy.OFF)
    dataset = dataset.with_options(options)
    return dataset


_BaseDatasetClass = Type[TypeVar('B', bound=BaseDataset)]


def make_ood_dataset(ood_dataset_cls: _BaseDatasetClass) -> _BaseDatasetClass:
  """Generate a BaseDataset with in/out distribution labels."""

  class _OodBaseDataset(ood_dataset_cls):
    """Combine two datasets to form one with in/out of distribution labels."""

    def __init__(
        self,
        in_distribution_dataset: BaseDataset,
        shuffle_datasets: bool = False,
        **kwargs):
      super(_OodBaseDataset, self).__init__(**kwargs)
      # This should be the builder for whatever split will be considered
      # in-distribution (usually the test split).
      self._in_distribution_dataset = in_distribution_dataset
      self._shuffle_datasets = shuffle_datasets

    def load(self,
             *,
             preprocess_fn=None,
             batch_size: int = -1) -> tf.data.Dataset:
      # Set up the in-distribution dataset using the provided dataset builder.
      if preprocess_fn:
        dataset_preprocess_fn = preprocess_fn
      else:
        dataset_preprocess_fn = (
            self._in_distribution_dataset._create_process_example_fn())  # pylint: disable=protected-access
      dataset_preprocess_fn = ops.compose(
          dataset_preprocess_fn,
          _create_ood_label_fn(True))
      dataset = self._in_distribution_dataset.load(
          preprocess_fn=dataset_preprocess_fn,
          batch_size=batch_size)
      dataset = dataset.map(
          _remove_fingerprint_id_key(self._in_distribution_dataset))

      # Set up the OOD dataset using this class.
      if preprocess_fn:
        ood_dataset_preprocess_fn = preprocess_fn
      else:
        ood_dataset_preprocess_fn = (
            super(_OodBaseDataset, self)._create_process_example_fn())
      ood_dataset_preprocess_fn = ops.compose(
          ood_dataset_preprocess_fn,
          _create_ood_label_fn(False))
      ood_dataset = super(_OodBaseDataset, self).load(
          preprocess_fn=ood_dataset_preprocess_fn,
          batch_size=batch_size)
      ood_dataset = ood_dataset.map(_remove_fingerprint_id_key(self))

      # Combine the two datasets.
      combined_dataset = dataset.concatenate(ood_dataset)
      if self._shuffle_datasets:
        combined_dataset = combined_dataset.shuffle(self._shuffle_buffer_size)
      return combined_dataset

    @property
    def num_examples(self):
      return (
          self._in_distribution_dataset.num_examples +
          super(_OodBaseDataset, self).num_examples)

  return _OodBaseDataset


# We may be able to refactor this to be part of preprocess_fn so we don't
# need to do a second `ds.map()`, but the ordering of composed functions is
# tricky.
def _remove_fingerprint_id_key(dataset):

  def f(example: types.Features) -> types.Features:
    del example[dataset._fingerprint_key]  # pylint: disable=protected-access
    return example

  return f


def _create_ood_label_fn(is_in_distribution: bool) -> PreProcessFn:
  """Returns a function that adds an `is_in_distribution` key to examles."""

  def _add_ood_label(example: types.Features) -> types.Features:
    if is_in_distribution:
      in_dist_label = tf.ones_like(example['labels'], tf.int32)
    else:
      in_dist_label = tf.zeros_like(example['labels'], tf.int32)
    example['is_in_distribution'] = in_dist_label
    return example

  return _add_ood_label

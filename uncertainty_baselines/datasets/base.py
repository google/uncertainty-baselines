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

import abc
import enum
from typing import Callable, Dict, Optional, Sequence, Union
from absl import logging
import six
import tensorflow.compat.v2 as tf


class Split(enum.Enum):
  TRAIN = 'train'
  VAL = 'validation'
  TEST = 'test'


class OodSplit(enum.Enum):
  IN = 'in'
  OOD = 'ood'

# For datasets like UCI, the tf.data.Dataset returned by _read_examples will
# have elements that are Sequence[tf.Tensor], but for TFDS datasets they will be
# Dict[str, tf.Tensor].
PreProcessFn = Callable[
  [Union[int, Sequence[tf.Tensor], Dict[str, tf.Tensor]]],
  Dict[str, tf.Tensor]]


@six.add_metaclass(abc.ABCMeta)
class BaseDataset(object):
  """Abstract base dataset class.

  Requires subclasses to override _read_examples, _create_process_example_fn.
  """

  def __init__(self,
      name: str,
      batch_size: int,
      eval_batch_size: int,
      num_train_examples: int,
      num_validation_examples: int,
      num_test_examples: int,
      shuffle_buffer_size: int = None,
      num_parallel_parser_calls: int = 64,
      data_dir: Optional[str] = None):
    """Create a tf.data.Dataset builder.

    Args:
      name: the name of this dataset.
      batch_size: the training batch size.
      eval_batch_size: the validation and test batch size.
      num_train_examples: the number of training examples in this dataset.
      num_validation_examples: the number of validation examples in this
        dataset.
      num_test_examples: the number of test examples in this dataset.
      shuffle_buffer_size: the number of example to use in the shuffle buffer
        for tf.data.Dataset.shuffle().
      num_parallel_parser_calls: the number of parallel threads to use while
        preprocessing in tf.data.Dataset.map().
      data_dir: optional dir to save TFDS data to. If none then the local
        filesystem is used. Required for using TPUs on Cloud.
    """
    self.name = name

    self.batch_size = batch_size
    self.eval_batch_size = eval_batch_size
    self.info = {
        'num_train_examples': num_train_examples,
        'num_validation_examples': num_validation_examples,
        'num_test_examples': num_test_examples,
    }

    if shuffle_buffer_size is None:
      self._shuffle_buffer_size = num_train_examples
    else:
      self._shuffle_buffer_size = shuffle_buffer_size

    self._num_parallel_parser_calls = num_parallel_parser_calls
    self._data_dir = data_dir
    self._tfds_kwargs = {}
    if self._data_dir:
      self._tfds_kwargs['data_dir'] = self._data_dir
      if self._data_dir.startswith('gs://'):
        self._tfds_kwargs['try_gcs'] = True

  def _is_training(self, split: Split) -> bool:
    return split == Split.TRAIN

  def _is_in_distribution(self, split: OodSplit) -> bool:
    return split == OodSplit.IN

  @property
  def _num_train_examples(self) -> int:
    return self.info['num_train_examples']

  @property
  def _num_validation_examples(self) -> int:
    return self.info['num_validation_examples']

  @property
  def _num_test_examples(self) -> int:
    return self.info['num_test_examples']

  def _read_examples(self, split: Split) -> tf.data.Dataset:
    """Return a tf.data.Dataset to be used by _create_process_example_fn."""
    raise NotImplementedError('Must override dataset _read_examples!')

  def _create_process_example_fn(self, split: Split) -> Optional[PreProcessFn]:
    """Create a function to perform dataset pre-processing, if needed.

    Args:
      split: A dataset split.

    Returns:
      None if no processing is necessary. Otherwise, a function which takes as
      inputs a single element of the dataset (passed from dataset.map()), and
      returns a dict with keys 'features' and 'labels' and their corresponding
      Tensor values.
    """
    raise NotImplementedError(
        'Must override dataset _create_process_example_fn!')

  def _batch(self, split: Split, dataset: tf.data.Dataset) -> tf.data.Dataset:
    """Get the batched version of `dataset`."""
    # `uneven_datasets` is a list of datasets with a number of validation and/or
    # test examples that is not evenly divisible by commonly used batch sizes.
    uneven_datasets = ['criteo', 'svhn']
    if self._is_training(split):
      batch_size = self.batch_size
    elif split == Split.VAL:
      batch_size = self.eval_batch_size
      if (self._num_validation_examples % batch_size != 0 and
          self.name not in uneven_datasets):
        logging.warn(
            'Batch size does not evenly divide the number of validation '
            'examples , cannot ensure static shapes on TPU. Batch size: %d, '
            'validation examples: %d',
            batch_size,
            self._num_validation_examples)
    else:
      batch_size = self.eval_batch_size
      if (self._num_test_examples % batch_size != 0 and
          self.name not in uneven_datasets):
        logging.warn(
            'Batch size does not evenly divide the number of test examples, '
            'cannot ensure static shapes on TPU. Batch size: %d, test '
            'examples: %d', batch_size, self._num_test_examples)
    # Note that we always drop the last batch when the batch size does not
    # evenly divide the number of examples.
    return dataset.batch(batch_size, drop_remainder=True)

  def build(
      self,
      split: Union[str, Split],
      as_tuple: bool = False,
      ood_split: Optional[Union[str, OodSplit]] = None) -> tf.data.Dataset:
    """Transforms the dataset from self._read_examples() to batch, repeat, etc.

    Note that we do not handle replication/sharding here, because the
    DistributionStrategy experimental_distribute_dataset() will shard the input
    files for us.

    Args:
      split: a dataset split, either one of the Split enum or their associated
        strings.
      as_tuple: whether or not to return a Dataset where each element is a Dict
        with at least the keys ['features', 'labels'], or a tuple of
        (feature, label). If there are keys besides 'features' and 'labels' in
        the Dict then this ignore them.
      ood_split: an optional OOD split, either one of the OodSplit enum or
        their associated strings.

    Returns:
      A tf.data.Dataset of elements that are a dict with keys 'features' and
      'labels' and their corresponding Tensor values.
    """
    if isinstance(split, str):
      split = Split(split)

    if isinstance(ood_split, str):
      ood_split = OodSplit(ood_split)

    dataset = self._read_examples(split)

    # Map the parser over the dataset.
    if ood_split:
      process_example_fn = self._create_ood_process_example_fn(split, ood_split)
      if split == Split.TRAIN:
        self.info['num_ood_examples'] = self._num_train_examples
      if split == Split.VAL:
        self.info['num_ood_examples'] = self._num_validation_examples
      if split == Split.TEST:
        self.info['num_ood_examples'] = self._num_test_examples
    else:
      process_example_fn = self._create_process_example_fn(split)
    if process_example_fn:
      dataset = dataset.map(
          process_example_fn,
          num_parallel_calls=self._num_parallel_parser_calls)
    # pylint: disable=g-long-lambda
    if as_tuple and ood_split:
      dataset = dataset.map(lambda d: (d['features'], d['labels'],
                                       d['is_in_distribution']))
    # pylint: enable=line-too-long
    elif as_tuple and not ood_split:
      dataset = dataset.map(lambda d: (d['features'], d['labels']))

    # Shuffle and repeat only for the training split.
    if self._is_training(split):
      dataset = dataset.shuffle(self._shuffle_buffer_size)
      dataset = dataset.repeat()

    dataset = self._batch(split, dataset)

    dataset = dataset.prefetch(-1)

    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = (
        tf.data.experimental.AutoShardPolicy.OFF)
    dataset = dataset.with_options(options)

    return dataset

  def _create_ood_process_example_fn(self, split: Split,
                                     ood_split: OodSplit) -> PreProcessFn:
    """Add additional labels to a pre-existing dataset for OOD.

    Args:
      split: A dataset split.
      ood_split: An OOD dataset split.

    Returns:
      A function which takes as inputs a single element of the dataset (passed
      from dataset.map()), and returns a dict with keys 'features', 'labels',
      and an additional key 'is_in_distribution', with 0s for and 'ood' split
      and 1s for 'in' split.
      and their corresponding Tensor values.
    """
    process_example_fn = self._create_process_example_fn(split)

    def _add_ood_label_parser(
        example: Dict[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
      processed = process_example_fn(example)
      if self._is_in_distribution(ood_split):
        in_dist_label = tf.ones_like(processed['labels'], tf.int32)
      else:
        in_dist_label = tf.zeros_like(processed['labels'], tf.int32)
      processed['is_in_distribution'] = in_dist_label
      return processed

    return _add_ood_label_parser

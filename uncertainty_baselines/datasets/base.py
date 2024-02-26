# coding=utf-8
# Copyright 2024 The Uncertainty Baselines Authors.
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

"""Abstract base classes which defines interfaces for datasets."""

import logging
from typing import Callable, Dict, Optional, Sequence, Type, TypeVar, Union

from robustness_metrics.common import ops
from robustness_metrics.common import types
from robustness_metrics.datasets import tfds as robustness_metrics_base
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds


# For datasets like UCI, the tf.data.Dataset returned by _read_examples will
# have elements that are Sequence[tf.Tensor], for TFDS datasets they will be
# Dict[Text, tf.Tensor] (types.Features), for Criteo they are a tf.Tensor.
PreProcessFn = Callable[
    [Union[int, tf.Tensor, Sequence[tf.Tensor], types.Features]],
    types.Features]
FilterFn = Callable[[types.Features], bool]

# Same as PreProcessFn except also takes an integer first argument.
_EnumeratedPreProcessFn = Callable[
    [int, Union[int, tf.Tensor, Sequence[tf.Tensor], types.Features]],
    types.Features]


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
  if validation_percent == 0.:
    train_split = tfds.Split.TRAIN
    # We cannot use None here because that will return all the splits if passed
    # to builder.as_dataset().
    validation_split = tfds.Split.VALIDATION
  else:
    num_train_examples = dataset_builder.info.splits['train'].num_examples
    num_validation_examples = int(num_train_examples * validation_percent)
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


_drd_datasets = [
    'ub_diabetic_retinopathy_detection',
    'diabetic_retinopathy_severity_shift_mild',
    'diabetic_retinopathy_severity_shift_moderate', 'aptos/btgraham-300',
    'aptos/blur-3-btgraham-300', 'aptos/blur-5-btgraham-300',
    'aptos/blur-10-btgraham-300', 'aptos/blur-20-btgraham-300'
]


class BaseDataset(robustness_metrics_base.TFDSDataset):
  """Abstract base dataset class.

  Requires subclasses to override _read_examples, _create_process_example_fn.
  """

  def __init__(self,
               name: str,
               dataset_builder: tfds.core.DatasetBuilder,
               split: Union[float, str, tfds.Split],
               seed: Optional[Union[int, tf.Tensor]] = None,
               is_training: Optional[bool] = None,
               shuffle_buffer_size: Optional[int] = None,
               num_parallel_parser_calls: int = tf.data.experimental.AUTOTUNE,
               drop_remainder: bool = False,
               mask_and_pad: bool = False,
               fingerprint_key: Optional[str] = None,
               download_data: bool = False,
               decoders: Optional[Dict[str, tfds.decode.Decoder]] = None,
               cache: bool = False,
               label_key: str = 'label',
               filter_fn: Optional[FilterFn] = None):
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
      seed: the seed used as a source of randomness.
      is_training: whether or not `split` is the training split. This is
        necessary because tf.data subsplits can sometimes be derived from the
        training split, such as when using part of the training split as a
        validation set, and this complicates calculating `is_training`
        in these cases. Only required when the passed split is not one of
        ['train', 'validation', 'test', tfds.Split.TRAIN, tfds.Split.VALIDATION,
        tfds.Split.TEST], otherwise it is set automatically.
      shuffle_buffer_size: the number of example to use in the shuffle buffer
        for tf.data.Dataset.shuffle().
      num_parallel_parser_calls: the number of parallel threads to use while
        preprocessing in tf.data.Dataset.map().
      drop_remainder: Whether or not to drop the last batch of data if the
        number of points is not exactly equal to the batch size.
      mask_and_pad: Whether or not to mask and pad batches such that when
        drop_remainder == False, partial batches are padded to a full batch and
        an additional `mask` feature is added to indicate which examples are
        padding.
      fingerprint_key: The name of the feature holding a string that will be
        used to create an element id using a fingerprinting function. If None,
        then `ds.enumerate()` is added before the `ds.map(preprocessing_fn)` is
        called and an `id` field is added to the example Dict.
      download_data: Whether or not to download data before loading.
      decoders: Optional TFDS decoders to provide to
        `dataset_builder.as_dataset`, the same as passed to `tfds.load`.
      cache: Whether or not to cache the dataset after it is returned from
        dataset_builder.as_dataset(...) (before preprocessing is applied).
      label_key: The name of the field holding the label.
      filter_fn: The filter function for tf.data.Dataset.filter().
    """
    self.name = name
    self._split = split

    # Stateless random ops require a (2,) shaped seed.
    if seed is None:
      self._seed = tf.random.uniform((2,), maxval=int(1e10), dtype=tf.int32)
    elif isinstance(seed, int):
      self._seed = (seed, seed + 1)
    elif isinstance(seed, tf.Tensor) and tf.shape(seed).shape == 0:
      self._seed = tf.stack([seed, seed+1])
    else:
      self._seed = seed

    self._num_parallel_parser_calls = num_parallel_parser_calls
    self._drop_remainder = drop_remainder
    self._mask_and_pad = mask_and_pad
    self._download_data = download_data
    self._decoders = decoders
    self._cache = cache

    known_splits = [
        'train', 'validation', 'test', tfds.Split.TRAIN, tfds.Split.VALIDATION,
        tfds.Split.TEST
    ]
    if is_training is None:
      if split in known_splits:
        is_training = split in ['train', tfds.Split.TRAIN]
      else:
        raise ValueError(
            'Received ambiguous split {}, must set is_training for splits other'
            ' than "train", "validation", "test".'.format(split)
        )

    self._is_training = is_training
    # TODO(znado): properly parse the number of train/validation/test examples
    # from the provided split, see `make_file_instructions(...)` in
    # tensorflow_datasets/core/tfrecords_reader.py.
    if 'train' in dataset_builder.info.splits and shuffle_buffer_size is None:
      num_train_examples = dataset_builder.info.splits['train'].num_examples
      self._shuffle_buffer_size = num_train_examples
    else:
      self._shuffle_buffer_size = shuffle_buffer_size
    super(BaseDataset, self).__init__(
        dataset_builder=dataset_builder,
        fingerprint_key=fingerprint_key,
        split=self._split,
        label_key=label_key)

    self._enumerate_id_key = '_enumerate_added_per_step_id'

    self._add_fingerprint_key = False
    if self._fingerprint_key is None:
      self._fingerprint_key = '_enumerate_added_example_id'
      self._add_fingerprint_key = True

    self._filter_fn = filter_fn

  # This method can be overridden to add custom info via info.metadata.
  @property
  def tfds_info(self) -> tfds.core.DatasetInfo:
    return self._dataset_builder.info

  @property
  def split(self):
    return self._split

  @property
  def num_examples(self):
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

  def _create_process_batch_fn(self, batch_size: int) -> Optional[PreProcessFn]:
    """Create a function to perform pre-processing on batches, such as mixup.

    Args:
      batch_size: the size of the batch to be processed.

    Returns:
      None if no processing is necessary. Otherwise, a function which takes as
      inputs a batch of elements of the dataset (passed from dataset.batch()),
      and returns a dict with keys 'features' and 'labels' and their
      corresponding Tensor values.
    """
    return None

  def _create_element_id(self, features: types.Features) -> types.Features:
    """Hash element id for a unique id per data element (NOT per-step)."""
    if 'element_id' in features:
      raise ValueError(
          '`element_id` should not be already present in the feature set.')
    fingerprint_feature = features[self._fingerprint_key]
    features['element_id'] = ops.fingerprint_int64(fingerprint_feature)
    return features

  def _add_enumerate_id(self, enumerate_key: str) -> _EnumeratedPreProcessFn:
    def _add_example_id(enumerate_id, example):
      """Turn an id added by ds.enumerate() as a field in the example dict."""
      if isinstance(example, (tf.Tensor, tuple)):
        example = {'features': example}
      example[enumerate_key] = enumerate_id
      return example
    return _add_example_id

  def _load(self,
            *,
            preprocess_fn: Optional[PreProcessFn] = None,
            process_batch_fn: Optional[PreProcessFn] = None,
            batch_size: int = -1) -> tf.data.Dataset:
    """Transforms the dataset from builder.as_dataset() to batch, repeat, etc.

    Note that we do not handle replication/sharding here, because the
    DistributionStrategy experimental_distribute_dataset() will shard the input
    files for us.

    Args:
      preprocess_fn: an optional preprocessing function, if not provided then a
        subclass must define _create_process_example_fn() which will be used to
        preprocess the data.
      process_batch_fn: an optional processing batch function. If not
        provided then _create_process_batch_fn() will be used to generate the
        function that will process a batch of data.
      batch_size: the batch size to use.

    Returns:
      A tf.data.Dataset of elements that are a dict with keys 'features' and
      'labels' and their corresponding Tensor values.
    """
    if batch_size <= 0:
      raise ValueError(
          'Must provide a positive batch size, received {}.'.format(batch_size))

    self._seed, self._shuffle_seed = tf.random.experimental.stateless_split(
        self._seed, num=2)

    if self._download_data:
      self._dataset_builder.download_and_prepare()
    dataset = self._dataset_builder.as_dataset(
        split=self._split, decoders=self._decoders)

    # Possibly cache the original dataset before preprocessing is applied.
    if self._cache:
      dataset = dataset.cache()

    # This must be done *before* repeating the dataset so that each example has
    # a unique and stable fingerprint key.
    if self._add_fingerprint_key:
      dataset = dataset.enumerate()
      add_fingerprint_key_fn = self._add_enumerate_id(self._fingerprint_key)
      dataset = dataset.map(
          add_fingerprint_key_fn,
          num_parallel_calls=self._num_parallel_parser_calls)

    # If we are truncating the validation/test dataset (self._drop_remainder)
    # we may as well repeat to speed things up.
    # TODO(nband): benchmark.
    # TODO(trandustin): Make this differing behavior consistent with other
    # ub.datasets.
    if (self.name in _drd_datasets and not self._is_training and
        self._drop_remainder):
      dataset = dataset.repeat()
      logging.info('Repeating dataset %s (training mode: %s).', self.name,
                   self._is_training)

    # Shuffle and repeat only for the training split.
    if self._is_training:
      dataset = dataset.shuffle(
          self._shuffle_buffer_size,
          seed=tf.cast(self._shuffle_seed[0], tf.int64),
          reshuffle_each_iteration=True)
      dataset = dataset.repeat()

    # Enumerate the dataset to generate a unique per-example, per-step id, that
    # is then added to the feature dict as `self._enumerate_id_key`.
    # Note that this is distinct from just a per-example id that is used by
    # Robustness Metrics to identify examples, because we want an id that is
    # different for each step so that we can fold it into a source of randomness
    # for deterministic random preprocessing.
    # This must be done *after* repeating the dataset so that each example has a
    # different key per-step.
    dataset = dataset.enumerate()
    add_per_step_id_key_fn = self._add_enumerate_id(self._enumerate_id_key)

    if preprocess_fn is None:
      preprocess_fn = self._create_process_example_fn()

    # `self._create_element_id` must come before `preprocess_fn` so that we
    # guarantee the field with key `self._fingerprint_key` is still present
    # (many preprocess_fn's may not return it).
    preprocess_fn = ops.compose(
        add_per_step_id_key_fn, self._create_element_id, preprocess_fn)
    if self._mask_and_pad:
      mask_fn = lambda ex: dict(mask=1., **ex)
      preprocess_fn = ops.compose(preprocess_fn, mask_fn)
    dataset = dataset.map(
        preprocess_fn,
        num_parallel_calls=self._num_parallel_parser_calls)

    if self._filter_fn:
      dataset = dataset.filter(self._filter_fn)

    if self._mask_and_pad and not self._drop_remainder:
      # If we're not dropping the remainder, but we are adding masking +
      # padding, then we append additional zero-valued examples with zero-valued
      # masks to the dataset such that batching with drop_remainder=True will
      # yield a dataset whose final batch is padded as needed.
      padding_example = tf.nest.map_structure(
          lambda spec: tf.zeros(spec.shape, spec.dtype)[None],
          dataset.element_spec)
      padding_example['mask'] = [0.]
      padding_dataset = tf.data.Dataset.from_tensor_slices(padding_example)
      dataset = dataset.concatenate(padding_dataset.repeat(batch_size - 1))
      dataset = dataset.batch(batch_size, drop_remainder=True)
    else:
      dataset = dataset.batch(batch_size, drop_remainder=self._drop_remainder)

    if process_batch_fn is None:
      process_batch_fn = self._create_process_batch_fn(batch_size)  # pylint: disable=assignment-from-none
    if process_batch_fn:
      dataset = dataset.map(
          process_batch_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    if not self._is_training and self.name not in _drd_datasets:
      dataset = dataset.cache()
    else:
      if not self._cache:
        logging.info(
            'Not caching dataset %s (training mode: %s).',
            self.name,
            self._is_training)

    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    # The AutoSharding policy in DistributionStrategy defaults to AUTO, which
    # will fallback to DATA if it can, which is safe to do but possibly
    # wasteful compared to `distribute_datasets_from_function`.
    options = tf.data.Options()
    # Optimize dataset performance.
    # Keep commented out, unclear if will always improve performance.
    # options.experimental_optimization.parallel_batch = True
    options.experimental_optimization.map_fusion = True
    options.experimental_optimization.map_parallelization = True
    options.experimental_threading.private_threadpool_size = 48
    options.experimental_threading.max_intra_op_parallelism = 1
    dataset = dataset.with_options(options)
    return dataset

  def load(  # pytype: disable=signature-mismatch  # overriding-parameter-count-checks
      self,
      *,
      preprocess_fn: Optional[PreProcessFn] = None,
      process_batch_fn: Optional[PreProcessFn] = None,
      batch_size: int = -1,
      strategy: Optional[tf.distribute.Strategy] = None) -> tf.data.Dataset:
    """Function definition to support multi-host dataset sharding.

    This is preferred over strategy.experimental_distribute_dataset(...) because
    not all datasets will have enough input files to have >=1 per host, which
    will result in an error unless the auto_shard_policy is set to
    tf.data.experimental.AutoShardPolicy.OFF. However, if auto sharding is OFF,
    then each host will process the same set of files, in the same order, which
    will be the same as using a single host. To correctly distribute across
    multiple hosts, we must either shard the input files across hosts, or
    shuffle the data in a different order on each host. In order to get a
    per-host id to fold into the shuffle RNG, we use
    strategy.distribute_datasets_from_function to get a
    tf.distribute.InputContext. This is preferrred over AutoShardPolicy.DATA
    because DATA will prepare and throw out (n - 1)/n elements of data, where n
    is the number of devices (DATA also relies on the data files being in the
    same ordering across hosts, which may be a fair assumption). On an ImageNet
    test run on a TPUv2-32, we saw a 37% slowdown using DATA instead of
    `distribute_datasets_from_function`. See this documentation for more info
    https://www.tensorflow.org/tutorials/distribute/input#sharding.


    Args:
      preprocess_fn: see `load()`.
      process_batch_fn: see `load()`.
      batch_size: the *global* batch size to use. This should equal
        `per_replica_batch_size * num_replica_in_sync`.
      strategy: the DistributionStrategy used to shard the dataset. Note that
        this is only required if TensorFlow for training, otherwise it can be
        ignored.

    Returns:
      A sharded dataset, with its seed combined with the per-host id.
    """
    if strategy:
      def _load_distributed(ctx: tf.distribute.InputContext):
        self._seed = tf.random.experimental.stateless_fold_in(
            self._seed, ctx.input_pipeline_id)
        per_replica_batch_size = ctx.get_per_replica_batch_size(batch_size)
        return self._load(
            preprocess_fn=preprocess_fn,
            process_batch_fn=process_batch_fn,
            batch_size=per_replica_batch_size)

      return strategy.distribute_datasets_from_function(_load_distributed)
    else:
      return self._load(preprocess_fn=preprocess_fn,
                        process_batch_fn=process_batch_fn,
                        batch_size=batch_size)


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
          preprocess_fn=ood_dataset_preprocess_fn, batch_size=batch_size)
      # We keep the fingerprint id in both dataset and ood_dataset

      # Combine the two datasets.
      try:
        combined_dataset = dataset.concatenate(ood_dataset)
      except TypeError:
        logging.info(
            'Two datasets have different types, concat feature and label only')

        def clean_keys(example):
          # only keep features and labels, remove the rest
          return {
              'features': example['features'],
              'labels': example['labels'],
              'is_in_distribution': example['is_in_distribution']
          }

        combined_dataset = dataset.map(clean_keys).concatenate(
            ood_dataset.map(clean_keys))
      if self._shuffle_datasets:
        combined_dataset = combined_dataset.shuffle(self._shuffle_buffer_size)
      return combined_dataset

    def num_examples(self, data_type='default'):
      if data_type == 'default':
        return (self._in_distribution_dataset.num_examples +
                super().num_examples)
      elif data_type == 'in_distribution':
        return self._in_distribution_dataset.num_examples
      elif data_type == 'ood':
        return super().num_examples
      else:
        raise NotImplementedError('The data_type %s is not valid.' % data_type)

  return _OodBaseDataset


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

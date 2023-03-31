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

"""Input pipeline utilities for the ViT experiments."""

import collections
import math
from typing import Callable, Dict, Optional, Union

from absl import logging
import flax
import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import al_utils  # local file import from baselines.jft
from uncertainty_baselines.datasets import tfds as ub_tfds

Tensor = Union[tf.Tensor, tf.SparseTensor, tf.RaggedTensor]
Features = Dict[str, Tensor]

# TODO(dusenberrymw): Make al_utils.SubsetDatasetBuilder subclass
# tfds.core.DatasetBuilder and remove this.
SubsetDatasetBuilder = al_utils.SubsetDatasetBuilder


def _get_dataset_builder(
    dataset: Union[str, tfds.core.DatasetBuilder,
                   SubsetDatasetBuilder],
    data_dir: Optional[str] = None
) -> Union[tfds.core.DatasetBuilder, SubsetDatasetBuilder]:
  """Returns a dataset builder."""
  if isinstance(dataset, str):
    dataset_builder = tfds.builder(dataset, data_dir=data_dir)
  elif isinstance(dataset, tfds.core.DatasetBuilder):
    dataset_builder = dataset
  elif isinstance(dataset, SubsetDatasetBuilder):
    dataset_builder = dataset
  else:
    raise ValueError(
        "`dataset` must be a string or tfds.core.DatasetBuilder or "
        f" SubsetDatasetBuilder. Received {dataset} instead.")
  return dataset_builder


def _get_process_split(split: str, process_index: int, process_count: int,
                       drop_remainder: bool) -> tfds.typing.SplitArg:
  """Returns the split for the given process given a multi-process setup."""
  splits = tfds.even_splits(
      split, n=process_count, drop_remainder=drop_remainder)
  process_split = splits[process_index]
  return process_split


def _get_process_num_examples(builder: tfds.core.DatasetBuilder, split: str,
                              process_batch_size: int, process_index: int,
                              process_count: int, drop_remainder: bool) -> int:
  """Returns the number of examples in a given process's split."""
  process_split = _get_process_split(
      split,
      process_index=process_index,
      process_count=process_count,
      drop_remainder=drop_remainder)
  num_examples = builder.info.splits[process_split].num_examples

  if drop_remainder:
    device_batch_size = process_batch_size // jax.local_device_count()
    num_examples = (
        math.floor(num_examples / device_batch_size) * device_batch_size)

  return num_examples


def get_num_examples(dataset: Union[str, tfds.core.DatasetBuilder,
                                    SubsetDatasetBuilder],
                     split: str,
                     process_batch_size: int,
                     drop_remainder: bool = True,
                     process_count: Optional[int] = None,
                     data_dir: Optional[str] = None) -> int:
  """Returns the total number of examples in a (sharded) dataset split.

  Args:
    dataset: Either a dataset name or a dataset builder object.
    split: Specifies which split of the data to load.
    process_batch_size: Per process batch size.
    drop_remainder: Whether to drop remainders when sharding across processes
      and batching.
    process_count: Number of global processes (over all "hosts") across
      which the dataset will be sharded. If None, then the number of global
      processes will be obtained from `jax.process_count()`.
    data_dir: Directory for the dataset files.

  Returns:
    The number of examples in the dataset split that will be read when sharded
    across available processes.
  """
  dataset_builder = _get_dataset_builder(dataset, data_dir)
  if process_count is None:
    process_count = jax.process_count()

  num_examples = 0
  for i in range(process_count):
    num_examples += _get_process_num_examples(
        dataset_builder,
        split=split,
        process_batch_size=process_batch_size,
        process_index=i,
        process_count=process_count,
        drop_remainder=drop_remainder)

  remainder = dataset_builder.info.splits[split].num_examples - num_examples
  if remainder:
    warning = (f"Dropping {remainder} examples from the {split} split of the "
               f"{dataset_builder.info.name} dataset.")
    logging.warning(warning)

  return num_examples


def _preprocess_with_per_example_rng(ds: tf.data.Dataset,
                                     preprocess_fn: Callable[[Features],
                                                             Features], *,
                                     rng: jnp.ndarray) -> tf.data.Dataset:
  """Maps `ds` using the preprocess_fn and a deterministic RNG per example.

  Args:
    ds: Dataset containing Python dictionary with the features. The 'rng'
      feature should not exist.
    preprocess_fn: Preprocessing function that takes a Python dictionary of
      tensors and returns a Python dictionary of tensors. The function should be
      convertible into a TF graph.
    rng: Base RNG to use. Per example RNGs will be derived from this by folding
      in the example index.

  Returns:
    The dataset mapped by the `preprocess_fn`.
  """

  def _fn(example_index: int, features: Features) -> Features:
    example_index = tf.cast(example_index, tf.int32)
    features["rng"] = tf.random.experimental.stateless_fold_in(
        tf.cast(rng, tf.int64), example_index)
    processed = preprocess_fn(features)
    if isinstance(processed, dict) and "rng" in processed:
      del processed["rng"]
    return processed

  return ds.enumerate().map(_fn, num_parallel_calls=tf.data.AUTOTUNE)


def _build_dataset(dataset: Union[str, tfds.core.DatasetBuilder,
                                  SubsetDatasetBuilder],
                   data_dir: Optional[str], split: str, shuffle_files: bool,
                   file_shuffle_seed: jnp.ndarray, process_index: int,
                   process_count: int, drop_remainder: bool) -> tf.data.Dataset:
  """Builds the dataset."""
  dataset_builder = _get_dataset_builder(dataset, data_dir)

  dataset_options = tf.data.Options()
  dataset_options.experimental_optimization.map_parallelization = True
  dataset_options.experimental_threading.private_threadpool_size = 48
  dataset_options.experimental_threading.max_intra_op_parallelism = 1

  read_config = tfds.ReadConfig(  # pytype: disable=wrong-arg-types  # jax-ndarray
      shuffle_seed=file_shuffle_seed, options=dataset_options)

  process_split = _get_process_split(
      split,
      process_index=process_index,
      process_count=process_count,
      drop_remainder=drop_remainder)

  ds = dataset_builder.as_dataset(
      split=process_split,
      shuffle_files=shuffle_files,
      read_config=read_config,
      decoders={"image": tfds.decode.SkipDecoding()})
  return ds


def get_data(
    dataset: Union[str, tfds.core.DatasetBuilder, SubsetDatasetBuilder],
    split: str,
    rng: Optional[jnp.ndarray],
    process_batch_size: int,
    preprocess_fn: Optional[Callable[[Features], Features]],
    cache: Union[str, bool] = False,
    num_epochs: Optional[int] = None,
    repeat_after_batching: bool = False,
    shuffle: bool = True,
    shuffle_buffer_size: int = 10_000,
    prefetch_size: int = 4,
    drop_remainder: bool = True,
    data_dir: Optional[str] = None,
    process_index: Optional[int] = None,
    process_count: Optional[int] = None,
) -> tf.data.Dataset:
  """Creates a standard input pipeline (shuffle, preprocess, batch).

  Args:
    dataset: Either a dataset name or a dataset builder object.
    split: Specifies which split of the data to load. Will be sharded across all
      available processes (globally over all "hosts"), and the unique sharded
      subsplit corresponding to the current process will be returned.
    rng: A jax.random.PRNG key to use for seeding shuffle operations and
      preprocessing ops. Must be set if shuffling.
    process_batch_size: Per process batch size.
    preprocess_fn: Function for preprocessing individual examples (which should
      be Python dictionary of tensors).
    cache: Whether to cache the unprocessed dataset in memory before
      preprocessing and batching ("loaded"), after preprocessing and batching
      ("batched"), or not at all (False).
    num_epochs: Number of epochs for which to repeat the dataset. None to repeat
      forever.
    repeat_after_batching: Whether to `repeat` the dataset before or after
      batching.
    shuffle: Whether to shuffle the dataset (both on file and example level).
    shuffle_buffer_size: Number of examples in the shuffle buffer.
    prefetch_size: The number of elements in the final dataset to prefetch in
      the background. This should be a small (say <10) positive integer or
      tf.data.AUTOTUNE.
    drop_remainder: Whether to drop remainders when batching and splitting
      across processes.
    data_dir: Directory for the dataset files.
    process_index: Integer id in the range [0, process_count) of the current
      process in a multi-process setup. If None, then the index will be obtained
      from `jax.process_index()`.
    process_count: Number of global processes (over all "hosts") across which
      the dataset will be sharded. If None, then the number of global processes
      will be obtained from `jax.process_count()`.

  Returns:
    The dataset with preprocessed, masked, padded, and batched examples for the
    unique sharded subset of `split` corresponding to the current process in a
    multi-process setup.
  """
  assert cache in ("loaded", "batched", False, None)

  rng_available = rng is not None
  if not rng_available and shuffle:
    raise ValueError("Please set 'rng' when shuffling.")

  if process_index is None:
    process_index = jax.process_index()

  if process_count is None:
    process_count = jax.process_count()

  if rng_available:
    rng = jax.random.fold_in(rng, process_index)  # Derive RNG for this process.
    rngs = list(jax.random.split(rng, 3))
  else:
    rngs = 3 * [[None, None]]

  ds = _build_dataset(
      dataset,
      data_dir=data_dir,
      split=split,
      shuffle_files=shuffle,
      file_shuffle_seed=rngs.pop()[0],
      process_index=process_index,
      process_count=process_count,
      drop_remainder=drop_remainder)

  if cache == "loaded":
    ds = ds.cache()

  if shuffle:
    ds = ds.shuffle(shuffle_buffer_size, seed=rngs.pop()[0])

  if not repeat_after_batching:
    ds = ds.repeat(num_epochs)

  mask_fn = lambda ex: dict(mask=1., **ex)
  if preprocess_fn is not None:
    preprocess_and_mask_fn = lambda ex: mask_fn(preprocess_fn(ex))
  else:
    preprocess_and_mask_fn = mask_fn

  if rng_available:
    ds = _preprocess_with_per_example_rng(
        ds, preprocess_and_mask_fn, rng=rngs.pop())
  else:
    ds = ds.map(preprocess_and_mask_fn, num_parallel_calls=tf.data.AUTOTUNE)

  # Batch and reshape to [num_devices, batch_size_per_device] with padding.
  num_devices = jax.local_device_count()
  batch_size_per_device = process_batch_size // num_devices

  if not drop_remainder:
    # If we're not dropping the remainder, then we append additional zero-valued
    # examples with zero-valued masks to the dataset such that batching with
    # drop_remainder=True will yield a dataset whose final batch is padded as
    # needed.
    # NOTE: We're batching the dataset over two dimensions,
    # `batch_size_per_device` and `num_devices`. Therefore, adding
    # `batch_size_per_device*num_devices - 1` padding examples covers the worst
    # case of 1 example left over after the first batching application with
    # batch size `batch_size_per_device` (since we'd need
    # `batch_size_per_device*num_devices - 1` additional examples).
    padding_example = tf.nest.map_structure(
        lambda spec: tf.zeros(spec.shape, spec.dtype)[None], ds.element_spec)
    padding_example["mask"] = [0.]
    padding_dataset = tf.data.Dataset.from_tensor_slices(padding_example)
    ds = ds.concatenate(
        padding_dataset.repeat(batch_size_per_device * num_devices - 1))

  batch_dims = [num_devices, batch_size_per_device]
  for batch_size in reversed(batch_dims):
    ds = ds.batch(batch_size, drop_remainder=True)

  if cache == "batched":
    ds = ds.cache()

  if repeat_after_batching:
    ds = ds.repeat(num_epochs)

  return ds.prefetch(prefetch_size)


def cifar_from_sql(sql_database: str,
                   num_classes: int) -> ub_tfds.TFDSBuilderFromSQLClientData:
  """Build a TFDS builder backed by CIFAR-like SQL Client Data."""

  cifar_features = tfds.features.FeaturesDict({
      "image": tfds.features.Image(shape=(32, 32, 3), dtype=tf.uint8),
      "label": tfds.features.ClassLabel(num_classes=num_classes),
  })

  cifar_element_spec = collections.OrderedDict([
      ("image", tf.TensorSpec(shape=(32, 32, 3), dtype=tf.uint8)),
      ("label", tf.TensorSpec(shape=(), dtype=tf.int64)),
  ])

  return ub_tfds.TFDSBuilderFromSQLClientData(
      sql_database=sql_database,
      tfds_features=cifar_features,
      element_spec=cifar_element_spec)


def start_input_pipeline(dataset, n_prefetch, devices=None):
  """Creates a data iterator with optional prefetching and padding."""
  it = iter(dataset)

  def _prepare(x):
    # Transforms x into read-only numpy array without copy if possible, see:
    # https://github.com/tensorflow/tensorflow/issues/33254#issuecomment-542379165
    return np.asarray(memoryview(x))

  it = (jax.tree_map(_prepare, xs) for xs in it)

  if n_prefetch:
    it = flax.jax_utils.prefetch_to_device(it, n_prefetch, devices=devices)
  return it

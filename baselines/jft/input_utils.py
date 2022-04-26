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

"""Input pipeline utilities for the ViT experiments."""

import math
from typing import Callable, Dict, Optional, Union

from absl import logging
import flax
import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

Tensor = Union[tf.Tensor, tf.SparseTensor, tf.RaggedTensor]
Features = Dict[str, Tensor]



def _get_dataset_builder(
    dataset: Union[str, tfds.core.DatasetBuilder],
    data_dir: Optional[str] = None) -> tfds.core.DatasetBuilder:
  """Returns a dataset builder."""
  if isinstance(dataset, str):
    dataset_builder = tfds.builder(dataset, data_dir=data_dir)
  elif isinstance(dataset, tfds.core.DatasetBuilder):
    dataset_builder = dataset
  else:
    raise ValueError(
        "`dataset` must be a string or tfds.core.DatasetBuilder. Received "
        f"{dataset} instead.")
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


def get_num_examples(dataset: Union[str, tfds.core.DatasetBuilder],
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


def _add_mask(batch, num_batch_dims):
  """Adds a mask to a dictionary of tensors."""
  mask = tf.ones(tf.shape(list(batch.values())[0])[:num_batch_dims])
  if "mask" in batch:
    mask *= batch["mask"]
  batch["mask"] = mask
  return batch


def _pad_reshape_batch(batch, flat_batch_size, num_devices):
  """Pads and reshapes the tensors in a flattened batch."""
  def f(x):
    actual_batch_size = tf.shape(x)[0]
    needed = flat_batch_size - actual_batch_size
    zeros = tf.zeros(tf.concat([[needed], x.shape[1:]], axis=0), dtype=x.dtype)
    new_x = tf.concat([x, zeros], axis=0)
    new_x = tf.reshape(new_x, tf.concat([[num_devices, -1], x.shape[1:]],
                                        axis=0))
    return new_x

  new_batch = {k: f(v) for k, v in batch.items()}
  return new_batch


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


def _build_dataset(dataset: Union[str, tfds.core.DatasetBuilder],
                   data_dir: Optional[str], split: str, shuffle_files: bool,
                   file_shuffle_seed: jnp.ndarray, process_index: int,
                   process_count: int, drop_remainder: bool) -> tf.data.Dataset:
  """Builds the dataset."""
  dataset_builder = _get_dataset_builder(dataset, data_dir)

  dataset_options = tf.data.Options()
  dataset_options.deterministic = True
  dataset_options.experimental_optimization.map_parallelization = True
  dataset_options.threading.private_threadpool_size = 48
  dataset_options.threading.max_intra_op_parallelism = 1

  read_config = tfds.ReadConfig(
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
    dataset: Union[str, tfds.core.DatasetBuilder],
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

  ds = ds.repeat(num_epochs if not repeat_after_batching else 1)

  if preprocess_fn is not None:
    if rng_available:
      ds = _preprocess_with_per_example_rng(ds, preprocess_fn, rng=rngs.pop())
    else:
      ds = ds.map(preprocess_fn, num_parallel_calls=tf.data.AUTOTUNE)

  # Batch and reshape to [num_devices, batch_size_per_device] with padding.
  num_devices = jax.local_device_count()
  if drop_remainder:
    # If we're dropping the remainder, we can take the fast path of double
    # batching to [num_devices, batch_size_per_device] and then adding a mask of
    # ones for the two batch dimensions.
    batch_size_per_device = process_batch_size // num_devices
    batch_dims = [num_devices, batch_size_per_device]
    for batch_size in reversed(batch_dims):
      ds = ds.batch(batch_size, drop_remainder=True)

    ds = ds.map(
        lambda xs: _add_mask(xs, 2), num_parallel_calls=tf.data.AUTOTUNE)
  else:
    # If we're not dropping the remainder, then we define a flattened batch size
    # that would divide evenly across devices, and then batch to that size with
    # drop_remainder=False. Then we add a mask of ones for the examples given,
    # pad each flattened batch with zeros (including the mask) to ensure all
    # batches have the same number of examples, and then reshape to
    # [num_devices, batch_size_per_device].
    batch_size_per_device = math.ceil(process_batch_size / num_devices)
    flat_batch_size = batch_size_per_device * num_devices
    ds = ds.batch(flat_batch_size, drop_remainder=drop_remainder)

    def f(xs):
      return _pad_reshape_batch(_add_mask(xs, 1), flat_batch_size, num_devices)

    ds = ds.map(f, num_parallel_calls=tf.data.AUTOTUNE)

  if cache == "batched":
    ds = ds.cache()

  if repeat_after_batching:
    ds = ds.repeat(num_epochs)

  return ds.prefetch(prefetch_size)


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

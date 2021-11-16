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

"""Input pipeline utilities for the ViT experiments."""
import math
from typing import Optional, Union, Callable

from absl import logging
from clu import deterministic_data
import flax
import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds


def _get_dataset_builder(
    dataset: Union[str, tfds.core.DatasetBuilder],
    data_dir: Optional[str] = None) -> tfds.core.DatasetBuilder:
  """Returns a dataset builder."""
  if isinstance(dataset, str):
    dataset_builder = tfds.builder(dataset, data_dir=data_dir)
  elif isinstance(dataset, tfds.core.DatasetBuilder):
    dataset_builder = dataset
  else:
    raise ValueError("`dataset` must be a string or tfds.core.DatasetBuilder. "
                     f"Received {dataset} instead.")
  return dataset_builder


def _get_host_num_examples(builder, split, batch_size, process_index,
                           process_count, drop_remainder):
  """Returns the number of examples in a given host's split."""
  # NOTE: Multiple read instructions could be generated if `split` consists of
  # multiple parts.
  ris = deterministic_data.get_read_instruction_for_host(
      split,
      dataset_info=builder.info,
      host_id=process_index,
      host_count=process_count,
      drop_remainder=drop_remainder)
  # Passing a dict of SplitInfos will not work with tfds 4.4.0, which is the
  # latest external pip installable version, so we need to convert to a dict of
  # {split: num_examples}.
  splits = {
      k: v.num_examples if isinstance(v, tfds.core.SplitInfo) else v
      for k, v in builder.info.splits.items()
  }
  abs_ris = ris.to_absolute(splits)

  num_examples = 0
  for abs_ri in abs_ris:
    start = abs_ri.from_ or 0
    end = abs_ri.to or builder.info.splits[abs_ri.splitname].num_examples
    num_examples += end - start

  if drop_remainder:
    device_batch_size = batch_size // jax.local_device_count()
    num_examples = (
        math.floor(num_examples / device_batch_size) * device_batch_size)
  return num_examples


def get_num_examples(dataset: Union[str, tfds.core.DatasetBuilder],
                     split: str,
                     host_batch_size: int,
                     drop_remainder: bool = True,
                     num_hosts: Optional[int] = None,
                     data_dir: Optional[str] = None) -> int:
  """Returns the total number of examples in a (sharded) dataset split.

  Args:
    dataset: Either a dataset name or a dataset builder object.
    split: Specifies which split of the data to load. Passed to the function
      `get_read_instruction_for_host()`.
    host_batch_size: Per host batch size.
    drop_remainder: Whether to drop remainders when sharding across hosts and
      batching.
    num_hosts: Number of hosts across which the dataset will be sharded. If
      None, then the number of hosts will be obtained from
      `jax.process_count()`.
    data_dir: Directory for the dataset files.

  Returns:
    The number of examples in the dataset split that will be read when sharded
    across avaiable hosts.
  """
  dataset_builder = _get_dataset_builder(dataset, data_dir)
  if num_hosts is None:
    num_hosts = jax.process_count()

  num_examples = 0
  for i in range(num_hosts):
    num_examples += _get_host_num_examples(
        dataset_builder,
        split=split,
        batch_size=host_batch_size,
        process_index=i,
        process_count=num_hosts,
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


def _pad_reshape_mask_batch(batch, flat_batch_size, num_devices,
                            num_batch_dims):
  """Adds a mask, pads, and reshapes the tensors in the batch."""
  batch = _add_mask(batch, num_batch_dims)

  def f(x):
    if num_batch_dims > 1:
      x = tf.reshape(x, tf.concat([[-1], x.shape[num_batch_dims:]], axis=0))
    actual_batch_size = tf.shape(x)[0]
    needed = flat_batch_size - actual_batch_size
    zeros = tf.zeros(tf.concat([[needed], x.shape[1:]], axis=0), dtype=x.dtype)
    new_x = tf.concat([x, zeros], axis=0)
    new_x = tf.reshape(new_x, tf.concat([[num_devices, -1], x.shape[1:]],
                                        axis=0))
    return new_x

  new_batch = {k: f(v) for k, v in batch.items()}
  return new_batch


def get_data(
    dataset: Union[str, tfds.core.DatasetBuilder],
    split: str,
    rng: Union[None, jnp.ndarray, tf.Tensor],
    host_batch_size: int,
    preprocess_fn: Optional[Callable[[deterministic_data.Features],
                                     deterministic_data.Features]],
    cache: bool = False,
    num_epochs: Optional[int] = None,
    repeat_after_batching: bool = False,
    shuffle: bool = True,
    shuffle_buffer_size: int = 10_000,
    prefetch_size: int = 4,
    drop_remainder: bool = True,
    data_dir: Optional[str] = None,
) -> tf.data.Dataset:
  """Creates standard input pipeline (shuffle, preprocess, batch).

  Args:
    dataset: Either a dataset name or a dataset builder object.
    split: Specifies which split of the data to load. Passed to the function
      `get_read_instruction_for_host()`.
    rng: A jax.random.PRNG key or a tf.Tensor for TF stateless seeds to use for
      seeding shuffle operations and preprocessing ops. Must be set if
      shuffling.
    host_batch_size: Per host batch size.
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
      across hosts.
    data_dir: Directory for the dataset files.

  Returns:
    The dataset with preprocessed, masked, padded, and batched examples.
  """
  assert cache in ("loaded", "batched", False, None)

  dataset_builder = _get_dataset_builder(dataset, data_dir)

  if rng is not None:
    rng = jax.random.fold_in(rng,
                             jax.process_index())  # Derive RNG for this host.

  if drop_remainder:
    remainder_options = deterministic_data.RemainderOptions.DROP
  else:
    remainder_options = deterministic_data.RemainderOptions.BALANCE_ON_PROCESSES
  host_split = deterministic_data.get_read_instruction_for_host(
      split,
      dataset_info=dataset_builder.info,
      remainder_options=remainder_options)

  dataset = deterministic_data.create_dataset(
      dataset_builder,
      split=host_split,
      batch_dims=(),
      rng=rng,
      filter_fn=None,
      preprocess_fn=preprocess_fn,
      decoders={"image": tfds.decode.SkipDecoding()},
      cache=cache == "loaded",
      num_epochs=num_epochs if not repeat_after_batching else 1,
      shuffle=shuffle,
      shuffle_buffer_size=shuffle_buffer_size,
      prefetch_size=0,
      pad_up_to_batches=None,
      drop_remainder=drop_remainder,
  )

  num_devices = jax.local_device_count()
  if drop_remainder:
    batch_dims = [num_devices, host_batch_size // num_devices]
    for batch_size in reversed(batch_dims):
      dataset = dataset.batch(batch_size, drop_remainder=True)
    flat_batch_size = batch_dims[0] * batch_dims[1]
    num_batch_dims = len(batch_dims)
  else:
    batch_size_per_device = math.ceil(host_batch_size / num_devices)
    flat_batch_size = batch_size_per_device * num_devices
    dataset = dataset.batch(flat_batch_size, drop_remainder=False)
    num_batch_dims = 1

  def f(xs):
    return _pad_reshape_mask_batch(xs, flat_batch_size, num_devices,
                                   num_batch_dims)

  dataset = dataset.map(f, num_parallel_calls=tf.data.AUTOTUNE)

  if cache == "batched":
    dataset = dataset.cache()

  if repeat_after_batching:
    dataset = dataset.repeat(num_epochs)

  return dataset.prefetch(prefetch_size)


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

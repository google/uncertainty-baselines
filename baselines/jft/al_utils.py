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

"""Utils for active learning datasets."""
import dataclasses
import logging
import re
from typing import Dict, Iterable, List, Optional, Set, Union

from clu.deterministic_data import DatasetBuilder
import jax
import jax.numpy as jnp
import tensorflow as tf
import tensorflow_datasets as tfds


# adapted from https://www.tensorflow.org/datasets/determinism
class TFDSIdToInt:
  """Format TFDS ids to int."""
  tfds_id_parser_re = re.compile(r'\w+-(\w+).\w+-(\d+)-of-\d+__(\d+)')

  shard_offsets: Dict[str, List[int]]

  def __init__(self, dataset_info: tfds.core.DatasetInfo, *,
               splitwise_id: bool):
    shard_offsets = {}
    current_offset = 0

    # Build a unique offset across or within splits depending on `splitwise_id`.
    for split, split_info in sorted(dataset_info.splits.items()):
      if splitwise_id:
        current_offset = 0

      split_shard_offsets = []
      for shard_length in split_info.shard_lengths:
        current_length = current_offset + shard_length
        split_shard_offsets.append(current_offset)
        current_offset = current_length

      shard_offsets[split] = split_shard_offsets

    self.shard_offsets = shard_offsets

  def __call__(self, tfds_id: str) -> int:
    """Format the tfds_id in a more human-readable."""
    match = self.tfds_id_parser_re.match(tfds_id)
    split_name, shard_id, ex_id = match.groups()  # pytype: disable=attribute-error  # re-none
    shard_offsets = self.shard_offsets[split_name]
    return shard_offsets[int(shard_id)] + int(ex_id)


def _subset_generator(*, dataset: tf.data.Dataset,
                      dataset_info: tfds.core.DatasetInfo,
                      subset_ids: Optional[Set[int]], splitwise_id: bool):
  """Create a subset based on ids."""
  tfds_id_to_int = TFDSIdToInt(dataset_info, splitwise_id=splitwise_id)

  def inner():
    if subset_ids is not None:
      # Hard fail on type errors
      assert all(map(lambda id: isinstance(id, int), subset_ids))

    for record in dataset:
      tfds_id = record['tfds_id'].numpy().decode('utf-8')
      int_id = tfds_id_to_int(tfds_id)
      if subset_ids is None or int_id in subset_ids:
        record['id'] = tf.constant(int_id)
        yield record

  return inner


# TODO(dusenberrymw): Make this subclass tfds.core.DatasetBuilder instead.
class SubsetDatasetBuilder(DatasetBuilder):
  """Subset Dataset Builder which is "just right" for clu."""

  def __init__(self, base_dataset_builder: tfds.core.DatasetBuilder, *,
               subset_ids: Optional[Iterable[int]]):
    """Init function.

    Args:
      base_dataset_builder: a DatasetBuilder for the underlying dataset
        (essentially an object with a as_dataset method)
      subset_ids: a dictionary of split: set of ids.
    """
    self.subset_ids = set(subset_ids) if subset_ids is not None else None
    self.base_dataset_builder = base_dataset_builder
    self.info = base_dataset_builder.info

  def as_dataset(self,  # pytype: disable=signature-mismatch  # overriding-parameter-count-checks
                 split: Union[str, tfds.core.ReadInstruction],
                 *,
                 shuffle_files: bool = False,
                 read_config: Optional[tfds.ReadConfig] = None,
                 **kwargs) -> tf.data.Dataset:
    """Constructs a dataset containing a subset of the original dataset."""
    # Add the 'tfds_id' key to the samples which we can then parse.
    # From: https://www.tensorflow.org/datasets/api_docs/python/tfds/ReadConfig
    if read_config is None:
      logging.info('Using an empty ReadConfig!')
      read_config = tfds.ReadConfig()
    read_config = dataclasses.replace(read_config, add_tfds_id=True)

    # Since we are selecting a subset of the dataset based on example ids,
    # shuffling the files isn't necessary.
    dataset = self.base_dataset_builder.as_dataset(
        split=split, shuffle_files=False, read_config=read_config, **kwargs)

    element_spec = dataset.element_spec.copy()
    element_spec['id'] = tf.TensorSpec(shape=(), dtype=tf.int64, name=None)
    logging.info(msg=f'element_spec = {element_spec}; '
                 f'type = {jax.tree_map(type, element_spec)}')

    dataset = tf.data.Dataset.from_generator(
        _subset_generator(
            dataset=dataset,
            dataset_info=self.base_dataset_builder.info,
            subset_ids=self.subset_ids,
            splitwise_id=True),
        output_signature=element_spec,
    )
    return dataset


def sample_class_balanced_ids(
    n,
    dataset,
    num_classes,
    # shuffle_buffer_size=50_000,
    shuffle_rng):
  """Return n class balanced sampled ids."""
  logging.info('Preparing dataset.')
  assert n % num_classes == 0, (f'Total #samples {n} is not a '
                                f'multiplier of num_classes {num_classes}.')
  # The commented-out implementation has OOM issues.

  # dataset = dataset.shuffle(shuffle_buffer_size, seed=shuffle_seed)
  # dataset = dataset.prefetch(1)
  # def _make_filter_fn(label):
  #   return lambda x: x['label'] == label

  # datasets = []
  # for label in range(num_classes):
  #   datasets.append(dataset.filter(_make_filter_fn(label)))
  # choice_dataset = tf.data.Dataset.range(num_classes).repeat(n // num_classes)
  # dataset = tf.data.Dataset.choose_from_datasets(datasets, choice_dataset)
  # result = dataset.map(lambda x: x['id'], num_parallel_calls=tf.data.AUTOTUNE)
  # result = result.prefetch(1)
  # logging.info('Result obtained.')
  # return list(result.as_numpy_iterator())

  iter_ds = iter_ds = iter(dataset)

  ids = []
  labels = []
  masks = []
  for _, batch in enumerate(iter_ds):
    batch_id = batch['id'].numpy()
    batch_label = batch['labels'].numpy()
    batch_mask = batch['mask'].numpy()

    # TODO(joost,andreas): if we run on multi host, we need to index
    # batch_outputs: batch_outputs[0]
    ids.append(batch_id)
    labels.append(batch_label)
    masks.append(batch_mask)

  ids = jnp.concatenate(ids, axis=1).flatten()
  labels = jnp.concatenate(labels, axis=1)
  labels = jnp.argmax(labels, axis=2).flatten()
  masks = jnp.concatenate(masks, axis=1).flatten()
  labels = jnp.where(masks, labels, -1)
  shuffled_index = jax.random.permutation(shuffle_rng, len(ids))
  ids = ids[shuffled_index]
  labels = labels[shuffled_index]
  result = []
  for label in range(num_classes):
    index_with_label = jnp.argwhere(label == labels, size=n // num_classes)
    index_with_label = index_with_label.flatten()
    result.extend(ids[index_with_label].tolist())
  return result

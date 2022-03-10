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

"""Utils for active learning datasets."""
import dataclasses
import logging
import re
from typing import Dict, Iterable, List, Optional, Set, Union

from clu.deterministic_data import DatasetBuilder
import jax
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
    split_name, shard_id, ex_id = match.groups()
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

  def as_dataset(self,
                 split: Union[str, tfds.core.ReadInstruction],
                 *,
                 shuffle_files: bool = False,
                 read_config: Optional[tfds.ReadConfig] = None,
                 **kwargs) -> tf.data.Dataset:
    # We don't allow an empty split by virtue of the parameter declaration,
    # so we always have a split.
    read_config = dataclasses.replace(
        kwargs.pop('read_config', tfds.ReadConfig()))
    # Add the 'tfds_id' key to the samples which we can then parse.
    # From: https://www.tensorflow.org/datasets/api_docs/python/tfds/ReadConfig
    read_config.add_tfds_id = True

    dataset = self.base_dataset_builder.as_dataset(
        split=split, shuffle_files=False, read_config=read_config, **kwargs)

    element_spec = dataset.element_spec.copy()
    element_spec['id'] = tf.TensorSpec(shape=(), dtype=tf.int64, name=None)
    logging.info(msg=f'element_spec = {element_spec}; '
                 f'type = {jax.tree_map(type, element_spec)}')

    dataset = tf.data.Dataset.from_generator(
        _subset_generator(
            dataset=dataset,
            subset_ids=self.subset_ids,
            splitwise_id=True),
        output_signature=element_spec,
    )

    # This is a bit more complex: potentially cache before or after calling
    # .shuffle. BUT don't cache for the pool set as it will be much larger than
    # the training set.
    reshuffle_each_iteration = shuffle_files and read_config.shuffle_reshuffle_each_iteration
    cache_data = self.subset_ids is not None

    if reshuffle_each_iteration and cache_data:
      dataset = dataset.cache()
    if shuffle_files:
      if self.subset_ids is not None:
        buffer_size = len(self.subset_ids)
      else:
        # TODO(andreas): what buffer size do we want actually for shuffling?
        #   10k seems like a safe thing.
        buffer_size = 10000
      dataset = dataset.shuffle(
          buffer_size=buffer_size,
          seed=read_config.shuffle_seed,
          reshuffle_each_iteration=read_config.shuffle_reshuffle_each_iteration)
    if not reshuffle_each_iteration and cache_data:
      dataset = dataset.cache()
    return dataset

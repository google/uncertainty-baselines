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
from typing import Any, Iterable, Optional, Set, Union

import jax
import tensorflow as tf
import tensorflow_datasets as tfds


def _subset_generator(*, dataset: tf.data.Dataset,
                      subset_ids: Optional[Set[int]]):
  """Create a subset based on ids."""

  def inner():
    if subset_ids is not None:
      # Hard fail on type errors
      assert all(map(lambda id: isinstance(id, int), subset_ids))

    #for int_id, record in enumerate(dataset):
    for i, record in dataset.enumerate():
      int_id = int(i)
      if subset_ids is None or int_id in subset_ids:
        print(f"{int_id}, {record['tfds_id']}")
        record['id'] = tf.constant(int_id)
        yield record

  return inner


class SubsetDatasetBuilder(tfds.core.DatasetBuilder):
  """Subset Dataset Builder."""
  VERSION = tfds.core.Version('1.0.0')

  def __init__(self, base_dataset_builder: tfds.core.DatasetBuilder, *,
               subset_ids: Optional[Iterable[int]], **kwargs: Any):
    """Init function.

    Args:
      base_dataset_builder: A TFDS DatasetBuilder for the underlying dataset.
      subset_ids: An optional list of ids. If none, then all examples are
        returned.
      **kwargs: Additional keyword arguments.
    """
    self.subset_ids = set(subset_ids) if subset_ids is not None else None
    self.base_dataset_builder = base_dataset_builder
    super().__init__(**kwargs)

  def as_dataset(self,
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
        _subset_generator(dataset=dataset, subset_ids=self.subset_ids),
        output_signature=element_spec,
    )
    return dataset

  def _info(self) -> tfds.core.DatasetInfo:
    """Dataset metadata (homepage, citation,...)."""
    # TODO(dusenberrymw): Check that split sizes are correct.
    # TODO(dusenberrymw): Add an extra 'id' feature.
    return self.base_dataset_builder.info

  # TODO(dusenberrymw): Switch to subclassing tfds.core.GeneratorBasedBuilder to
  # avoid needing to include these.
  def _download_and_prepare(self, dl_manager, download_config=None):
    """Downloads and prepares dataset for reading."""
    raise NotImplementedError

  def _as_dataset(self, split, decoders, read_config, shuffle_files=False):
    """Constructs a `tf.data.Dataset`."""
    raise NotImplementedError

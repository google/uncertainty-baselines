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

"""Custom CIFAR-10 dataset that returns a subset for training.

Alternative version that writes out TFRecords here:
https://colab.research.google.com/drive/1McRC0es1ehwUL_jQQcBdC05wsPGgNWE3
"""
import logging
import jax
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

cifar = tfds.image_classification.cifar


def _subset_generator(dataset, subset_ids):
  """Create a subset based on ids."""

  def inner():
    for record in dataset:
      # HACK: the last 5 characters of 'id' are generally the ids.
      # However, the dataset info only specifies that the type of id
      # is BYTES, so we also support that.
      try:
        int_id = np.int32(record['id'].numpy()[-5:])
      except ValueError:
        # ID is encoded differently, then just interpret as bytes
        int_id = int.from_bytes(record['id'].numpy(), 'big')
        # jax prefers int32
        # https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html#double-64bit-precision
        int_id = np.int32(int_id % np.iinfo(np.int32).max)

      if subset_ids is not None:
        # Hard fail on type errors
        assert all(map(lambda id: isinstance(id, int), subset_ids))

      if subset_ids is None or int_id in subset_ids:
        record['id'] = tf.constant(int_id)
        yield record

  return inner


class Cifar10Subset(cifar.Cifar10):
  """CIFAR-10 Subset.

    Implement CIFAR-10 with the added functionality taking a subst by id.

    Args:
      subset_ids: a dictionary of split: set of ids.

    Returns:
      A Cifar10Subset object.
  """

  # TODO(trandustin, wangzi): Add type annotation for subset_id that works.
  # def __init__(self, *, subset_ids: Dict[str, Set[int]], **kwargs):
  def __init__(self, *, subset_ids, data_dir=None, **kwargs):
    super().__init__(data_dir=data_dir, **kwargs)
    self.subset_ids = subset_ids

  def _as_dataset(self, *args, split, **kwargs):
    dataset = super()._as_dataset(*args, split=split, **kwargs)

    # HACK: Fix id to be an int. Assuming 'label' is an int, too.
    element_spec = dataset.element_spec.copy()
    logging.info(msg=f'element_spec = {element_spec}; '
                 f'type = {jax.tree_map(type, element_spec)}')
    element_spec['id'] = element_spec['label']

    # NOTE: if this line errors out, make sure to update your
    # tensorflow-datasets package to the right version.
    split_name = split.split

    return tf.data.Dataset.from_generator(
        _subset_generator(dataset, self.subset_ids[split_name]),
        output_signature=element_spec,
    ).cache()

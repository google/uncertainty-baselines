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

"""Library of dataloaders to use in Introspective Active Sampling.

This file contains a library of dataloaders that return three features for each
example: example_id, input feature, and label. The example_id is a unique ID
that will be used to keep track of the bias label for that example. The input
feature will vary depending on the type of data (feature vector, image, etc.),
and the label is specific to the main task.
"""

import dataclasses
import json
from typing import Any, Dict, Iterator, List, Optional, Tuple

import tensorflow as tf
import tensorflow_datasets as tfds


DATASET_REGISTRY = {}
DATA_DIR = '/tmp'
_DEFAULT_PATH_CARDIOTOX_TRAIN_FEATURE = ''
_DEFAULT_PATH_CARDIOTOX_TRAIN_LABEL = ''
_DEFAULT_PATH_CARDIOTOX_VALIDATION_FEATURE = ''
_DEFAULT_PATH_CARDIOTOX_VALIDATION_LABEL = ''
_DEFAULT_PATH_CARDIOTOX_TEST_FEATURE = ''
_DEFAULT_PATH_CARDIOTOX_TEST_LABEL = ''
_DEFAULT_PATH_CARDIOTOX_TEST2_FEATURE = ''
_DEFAULT_PATH_CARDIOTOX_TEST2_LABEL = ''


def register_dataset(name: str):
  """Provides decorator to register functions that return dataset."""
  def save(dataset_builder):
    DATASET_REGISTRY[name] = dataset_builder
    return dataset_builder

  return save


def get_dataset(name: str):
  """Retrieves dataset based on name."""
  if name not in DATASET_REGISTRY:
    raise ValueError(
        f'Unknown dataset: {name}\nPossible choices: {DATASET_REGISTRY.keys()}')
  return DATASET_REGISTRY[name]


@dataclasses.dataclass
class Dataloader:
  train_splits: tf.data.Dataset  # Result of tfds.load with 'split' arg.
  val_splits: tf.data.Dataset  # Result of tfds.load with 'split' arg.
  train_ds: tf.data.Dataset  # Dataset with all the train splits combined.
  eval_ds: Optional[Dict[
      str,
      tf.data.Dataset]] = None  # Validation and any additional test datasets.


def gather_data_splits(slice_idx: List[int],
                       dataset: tf.data.Dataset) -> tf.data.Dataset:
  """Gathers slices of a split dataset based on passed indices."""
  data_slice = dataset[slice_idx[0]]
  for idx in slice_idx[1:]:
    data_slice = data_slice.concatenate(dataset[idx])
  return data_slice


class CardiotoxFingerprintDataset(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for cardiotoxicity fingerprint dataset."""

  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
  }

  def _info(self) -> tfds.core.DatasetInfo:
    """Dataset metadata (homepage, citation,...)."""
    return tfds.core.DatasetInfo(
        builder=self,
        features=tfds.features.FeaturesDict({
            'example_id': tfds.features.Text(),
            'feature': tfds.features.Tensor(shape=(1024,),
                                            dtype=tf.int64),
            'label': tfds.features.ClassLabel(num_classes=2),
        }),
        supervised_keys=('feature', 'label', 'example_id'),
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Loads data and defines splits."""
    return {
        'train': self._generate_examples(
            features_path=_DEFAULT_PATH_CARDIOTOX_TRAIN_FEATURE,
            label_path=_DEFAULT_PATH_CARDIOTOX_TRAIN_LABEL),
        'validation': self._generate_examples(
            features_path=_DEFAULT_PATH_CARDIOTOX_VALIDATION_FEATURE,
            label_path=_DEFAULT_PATH_CARDIOTOX_VALIDATION_LABEL),
        'test': self._generate_examples(
            features_path=_DEFAULT_PATH_CARDIOTOX_TEST_FEATURE,
            label_path=_DEFAULT_PATH_CARDIOTOX_TEST_LABEL),
        'test2': self._generate_examples(
            features_path=_DEFAULT_PATH_CARDIOTOX_TEST2_FEATURE,
            label_path=_DEFAULT_PATH_CARDIOTOX_TEST2_LABEL),
    }

  def _generate_examples(self, features_path, label_path) -> Iterator[
      Tuple[str, Dict[str, Any]]]:
    """Generates examples for each split."""
    with tf.io.gfile.GFile(features_path, 'r') as features_file:
      x = json.load(features_file)
    with tf.io.gfile.GFile(label_path, 'r') as label_file:
      y = json.load(label_file)
    for idx, molecule_id in enumerate(x):
      key = '_'.join([molecule_id, str(idx)])
      yield key, {
          'example_id': molecule_id,
          'feature': x[molecule_id],
          'label': y[molecule_id]
      }


@register_dataset('cardiotoxicity')
def get_cardiotoxicity_dataset(
    num_splits: int, batch_size: int
) -> Dataloader:
  """Returns datasets for training, validation, and possibly test sets.

  Args:
    num_splits: Integer for number of slices of the dataset.
    batch_size: Integer for number of examples per batch.

  Returns:
    A tuple containing the split training data, split validation data, the
    combined training dataset, and a dictionary mapping evaluation dataset names
    to their respective combined datasets.
  """
  split_size_in_pct = int(100 / num_splits)
  val_splits = tfds.load(
      'cardiotox_fingerprint_dataset',
      split=[
          f'validation[{k}%:{k+split_size_in_pct}%]'
          for k in range(0, 100, split_size_in_pct)
      ],
      data_dir=DATA_DIR,
      batch_size=batch_size,
      try_gcs=False,
      as_supervised=True)

  train_splits = tfds.load(
      'cardiotox_fingerprint_dataset',
      split=[
          f'train[{k}%:{k+split_size_in_pct}%]'
          for k in range(0, 100, split_size_in_pct)
      ],
      data_dir=DATA_DIR,
      batch_size=batch_size,
      try_gcs=False,
      as_supervised=True)

  test_ds = tfds.load(
      'cardiotox_fingerprint_dataset',
      split='test',
      data_dir=DATA_DIR,
      batch_size=batch_size,
      try_gcs=False,
      as_supervised=True,
      with_info=False)

  test2_ds = tfds.load(
      'cardiotox_fingerprint_dataset',
      split='test2',
      data_dir=DATA_DIR,
      batch_size=batch_size,
      try_gcs=False,
      as_supervised=True,
      with_info=False)

  train_ds = gather_data_splits(list(range(num_splits)), train_splits)
  val_ds = gather_data_splits(list(range(num_splits)), val_splits)
  eval_datasets = {
      'val': val_ds,
      'test': test_ds,
      'test2': test2_ds
  }
  return Dataloader(train_splits, val_splits, train_ds, eval_datasets)



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

"""Data loader for the Criteo dataset."""

import os.path
from typing import Dict, Optional, Union

from absl import logging
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds
from uncertainty_baselines.datasets import base


NUM_INT_FEATURES = 13  # Number of Criteo integer features.
NUM_CAT_FEATURES = 26  # Number of Criteo categorical features.
NUM_TOTAL_FEATURES = NUM_INT_FEATURES + NUM_CAT_FEATURES

_INT_KEY_TMPL = 'int-feature-%d'
_CAT_KEY_TMPL = 'categorical-feature-%d'


def _build_dataset(glob_dir: str, is_training: bool) -> tf.data.Dataset:
  cycle_len = 10 if is_training else 1
  dataset = tf.data.Dataset.list_files(glob_dir, shuffle=is_training)
  dataset = dataset.interleave(tf.data.TFRecordDataset, cycle_length=cycle_len)
  return dataset


def feature_name(idx: int) -> str:
  assert 0 < idx <= NUM_TOTAL_FEATURES
  if idx <= NUM_INT_FEATURES:
    return _INT_KEY_TMPL % idx
  return _CAT_KEY_TMPL % idx


def _make_features_spec() -> Dict[str, tf.io.FixedLenFeature]:
  features = {'clicked': tf.io.FixedLenFeature([1], tf.float32)}
  for idx in range(1, NUM_INT_FEATURES + 1):
    features[feature_name(idx)] = tf.io.FixedLenFeature([1], tf.float32, -1)
  for idx in range(NUM_INT_FEATURES + 1, NUM_TOTAL_FEATURES + 1):
    features[feature_name(idx)] = tf.io.FixedLenFeature([1], tf.string, '')
  return features


def apply_randomization(features, label, randomize_prob):
  """Randomize each categorical feature with some probability."""
  for idx in range(NUM_INT_FEATURES + 1, NUM_TOTAL_FEATURES + 1):
    key = feature_name(idx)

    def rnd_tok():
      return tf.as_string(
          tf.random.uniform(tf.shape(features[key]), 0, 99999999, tf.int32))  # pylint: disable=cell-var-from-loop

    # Ignore lint since tf.cond should evaluate lambda immediately.
    features[key] = tf.cond(tf.random.uniform([]) < randomize_prob,
                            rnd_tok,
                            lambda: features[key])  # pylint: disable=cell-var-from-loop
  return features, label


_CITATION = """
@article{criteo,
  title = {Display Advertising Challenge},
  url = {https://www.kaggle.com/c/criteo-display-ad-challenge.},
}
"""


class _CriteoDatasetBuilder(tfds.core.DatasetBuilder):
  """Minimal TFDS DatasetBuilder for Criteo, does not support downloading."""
  VERSION = tfds.core.Version('0.0.0')

  def __init__(self, data_dir, **kwargs):
    super().__init__(data_dir=data_dir, **kwargs)
    # We have to override self._data_dir to prevent the parent class from
    # appending the class name and version.
    self._data_dir = data_dir

  def _download_and_prepare(self, dl_manager, download_config=None):
    """Downloads and prepares dataset for reading."""
    raise NotImplementedError(
        'Must provide a data_dir with the files already downloaded to.')

  def _as_dataset(
      self,
      split: tfds.Split,
      decoders=None,
      read_config=None,
      shuffle_files=False) -> tf.data.Dataset:
    """Constructs a `tf.data.Dataset`.

    Args:
      split: `tfds.Split` which subset of the data to read.
      decoders: Unused.
      read_config: Unused.
      shuffle_files: Unused.

    Returns:
      `tf.data.Dataset`
    """
    del decoders
    del read_config
    del shuffle_files
    is_training = False
    if isinstance(split, tfds.core.ReadInstruction):
      logging.warn(
          'ReadInstruction splits are currently not supported. Using '
          'the split name `%s` instead of `%s`.', split.split_name, split)
      split = tfds.Split(split.split_name)
    if split == tfds.Split.TRAIN:
      file_pattern = 'train-*-of-*'
      is_training = True
    elif split == tfds.Split.VALIDATION:
      file_pattern = 'validation-*-of-*'
    elif split == tfds.Split.TEST:
      file_pattern = 'test-*-of-*'
    else:
      raise ValueError('Unsupported split given: {}.'.format(split))
    return _build_dataset(
        glob_dir=os.path.join(self._data_dir, file_pattern),
        is_training=is_training)

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the `tfds.core.DatasetInfo` object."""
    features = {'clicked': tfds.features.ClassLabel(num_classes=2)}
    for idx in range(1, NUM_INT_FEATURES + 1):
      features[feature_name(idx)] = tfds.features.Tensor(
          shape=(1,), dtype=tf.float32)
    for idx in range(NUM_INT_FEATURES + 1, NUM_TOTAL_FEATURES + 1):
      features[feature_name(idx)] = tfds.features.Tensor(
          shape=(1,), dtype=tf.string)
    info = tfds.core.DatasetInfo(
        builder=self,
        description='Criteo Display Advertising Challenge',
        features=tfds.features.FeaturesDict(features),
        homepage='https://www.kaggle.com/c/criteo-display-ad-challenge/data',
        citation=_CITATION,
        metadata=None)
    # Instead of having a single element shard_lengths, we should really have a
    # list of the number of elements in each file shard in each split.
    split_infos = [
        tfds.core.SplitInfo(
            name=tfds.Split.VALIDATION,
            shard_lengths=[4420308],
            num_bytes=0,
        ),
        tfds.core.SplitInfo(
            name=tfds.Split.TEST,
            shard_lengths=[4420309],
            num_bytes=0,
        ),
        tfds.core.SplitInfo(
            name=tfds.Split.TRAIN,
            shard_lengths=[int(37e6)],
            num_bytes=0,
        ),
    ]
    split_dict = tfds.core.SplitDict(
        split_infos, dataset_name='__criteo_dataset_builder')
    info.set_splits(split_dict)
    return info


class CriteoDataset(base.BaseDataset):
  """Criteo dataset builder class."""

  def __init__(self,
               split: Union[float, str],
               seed: Optional[Union[int, tf.Tensor]] = None,
               validation_percent: float = 0.0,
               shuffle_buffer_size: Optional[int] = None,
               num_parallel_parser_calls: int = 64,
               data_dir: Optional[str] = None,
               is_training: Optional[bool] = None):
    """Create a Criteo tf.data.Dataset builder.

    Args:
      split: a dataset split, either a custom tfds.Split or one of the
        tfds.Split enums [TRAIN, VALIDAITON, TEST] or their lowercase string
        names. For Criteo it can also be a float to represent the level of data
        augmentation.
      seed: the seed used as a source of randomness.
      validation_percent: the percent of the training set to use as a validation
        set.
      shuffle_buffer_size: the number of example to use in the shuffle buffer
        for tf.data.Dataset.shuffle().
      num_parallel_parser_calls: the number of parallel threads to use while
        preprocessing in tf.data.Dataset.map().
      data_dir: Path to a directory containing the Criteo datasets, with
        filenames train-*-of-*', 'validate.tfr', 'test.tfr'.
      is_training: Whether or not the given `split` is the training split. Only
        required when the passed split is not one of ['train', 'validation',
        'test', tfds.Split.TRAIN, tfds.Split.VALIDATION, tfds.Split.TEST].
    """
    # If receive a corruption level as a split, load the test set and save the
    # corruption level for use in preprocessing.
    if isinstance(split, float):
      self._corruption_level = split
      split = 'test'
    else:
      self._corruption_level = None
    dataset_builder = _CriteoDatasetBuilder(data_dir=data_dir)
    if is_training is None:
      is_training = split in ['train', tfds.Split.TRAIN]
    new_split = base.get_validation_percent_split(dataset_builder,
                                                  validation_percent, split)
    super().__init__(
        name='criteo',
        dataset_builder=dataset_builder,
        split=new_split,
        seed=seed,
        is_training=is_training,
        shuffle_buffer_size=shuffle_buffer_size,
        num_parallel_parser_calls=num_parallel_parser_calls,
        download_data=False)

  def _create_process_example_fn(self) -> base.PreProcessFn:

    def _example_parser(example: Dict[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
      """Parse features and labels from a serialized tf.train.Example."""
      features_spec = _make_features_spec()
      features = tf.io.parse_example(example['features'], features_spec)
      features = {k: tf.squeeze(v, axis=0) for k, v in features.items()}
      labels = tf.cast(features.pop('clicked'), tf.int32)

      if self._corruption_level is not None:
        if self._corruption_level < 0.0 or self._corruption_level > 1.0:
          raise ValueError('shift_level not in [0, 1]: {}'.format(
              self._corruption_level))
        features, labels = apply_randomization(
            features, labels, self._corruption_level)

      return {
          'features': features,
          'labels': labels,
      }

    return _example_parser

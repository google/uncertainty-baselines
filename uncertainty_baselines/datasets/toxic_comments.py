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

"""Data loader for the Jigsaw toxicity classification datasets."""
import os
from typing import Any, Dict, Optional, Tuple

from absl import logging

import tensorflow as tf
import tensorflow_datasets as tfds

from uncertainty_baselines.datasets import base

_TOXICITY_SUBTYPE_NAMES = ('identity_attack', 'insult', 'obscene',
                           'severe_toxicity', 'sexual_explicit', 'threat')

# CivilCommentsIdentities additional identity labels.
_IDENTITY_LABELS = ('male', 'female', 'transgender', 'other_gender',
                    'heterosexual', 'homosexual_gay_or_lesbian', 'bisexual',
                    'other_sexual_orientation', 'christian', 'jewish', 'muslim',
                    'hindu', 'buddhist', 'atheist', 'other_religion', 'black',
                    'white', 'asian', 'latino', 'other_race_or_ethnicity',
                    'physical_disability',
                    'intellectual_or_learning_disability',
                    'psychiatric_or_mental_illness', 'other_disability')

_DATA_SPLIT_NAMES = map(
    str, [tfds.Split.TRAIN, tfds.Split.VALIDATION, tfds.Split.TEST])

_TF_RECORD_NAME_PATTERNS = {
    name: name + '_*.tfrecord' for name in _DATA_SPLIT_NAMES
}


def _build_dataset(glob_dir: str, is_training: bool) -> tf.data.Dataset:
  cycle_len = 10 if is_training else 1
  dataset = tf.data.Dataset.list_files(glob_dir, shuffle=is_training)
  dataset = dataset.interleave(tf.data.TFRecordDataset, cycle_length=cycle_len)
  return dataset


def _make_features_spec(
    max_seq_length: int,
    additional_labels: Tuple[str]) -> Dict[str, tf.io.FixedLenFeature]:
  """Makes a specification dictionary for reading / writing TF Examples."""
  features_spec_dict = {
      'id': tf.io.FixedLenFeature([], tf.int64),
      'input_ids': tf.io.FixedLenFeature([max_seq_length], tf.int64),
      'input_mask': tf.io.FixedLenFeature([max_seq_length], tf.int64),
      'segment_ids': tf.io.FixedLenFeature([max_seq_length], tf.int64),
      'features': tf.io.FixedLenFeature([], tf.string),
      'labels': tf.io.FixedLenFeature([], tf.float32),
  }

  for subtype in additional_labels:
    features_spec_dict.update({subtype: tf.io.FixedLenFeature([], tf.float32)})

  return features_spec_dict


class _JigsawToxicityDatasetBuilder(tfds.core.DatasetBuilder):
  """Minimal TFDS DatasetBuilder for the Jigsaw toxicity dataset."""
  VERSION = tfds.core.Version('0.0.0')

  def __init__(
      self,
      tfds_dataset_builder: tfds.core.DatasetBuilder,
      max_seq_length: int,
      data_dir: Optional[str],
      **kwargs):
    self._tfds_dataset_builder = tfds_dataset_builder
    self._max_seq_length = max_seq_length
    super().__init__(
        data_dir=data_dir, **kwargs)
    # We have to override self._data_dir to prevent the parent class from
    # appending the class name and version.
    self._data_dir = data_dir

  def _download_and_prepare(self, dl_manager, download_config=None):
    """Downloads and prepares dataset for reading."""
    return self._tfds_dataset_builder._download_and_prepare(  # pylint: disable=protected-access
        dl_manager, download_config)

  def _as_dataset(
      self,
      split: tfds.Split,
      decoders=None,
      read_config=None,
      shuffle_files=False) -> tf.data.Dataset:
    raise NotImplementedError

  # Note that we override `as_dataset` instead of `_as_dataset` to avoid any
  # `data_dir` reading logic.
  def as_dataset(
      self,
      split: tfds.Split,
      *,
      batch_size=None,
      decoders=None,
      read_config=None,
      shuffle_files=False,
      as_supervised=False) -> tf.data.Dataset:
    """Constructs a `tf.data.Dataset`, see parent class for documentation."""
    if self._data_dir:
      # Reading locally.
      logging.info('Reading from local TFRecords with BERT features %s',
                   self._data_dir)
      is_training = split == tfds.Split.TRAIN
      return _build_dataset(
          glob_dir=os.path.join(
              self._data_dir, _TF_RECORD_NAME_PATTERNS[split]),
          is_training=is_training)
    else:
      logging.info('Reading from TFDS.')
      return self._tfds_dataset_builder.as_dataset(
          split=split,
          decoders=decoders,
          read_config=read_config,
          shuffle_files=shuffle_files,
          batch_size=batch_size,
          as_supervised=as_supervised)

  def _info(self) -> tfds.core.DatasetInfo:
    raise NotImplementedError

  # Note that we are overriding info instead of _info() so that we properly
  # generate the full DatasetInfo.
  @property
  def info(self) -> tfds.core.DatasetInfo:
    """Returns the `tfds.core.DatasetInfo` object."""
    info = self._tfds_dataset_builder.info
    if info.metadata is None:
      info._metadata = tfds.core.MetadataDict()  # pylint: disable=protected-access
    info.metadata['num_classes'] = 1
    info.metadata['max_seq_length'] = self._max_seq_length
    return info


class _JigsawToxicityDataset(base.BaseDataset):
  """Dataset builder abstract class."""

  def __init__(self,
               name: str,
               split: str,
               additional_labels: Tuple[str] = _TOXICITY_SUBTYPE_NAMES,
               validation_percent: float = 0.0,
               shuffle_buffer_size: Optional[int] = None,
               max_seq_length: Optional[int] = 512,
               num_parallel_parser_calls: int = 64,
               drop_remainder: bool = False,
               try_gcs: bool = False,
               download_data: bool = False,
               data_dir: Optional[str] = None,
               is_training: Optional[bool] = None):  # pytype: disable=annotation-type-mismatch
    """Create a tf.data.Dataset builder.

    Args:
      name: name of the dataset.
      split: a dataset split, either a custom tfds.Split or one of the
        tfds.Split enums [TRAIN, VALIDAITON, TEST] or their lowercase string
        names.
      additional_labels: names of additional labels (e.g. toxicity subtypes),
        as well as identity labels for the case of CivilCommentsIdentities.
      validation_percent: the percent of the training set to use as a
        validation set.
      shuffle_buffer_size: the number of example to use in the shuffle buffer
        for tf.data.Dataset.shuffle().
      max_seq_length: maximum sequence length of the tokenized sentences.
      num_parallel_parser_calls: the number of parallel threads to use while
        preprocessing in tf.data.Dataset.map().
      drop_remainder: whether or not to drop the last batch of data if the
        number of points is not exactly equal to the batch size.
      try_gcs: Whether or not to try to use the GCS stored versions of dataset
        files. Currently unsupported.
      download_data: Whether or not to download data before loading. Currently
        unsupported.
      data_dir: optional dir to read data from. If none then the local
        filesystem is used. Required for using TPUs on Cloud.
      is_training: Whether or not the given `split` is the training split. Only
        required when the passed split is not one of ['train', 'validation',
        'test', tfds.Split.TRAIN, tfds.Split.VALIDATION, tfds.Split.TEST].
    """
    dataset_builder = _JigsawToxicityDatasetBuilder(
        tfds.builder(name, try_gcs=try_gcs), max_seq_length, data_dir)
    self.additional_labels = additional_labels

    self.feature_spec = _make_features_spec(max_seq_length, additional_labels)
    self.split_names = _DATA_SPLIT_NAMES
    self._data_dir = data_dir

    if is_training is None:
      is_training = split in ['train', tfds.Split.TRAIN]

    if validation_percent == 0:
      # This value will never be used.
      num_validation_examples = (
          dataset_builder.info.splits['validation'].num_examples)
    else:
      num_train_examples = dataset_builder.info.splits['train'].num_examples
      num_validation_examples = (
          int(num_train_examples * validation_percent))

    # Reading locally does not support tfds.core.ReadInstruction (yet), so we
    # also default to split = {'train', 'validation'} if data_dir is provided.
    if split == tfds.Split.TRAIN:
      if validation_percent == 0 or data_dir:
        split = 'train'
      else:
        split = tfds.core.ReadInstruction(
            'train', to=-num_validation_examples, unit='abs')
    elif split == tfds.Split.VALIDATION:
      if validation_percent == 0 or data_dir:
        split = 'validation'
      else:
        split = tfds.core.ReadInstruction(
            'train', from_=-num_validation_examples, unit='abs')

    super().__init__(
        name=name,
        dataset_builder=dataset_builder,
        split=split,
        is_training=is_training,
        shuffle_buffer_size=shuffle_buffer_size,
        num_parallel_parser_calls=num_parallel_parser_calls,
        drop_remainder=drop_remainder,
        download_data=download_data)

  def _create_process_example_fn(self) -> base.PreProcessFn:
    """Create a pre-process function to return labels and sentence tokens."""

    def _example_parser(example: Dict[str, tf.Tensor]) -> Dict[str, Any]:
      """Preprocesses sentences as well as toxicity and other labels."""
      if self._data_dir:
        return tf.io.parse_example(example['features'], self.feature_spec)
      else:
        label = example['toxicity']
        feature = example['text']

        # Read in sentences.
        parsed_example = {'features': feature, 'labels': label}

        # Add additional labels.
        parsed_example.update({
            subtype_name: example[subtype_name]
            for subtype_name in self.additional_labels
        })

        return parsed_example

    return _example_parser


class WikipediaToxicityDataset(_JigsawToxicityDataset):
  """Data loader for Wikipedia Toxicity Subtype datasets."""

  def __init__(self, *, validation_percent: float = 0.1, **kwargs):
    # Requires validation_percent > 0 since this dataset doesn't have
    # its own validation set.
    if validation_percent <= 0:
      raise ValueError('validation_percent must be positive.')

    additional_labels = tuple(
        toxicity_name for toxicity_name in _TOXICITY_SUBTYPE_NAMES
        if toxicity_name != 'sexual_explicit')

    super().__init__(
        name='wikipedia_toxicity_subtypes',
        validation_percent=validation_percent,
        additional_labels=additional_labels,
        **kwargs)


class CivilCommentsDataset(_JigsawToxicityDataset):
  """Data loader for Civil Comments datasets."""

  def __init__(self, **kwargs):
    super().__init__(name='civil_comments', **kwargs)


class CivilCommentsIdentitiesDataset(_JigsawToxicityDataset):
  """Data loader for Civil Comments Identities datasets."""

  def __init__(self, **kwargs):
    super().__init__(
        name='civil_comments/CivilCommentsIdentities',
        additional_labels=_TOXICITY_SUBTYPE_NAMES + _IDENTITY_LABELS,
        **kwargs)

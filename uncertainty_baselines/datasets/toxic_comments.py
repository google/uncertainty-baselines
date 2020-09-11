# coding=utf-8
# Copyright 2020 The Uncertainty Baselines Authors.
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

# Lint as: python3
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

_DATA_SPLIT_NAMES = (base.Split.TRAIN.value, base.Split.TEST.value,
                     base.Split.VAL.value)

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
      'labels': tf.io.FixedLenFeature([], tf.float32)
  }

  for subtype in additional_labels:
    features_spec_dict.update({subtype: tf.io.FixedLenFeature([], tf.float32)})

  return features_spec_dict


class _JigsawToxicityDataset(base.BaseDataset):
  """Dataset builder abstract class."""

  def __init__(
      self,
      name: str,
      batch_size: int,
      eval_batch_size: int,
      additional_labels: Tuple[str] = _TOXICITY_SUBTYPE_NAMES,
      validation_fraction: float = 0.0,
      shuffle_buffer_size: Optional[int] = None,
      max_seq_length: Optional[int] = 512,
      num_parallel_parser_calls: int = 64,
      data_dir: Optional[str] = None,
      **unused_kwargs: Dict[str, Any]):
    """Create a tf.data.Dataset builder.

    Args:
      name: name of the dataset.
      batch_size: the training batch size.
      eval_batch_size: the validation and test batch size.
      additional_labels: names of additional labels (e.g. toxicity subtypes),
        as well as identity labels for the case of CivilCommentsIdentities.
      validation_fraction: the percent of the training set to use as a
        validation set.
      shuffle_buffer_size: the number of example to use in the shuffle buffer
        for tf.data.Dataset.shuffle().
      max_seq_length: maximum sequence length of the tokenized sentences.
      num_parallel_parser_calls: the number of parallel threads to use while
        preprocessing in tf.data.Dataset.map().
      data_dir: optional dir to save TFDS data to. If none then the local
        filesystem is used. Required for using TPUs on Cloud.
    """
    dataset_info = tfds.builder(name).info
    self.validation_fraction = validation_fraction
    self.additional_labels = additional_labels

    self.feature_spec = _make_features_spec(max_seq_length, additional_labels)
    self.split_names = _DATA_SPLIT_NAMES
    self._file_name_patterns = _TF_RECORD_NAME_PATTERNS

    num_train_examples = dataset_info.splits['train'].num_examples
    num_test_examples = dataset_info.splits['test'].num_examples

    if self.validation_fraction > 0:
      num_validation_examples = (
          int(num_train_examples * self.validation_fraction))
      num_train_examples -= num_validation_examples
    else:
      num_validation_examples = dataset_info.splits['validation'].num_examples

    super(_JigsawToxicityDataset, self).__init__(
        name=name,
        num_train_examples=num_train_examples,
        num_validation_examples=num_validation_examples,
        num_test_examples=num_test_examples,
        batch_size=batch_size,
        eval_batch_size=eval_batch_size,
        shuffle_buffer_size=shuffle_buffer_size,
        num_parallel_parser_calls=num_parallel_parser_calls,
        data_dir=data_dir)
    self.info['num_classes'] = 1
    self.info['max_seq_length'] = max_seq_length

  def _read_examples(self, split: base.Split) -> tf.data.Dataset:
    """Creates a dataset to be processed by _create_process_example_fn."""
    if self._data_dir:
      logging.info('Reading from local TFRecords with BERT features %s',
                   self._data_dir)
      return self._read_examples_local(split)
    else:
      logging.info('Reading from TFDS.')
      return self._read_examples_tfds(split)

  def _read_examples_local(self, split: base.Split) -> tf.data.Dataset:
    """Creates a dataset from local TFRecords."""
    is_training = split == base.Split.TRAIN
    return _build_dataset(
        glob_dir=os.path.join(self._data_dir,
                              self._file_name_patterns[split.value]),
        is_training=is_training)

  def _read_examples_tfds(self, split: base.Split) -> tf.data.Dataset:
    """Creates a dataset from TFDS API."""
    if split == base.Split.TRAIN:
      if self.validation_fraction == 0:
        train_split = 'train'
      else:
        train_split = tfds.core.ReadInstruction(
            'train', to=-self._num_validation_examples, unit='abs')
      return tfds.load(
          self.name,
          split=train_split,
          try_gcs=True,
          data_dir=self._data_dir)
    elif split == base.Split.VAL:
      if self.validation_fraction == 0:
        return tfds.load(
            self.name,
            split='validation',
            try_gcs=True,
            data_dir=self._data_dir)
      else:
        val_split = tfds.core.ReadInstruction(
            'train', from_=-self._num_validation_examples, unit='abs')
        return tfds.load(
            self.name,
            split=val_split,
            try_gcs=True,
            data_dir=self._data_dir)
    else:
      return tfds.load(
          self.name,
          split='test',
          try_gcs=True,
          data_dir=self._data_dir)

  def _create_process_example_fn(self, split: base.Split) -> base.PreProcessFn:
    """Create a pre-process function to return labels and sentence tokens."""

    def _example_parser(example: Dict[str, tf.Tensor]) -> Dict[str, Any]:
      """Preprocesses sentences as well as toxicity and other labels."""
      if self._data_dir:
        return tf.io.parse_example(example, self.feature_spec)
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

  def __init__(self, validation_fraction: float = 0.1, **kwargs):
    # Requires validation_fraction > 0 since this dataset doesn't have
    # its own validation set.
    if validation_fraction <= 0:
      raise ValueError('Validation_fraction must be positive.')

    additional_labels = tuple(
        toxicity_name for toxicity_name in _TOXICITY_SUBTYPE_NAMES
        if toxicity_name != 'sexual_explicit')

    super(WikipediaToxicityDataset, self).__init__(
        name='wikipedia_toxicity_subtypes',
        validation_fraction=validation_fraction,
        additional_labels=additional_labels,
        **kwargs)


class CivilCommentsDataset(_JigsawToxicityDataset):
  """Data loader for Civil Comments datasets."""

  def __init__(self, **kwargs):
    super(CivilCommentsDataset, self).__init__(name='civil_comments', **kwargs)


class CivilCommentsIdentitiesDataset(_JigsawToxicityDataset):
  """Data loader for Civil Comments Identities datasets."""

  def __init__(self, **kwargs):
    super(CivilCommentsIdentitiesDataset, self).__init__(
        name='civil_comments/CivilCommentsIdentities',
        additional_labels=_TOXICITY_SUBTYPE_NAMES + _IDENTITY_LABELS,
        **kwargs)

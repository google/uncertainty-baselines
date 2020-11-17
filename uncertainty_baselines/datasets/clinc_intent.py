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
"""Data loader for the Clinc Intent Detection dataset.

Clinc Intent Detection dataset [1] is a text classification task for
intent detection in a task-oriented dialog system. The data covers 150 intent
classes over 10 domains, and contains 150 short (< 30 tokens) spoken queries
for each class. The dataset also contains a large amount of out-of-domain
queries to assess the intent classifier's ability in handling users' request
that is out of the scope of the known services.

Note:
1. The data is split into train-validation-test ratio of roughly 65%:15%:20%,
   and the in-domain (IND) data and the out-of-domain (OOD) data are separated
   into different splits which can be used separately or together.
   Current implementation supports reading only IND data, only OOD data
   or both.

2. The dataset is stored in TFRecord format. It contains both the original
   sentence and intent name, stored in the fields _INTENT_NAME and
   _UTTERANCE_NAME, respectively.
   It also contains the tokenized version of the sentence and the label (stored
   in fields _FEATURE_NAME and _LABEL_NAME) using tf.keras' default tokenizer
   with a vocabulary size of 7292 (which covers all words in the dataset).


## References
[1]: Tony Finch. An Evaluation Dataset for Intent Classification and
     Out-of-Scope Prediction.
     In _Empirical Methods in Natural Language Processing_, 2019.
     https://www.aclweb.org/anthology/D19-1131.pdf
"""
import json
import os.path

from typing import Any, Dict, Tuple

import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds
from uncertainty_baselines.datasets import base

# filenames for in-domain (IND), out-of-domain (OOD) and combined datasets
_FILENAME_TRAIN_IND = 'train.tfrecord'
_FILENAME_VAL_IND = 'val.tfrecord'
_FILENAME_TEST_IND = 'test.tfrecord'

_FILENAME_TRAIN_OOD = 'train_ood.tfrecord'
_FILENAME_VAL_OOD = 'val_ood.tfrecord'
_FILENAME_TEST_OOD = 'test_ood.tfrecord'

_FILENAME_TRAIN_ALL = 'train_all.tfrecord'
_FILENAME_VAL_ALL = 'val_all.tfrecord'
_FILENAME_TEST_ALL = 'test_all.tfrecord'

_FILENAME_TOKENZIER = 'keras_tokenizer.json'

_NUM_TRAIN_IND = 15000
_NUM_TRAIN_OOD = 100
_NUM_VAL_IND = 3000
_NUM_VAL_OOD = 100
_NUM_TEST_IND = 4500
_NUM_TEST_OOD = 1000

_NUM_TRAIN_ALL = _NUM_TRAIN_IND + _NUM_TRAIN_OOD
_NUM_VAL_ALL = _NUM_VAL_IND + _NUM_VAL_OOD
_NUM_TEST_ALL = _NUM_TEST_IND + _NUM_TEST_OOD

_LABEL_NAME = 'intent_label'
_FEATURE_NAME = 'utterance_indices'
_INTENT_NAME = 'intent_name'
_UTTERANCE_NAME = 'utterance'
_NUM_TOKEN_NAME = 'utterance_num_tokens'

_FEATURE_LENGTH = 32  # Maximum number of tokens per sentence


def _build_dataset(glob_dir: str, is_training: bool) -> tf.data.Dataset:
  cycle_len = 10 if is_training else 1
  dataset = tf.data.Dataset.list_files(glob_dir, shuffle=is_training)
  dataset = dataset.interleave(tf.data.TFRecordDataset, cycle_length=cycle_len)
  return dataset


def _make_features_spec() -> Dict[str, tf.io.FixedLenFeature]:
  return {
      _LABEL_NAME: tf.io.FixedLenFeature([], tf.int64),
      _FEATURE_NAME: tf.io.FixedLenFeature([_FEATURE_LENGTH], tf.int64),
      _INTENT_NAME: tf.io.FixedLenFeature([], tf.string),
      _UTTERANCE_NAME: tf.io.FixedLenFeature([], tf.string),
      _NUM_TOKEN_NAME: tf.io.FixedLenFeature([], tf.int64),
  }


def _load_tokenizer(
    tokenizer_dir: str) -> tf.keras.preprocessing.text.Tokenizer:
  with tf.io.gfile.GFile(tokenizer_dir) as tokenizer_file:
    tokenizer_json = json.load(tokenizer_file)
  return tf.keras.preprocessing.text.tokenizer_from_json(tokenizer_json)


def _get_num_examples_and_filenames(
    data_mode: str) -> Tuple[Dict[str, int], Dict[str, str]]:
  """Retrieves the number of examples and filenames according to data mode."""
  if data_mode == 'ind':
    num_examples = {
        'train': _NUM_TRAIN_IND,
        'validation': _NUM_VAL_IND,
        'test': _NUM_TEST_IND
    }
    file_names = {
        'train': _FILENAME_TRAIN_IND,
        'validation': _FILENAME_VAL_IND,
        'test': _FILENAME_TEST_IND
    }
  elif data_mode == 'ood':
    num_examples = {
        'train': _NUM_TRAIN_OOD,
        'validation': _NUM_VAL_OOD,
        'test': _NUM_TEST_OOD
    }
    file_names = {
        'train': _FILENAME_TRAIN_OOD,
        'validation': _FILENAME_VAL_OOD,
        'test': _FILENAME_TEST_OOD
    }
  elif data_mode == 'all':
    num_examples = {
        'train': _NUM_TRAIN_ALL,
        'validation': _NUM_VAL_ALL,
        'test': _NUM_TEST_ALL
    }
    file_names = {
        'train': _FILENAME_TRAIN_ALL,
        'validation': _FILENAME_VAL_ALL,
        'test': _FILENAME_TEST_ALL
    }
  else:
    raise ValueError('"data_mode" can only be one of "ind", "ood" or "all". '
                     'Got "{}".'.format(data_mode))

  return num_examples, file_names


_CITATION = """
Tony Finch. An Evaluation Dataset for Intent Classification and
Out-of-Scope Prediction.
In _Empirical Methods in Natural Language Processing_, 2019.
https://www.aclweb.org/anthology/D19-1131.pdf
"""
_DESCRIPTION = (
    'Clinc Intent Detection dataset [1] is a text classification task for '
    'intent detection in a task-oriented dialog system.')


class _ClincIntentionDatasetBuilder(tfds.core.DatasetBuilder):
  """Minimal TFDS DatasetBuilder for CLINC, does not support downloading."""
  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
  }

  def __init__(self, data_dir, data_mode, **kwargs):
    self._num_examples, self._file_names = _get_num_examples_and_filenames(
        data_mode)
    super(_ClincIntentionDatasetBuilder, self).__init__(
        data_dir=data_dir, **kwargs)
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
    """Constructs a `tf.data.Dataset`."""
    del decoders
    del read_config
    del shuffle_files
    if split == tfds.Split.TRAIN:
      return _build_dataset(
          glob_dir=os.path.join(self._data_dir, self._file_names['train']),
          is_training=True)
    elif split == tfds.Split.VALIDATION:
      return _build_dataset(
          glob_dir=os.path.join(self._data_dir, self._file_names['validation']),
          is_training=False)
    elif split == tfds.Split.TEST:
      return _build_dataset(
          glob_dir=os.path.join(self._data_dir, self._file_names['test']),
          is_training=False)
    raise ValueError('Unsupported split given: {}.'.format(split))

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the `tfds.core.DatasetInfo` object."""
    features = {
        _LABEL_NAME: tfds.features.ClassLabel(num_classes=150),
        _FEATURE_NAME: tfds.features.Tensor(
            shape=[_FEATURE_LENGTH], dtype=tf.int64),
        _INTENT_NAME: tfds.features.Tensor(shape=[], dtype=tf.string),
        _UTTERANCE_NAME: tfds.features.Tensor(shape=[], dtype=tf.string),
        _NUM_TOKEN_NAME: tfds.features.Tensor(shape=[], dtype=tf.int64),
    }
    info = tfds.core.DatasetInfo(
        builder=self,
        description=_DESCRIPTION,
        features=tfds.features.FeaturesDict(features),
        homepage='https://github.com/clinc/oos-eval/blob/master/data/data_full.json',
        citation=_CITATION,
        # Note that while metadata seems to be the most appropriate way to store
        # arbitrary info, it will not be printed when printing out the dataset
        # info.
        metadata=tfds.core.MetadataDict(feature_size=_FEATURE_LENGTH))
    split_dict = tfds.core.SplitDict('clinc_intent')
    # Instead of having a single element shard_lengths, we should really have a
    # list of the number of elements in each file shard in each split.
    split_dict.add(tfds.core.SplitInfo(
        name=tfds.Split.TRAIN,
        shard_lengths=[self._num_examples['train']]))
    split_dict.add(tfds.core.SplitInfo(
        name=tfds.Split.VALIDATION,
        shard_lengths=[self._num_examples['validation']]))
    split_dict.add(tfds.core.SplitInfo(
        name=tfds.Split.TEST,
        shard_lengths=[self._num_examples['test']]))
    info.update_splits_if_different(split_dict)
    return info


class ClincIntentDetectionDataset(base.BaseDataset):
  """Clinc Intent Detection dataset builder class."""

  def __init__(
      self,
      split: str,
      shuffle_buffer_size: int = None,
      num_parallel_parser_calls: int = 64,
      data_mode: str = 'ind',
      data_dir: str = None,
      download_data: bool = False,
      **unused_kwargs: Dict[str, Any]):
    """Create a CLINC tf.data.Dataset builder.

    Args:
      split: a dataset split, either a custom tfds.Split or one of the
        tfds.Split enums [TRAIN, VALIDAITON, TEST] or their lowercase string
        names.
      shuffle_buffer_size: the number of example to use in the shuffle buffer
        for tf.data.Dataset.shuffle().
      num_parallel_parser_calls: the number of parallel threads to use while
        preprocessing in tf.data.Dataset.map().
      data_mode: One of ['ind', 'ood', 'all'] to decide whether to read only
        in-domain data, or read only out-of-domain data, or read both.
      data_dir: path to a directory containing the CLINC datasets, with
        filenames train-*-of-*', 'validate.tfr', 'test.tfr'.
      download_data: Whether or not to download data before loading. Currently
        unsupported.
    """
    self.tokenizer = _load_tokenizer(
        tokenizer_dir=os.path.join(data_dir, _FILENAME_TOKENZIER))

    super(ClincIntentDetectionDataset, self).__init__(
        name='clinc_intent',
        dataset_builder=_ClincIntentionDatasetBuilder(data_dir, data_mode),
        split=split,
        shuffle_buffer_size=shuffle_buffer_size,
        num_parallel_parser_calls=num_parallel_parser_calls,
        download_data=False)

  def _create_process_example_fn(self) -> base.PreProcessFn:

    def _example_parser(example: tf.Tensor) -> Dict[str, tf.Tensor]:
      """Parse features and labels from a serialized tf.train.Example."""
      features_spec = _make_features_spec()
      features = tf.io.parse_example(example, features_spec)
      labels = tf.cast(features.pop(_LABEL_NAME), tf.int32)
      utterance_indices = features[_FEATURE_NAME]
      num_tokens = features[_NUM_TOKEN_NAME]
      return {
          'features': utterance_indices,
          'labels': labels,
          'num_tokens': num_tokens,
      }

    return _example_parser

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

from typing import Any, Dict, Optional, Tuple

import tensorflow.compat.v2 as tf
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
        'val': _NUM_VAL_IND,
        'test': _NUM_TEST_IND
    }
    file_names = {
        'train': _FILENAME_TRAIN_IND,
        'val': _FILENAME_VAL_IND,
        'test': _FILENAME_TEST_IND
    }
  elif data_mode == 'ood':
    num_examples = {
        'train': _NUM_TRAIN_OOD,
        'val': _NUM_VAL_OOD,
        'test': _NUM_TEST_OOD
    }
    file_names = {
        'train': _FILENAME_TRAIN_OOD,
        'val': _FILENAME_VAL_OOD,
        'test': _FILENAME_TEST_OOD
    }
  elif data_mode == 'all':
    num_examples = {
        'train': _NUM_TRAIN_ALL,
        'val': _NUM_VAL_ALL,
        'test': _NUM_TEST_ALL
    }
    file_names = {
        'train': _FILENAME_TRAIN_ALL,
        'val': _FILENAME_VAL_ALL,
        'test': _FILENAME_TEST_ALL
    }
  else:
    raise ValueError('"data_mode" can only be one of "ind", "ood" or "all". '
                     'Got "{}".'.format(data_mode))

  return num_examples, file_names


class ClincIntentDetectionDataset(base.BaseDataset):
  """Clinc Intent Detection dataset builder class."""

  def __init__(self,
               batch_size: int,
               eval_batch_size: int,
               shuffle_buffer_size: int = None,
               num_parallel_parser_calls: int = 64,
               data_dir: Optional[str] = None,
               data_mode: str = 'ind',
               **unused_kwargs: Dict[str, Any]):
    """Initializer.

    Args:
      batch_size: the training batch size.
      eval_batch_size: the validation and test batch size.
      shuffle_buffer_size: the number of example to use in the shuffle buffer
        for tf.data.Dataset.shuffle().
      num_parallel_parser_calls: the number of parallel threads to use while
        preprocessing in tf.data.Dataset.map().
      data_dir: directory containing the TFRecord datasets and the tokenizer.
      data_mode: One of ['ind', 'ood', 'all'] to decide whether to read only
        in-domain data, or read only out-of-domain data, or read both.
    """
    num_examples, self._file_names = _get_num_examples_and_filenames(data_mode)
    self.tokenizer = _load_tokenizer(
        tokenizer_dir=os.path.join(data_dir, _FILENAME_TOKENZIER))

    super(ClincIntentDetectionDataset, self).__init__(
        name='clinc_intent',
        num_train_examples=num_examples['train'],
        num_validation_examples=num_examples['val'],
        num_test_examples=num_examples['test'],
        batch_size=batch_size,
        eval_batch_size=eval_batch_size,
        shuffle_buffer_size=shuffle_buffer_size,
        num_parallel_parser_calls=num_parallel_parser_calls,
        data_dir=data_dir)

    self.info['num_classes'] = 150
    self.info['feature_size'] = _FEATURE_LENGTH

  def _read_examples(self, split: base.Split) -> tf.data.Dataset:
    if split == base.Split.TRAIN:
      return _build_dataset(
          glob_dir=os.path.join(self._data_dir, self._file_names['train']),
          is_training=True)
    if split == base.Split.VAL:
      return _build_dataset(
          glob_dir=os.path.join(self._data_dir, self._file_names['val']),
          is_training=False)
    if split == base.Split.TEST:
      return _build_dataset(
          glob_dir=os.path.join(self._data_dir, self._file_names['test']),
          is_training=False)

  def _create_process_example_fn(self, split: base.Split) -> base.PreProcessFn:

    def _example_parser(example: tf.train.Example) -> Dict[str, tf.Tensor]:
      """Parse features and labels from a serialized tf.train.Example."""
      features_spec = _make_features_spec()
      features = tf.io.parse_example(example, features_spec)
      labels = tf.cast(features.pop(_LABEL_NAME), tf.int32)
      utterance_indices = features[_FEATURE_NAME]
      num_tokens = features[_NUM_TOKEN_NAME]
      return {'features': utterance_indices, 'labels': labels,
              'num_tokens': num_tokens}

    return _example_parser

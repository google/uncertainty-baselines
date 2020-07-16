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
"""Data loader for the Jigsaw Toxic Comment classification dataset.

A dataset for training toxicity detection systems. It contains comments from
Wikipedia’s talk page edits that contains detecting different types of
toxicity like threats, obscenity, insults, and identity-based hate.

The sentences in this dataset are pre-processed using a standard WordPiece
tokenizer using the standard BERT vocabulary. The original testing dataset
contains records without valid labels, which are not included in this version.

DISCLAIMER: This dataset contains text that may be considered profane, vulgar,
or offensive.

## References

[1]: Betty van Aken, Julian Risch, Ralf Krestel, Alexander Löser.
     Challenges for Toxic Comment Classification: An In-Depth Error Analysis.
     In _Proceedings of the 2nd Workshop on Abusive Language Online_, 2018.
     https://www.aclweb.org/anthology/W18-5105/
"""
import os.path

from typing import Dict

import tensorflow.compat.v2 as tf
from uncertainty_baselines.datasets import base

_FILENAME_TRAIN = 'train.tfrecord'
_FILENAME_TEST = 'test.tfrecord'

_FILENAME_TOKENZIER = 'keras_tokenizer.json'

_NUM_TRAIN = 159571
_NUM_TEST = 63978

_LABEL_BINARY_NAME = 'label_binary_ids'
_LABEL_MULTI_NAME = 'label_ids'

_SENTENCE_ID_NAME = 'input_ids'
_SENTENCE_TOKEN_NAME = 'input_tokens'
_INPUT_MASK_NAME = 'input_mask'
_SEGMENT_ID_NAME = 'segment_ids'

_FEATURE_LENGTH = 512  # Maximum number of tokens per sentence.
_LABEL_LENGTH = 6  # Number of Toxicity categories.


def _build_dataset(glob_dir: str, is_training: bool) -> tf.data.Dataset:
  cycle_len = 10 if is_training else 1
  dataset = tf.data.Dataset.list_files(glob_dir, shuffle=is_training)
  dataset = dataset.interleave(tf.data.TFRecordDataset, cycle_length=cycle_len)
  return dataset


def _make_features_spec() -> Dict[str, tf.io.FixedLenFeature]:
  return {
      _LABEL_BINARY_NAME: tf.io.FixedLenFeature([], tf.int64),
      _LABEL_MULTI_NAME: tf.io.FixedLenFeature([_LABEL_LENGTH], tf.int64),
      _SENTENCE_ID_NAME: tf.io.FixedLenFeature([_FEATURE_LENGTH], tf.int64),
      _SENTENCE_TOKEN_NAME: tf.io.FixedLenFeature([], tf.string),
      _INPUT_MASK_NAME: tf.io.FixedLenFeature([_FEATURE_LENGTH], tf.int64),
      _SEGMENT_ID_NAME: tf.io.FixedLenFeature([_FEATURE_LENGTH], tf.int64),
  }


class JigsawToxicCommentDataset(base.BaseDataset):
  """Clinc Intent Detection dataset builder class."""

  def __init__(self,
               batch_size: int,
               eval_batch_size: int,
               validation_percent: float = 0.05,
               shuffle_buffer_size: int = None,
               num_parallel_parser_calls: int = 64,
               dataset_dir: str = None):
    """Initializer.

    Args:
      batch_size: the training batch size.
      eval_batch_size: the validation and test batch size.
      validation_percent: the percent of the training set to use as a validation
        set.
      shuffle_buffer_size: the number of example to use in the shuffle buffer
        for tf.data.Dataset.shuffle().
      num_parallel_parser_calls: the number of parallel threads to use while
        preprocessing in tf.data.Dataset.map().
      dataset_dir: directory containing the TFRecord datasets and the tokenizer.
    """
    self._dataset_dir = dataset_dir
    num_validation_examples = int(_NUM_TRAIN * validation_percent)
    num_training_examples = _NUM_TRAIN - num_validation_examples


    super(JigsawToxicCommentDataset, self).__init__(
        name='jigsaw_toxic_comment',
        num_train_examples=num_training_examples,
        num_validation_examples=num_validation_examples,
        num_test_examples=_NUM_TEST,
        batch_size=batch_size,
        eval_batch_size=eval_batch_size,
        shuffle_buffer_size=shuffle_buffer_size,
        num_parallel_parser_calls=num_parallel_parser_calls)

    self.info['num_classes'] = 5  # Number of toxicity categories.
    self.info['feature_size'] = _FEATURE_LENGTH

  def _read_examples(self, split: base.Split) -> tf.data.Dataset:
    if split == base.Split.TEST:
      return _build_dataset(
          glob_dir=os.path.join(self._dataset_dir, _FILENAME_TEST),
          is_training=False)

    if split == base.Split.VAL:
      if self._num_validation_examples == 0:
        raise ValueError(
            'No validation set provided. Set `validation_percent > 0.0` to '
            'take a subset of the training set as validation.')

      # take the first _num_validation_examples of training data as validation.
      dataset = _build_dataset(
          glob_dir=os.path.join(self._dataset_dir, _FILENAME_TRAIN),
          is_training=False)
      return dataset.take(self._num_validation_examples)

    if split == base.Split.TRAIN:
      dataset = _build_dataset(
          glob_dir=os.path.join(self._dataset_dir, _FILENAME_TRAIN),
          is_training=True)
      if self._num_validation_examples > 0:
        return dataset.skip(self._num_validation_examples)

      return dataset

  def _create_process_example_fn(self, split: base.Split) -> base.PreProcessFn:

    def _example_parser(example: tf.train.Example) -> Dict[str, tf.Tensor]:
      """Parse features and labels from a serialized tf.train.Example."""
      features_spec = _make_features_spec()
      features = tf.io.parse_example(example, features_spec)
      labels = tf.cast(features.pop(_LABEL_MULTI_NAME), tf.int32)
      labels_binary = tf.cast(features.pop(_LABEL_BINARY_NAME), tf.int32)

      return {
          'features': features[_SENTENCE_ID_NAME],
          'input_mask': features[_INPUT_MASK_NAME],
          'segment_ids': features[_SEGMENT_ID_NAME],
          'labels': labels,
          'labels_binary': labels_binary,
      }

    return _example_parser

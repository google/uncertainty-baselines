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

"""Genomics OOD dataset builder."""

import os
from typing import Any, Dict

import tensorflow.compat.v2 as tf
from uncertainty_baselines.datasets import base

# Training data contains 10 bacteria classes and 100,000 examples per class
_NUM_TRAIN = 100000 * 10
# Val and test data contain 10 bacteria classes and 10,000 examples per class
_NUM_VAL = 10000 * 10
_NUM_TEST = 10000 * 10

# tfrecord data filenames
_TRAIN_FILEPATTERN = 'genomics_ood-train.tfrecord*'
_VAL_FILEPATTERN = 'genomics_ood-validation.tfrecord*'
_TEST_FILEPATTERN = 'genomics_ood-test.tfrecord*'
_VAL_OOD_FILEPATTERN = 'genomics_ood-validation_ood.tfrecord*'
_TEST_OOD_FILEPATTERN = 'genomics_ood-test_ood.tfrecord*'


def tfrecord_filepattern(split, data_mode):
  """Filenames of different subtypes of data."""
  if split == 'train' and data_mode == 'ind':
    return _TRAIN_FILEPATTERN
  elif split == 'validation' and data_mode == 'ind':
    return _VAL_FILEPATTERN
  elif split == 'test' and data_mode == 'ind':
    return _TEST_FILEPATTERN
  elif split == 'validation' and data_mode == 'ood':
    return _VAL_OOD_FILEPATTERN
  elif split == 'test' and data_mode == 'ood':
    return _TEST_OOD_FILEPATTERN
  else:
    raise ValueError(
        'No such a combination of split:{} and data_mode:{}'.format(
            split, data_mode))


class GenomicsOodDataset(base.BaseDataset):
  """Genomics OOD dataset builder class."""

  def __init__(self,
               batch_size: int,
               eval_batch_size: int,
               shuffle_buffer_size: int = None,
               num_parallel_parser_calls: int = 64,
               eval_filter_class_id: int = -1,
               data_mode: str = 'ind',
               data_dir: str = None,
               **unused_kwargs: Dict[str, Any]):
    """Create an Genomics OOD tf.data.Dataset builder.

    Args:
      batch_size: the training batch size.
      eval_batch_size: the validation and test batch size.
      shuffle_buffer_size: the number of example to use in the shuffle buffer
        for tf.data.Dataset.shuffle().
      num_parallel_parser_calls: the number of parallel threads to use while
        preprocessing in tf.data.Dataset.map().
      eval_filter_class_id: evalulate inputs from a particular class only.
      data_mode: either 'ind' or 'ood' to decide whether to read in-distribution
        data or out-of-domain data.
      data_dir: data directory.
    """
    self._data_mode = data_mode


    super(GenomicsOodDataset, self).__init__(
        name='genomics_ood',
        num_train_examples=_NUM_TRAIN,
        num_validation_examples=_NUM_VAL,
        num_test_examples=_NUM_TEST,
        batch_size=batch_size,
        eval_batch_size=eval_batch_size,
        shuffle_buffer_size=shuffle_buffer_size,
        num_parallel_parser_calls=num_parallel_parser_calls,
        data_dir=data_dir)

  def _read_examples(self, split: base.Split) -> tf.data.Dataset:
    """We use the original 'validation' set as test."""
    # TODO(jjren) change to tfds.load once tfds dataset is available
    file_pattern = tfrecord_filepattern(split.value, self._data_mode)
    file_list = tf.io.gfile.glob(os.path.join(self._data_dir, file_pattern))

    def parse_single_tfexample(_, serialized_example):
      # Read data from serialized examples.
      # seq: the input DNA sequence composed by {A, C, G, T}.
      # label_id: the predictive target, i.e., the index of bacteria class.
      # label_name: the name of the bacteria class corresponding to label_id
      # seq_info: the source of the DNA sequence, i.e., the genome name,
      # NCBI accession number, and the position where it was sampled from.
      # domain: if the bacteria is in-distribution (in), or OOD (ood)
      features = tf.io.parse_single_example(
          serialized_example,
          features={
              'seq': tf.io.FixedLenFeature([], tf.string),
              'label': tf.io.FixedLenFeature([], tf.int64),
              'seq_info': tf.io.VarLenFeature(tf.string),
              'domain': tf.io.VarLenFeature(tf.string),
          })
      return features

    dataset = tf.data.TFRecordDataset(file_list).map(
        lambda v: parse_single_tfexample(v, v))
    return dataset

  def _create_process_example_fn(self, split: base.Split) -> base.PreProcessFn:
    del split

    def _example_parser(example: Dict[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
      """A pre-process function to return images in [0, 1]."""

      def encode_seq(seq):
        """Encode DNA sequence into integers."""
        # Convert a input DNA sequence of type string into a list of integers
        # by replacing {A, C, G, T} with {0, 1, 2, 3}.
        # eg, 'CAGTA' (input) --> '10230' --> [1,0,2,3,0] (output)
        seq = tf.strings.regex_replace(seq, 'A', '0')
        seq = tf.strings.regex_replace(seq, 'C', '1')
        seq = tf.strings.regex_replace(seq, 'G', '2')
        seq = tf.strings.regex_replace(seq, 'T', '3')
        seq_list = tf.strings.bytes_split(seq)
        seq = tf.strings.to_number(seq_list, out_type=tf.int32)
        return seq

      seq = encode_seq(example['seq'])
      label = tf.cast(example['label'], tf.int32)
      parsed_example = {
          'features': seq,
          'labels': label,
      }
      return parsed_example

    return _example_parser

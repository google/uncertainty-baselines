# coding=utf-8
# Copyright 2021 The Uncertainty Baselines Authors.
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
from typing import Any, Dict, Optional

import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds
from uncertainty_baselines.datasets import base

# Training data contains 10 bacteria classes and 100,000 examples per class
_NUM_TRAIN = 100000 * 10
# Val and test data contain 10 bacteria classes and 10,000 examples per class
_NUM_VAL = 10000 * 10
_NUM_TEST = 10000 * 10

# TFRecord data filenames.
_TRAIN_FILEPATTERN = 'genomics_ood-train.tfrecord*'
_VAL_FILEPATTERN = 'genomics_ood-validation.tfrecord*'
_TEST_FILEPATTERN = 'genomics_ood-test.tfrecord*'
_VAL_OOD_FILEPATTERN = 'genomics_ood-validation_ood.tfrecord*'
_TEST_OOD_FILEPATTERN = 'genomics_ood-test_ood.tfrecord*'


def _tfrecord_filepattern(split, data_mode):
  """Filenames of different subtypes of data."""
  if split == tfds.Split.TRAIN and data_mode == 'ind':
    return _TRAIN_FILEPATTERN
  elif split == tfds.Split.VALIDATION and data_mode == 'ind':
    return _VAL_FILEPATTERN
  elif split == tfds.Split.TEST and data_mode == 'ind':
    return _TEST_FILEPATTERN
  elif split == tfds.Split.VALIDATION and data_mode == 'ood':
    return _VAL_OOD_FILEPATTERN
  elif split == tfds.Split.TEST and data_mode == 'ood':
    return _TEST_OOD_FILEPATTERN
  else:
    raise ValueError(
        'No such a combination of split={} and data_mode={}'.format(
            split, data_mode))


class _GenomicsOodDatasetBuilder(tfds.core.DatasetBuilder):
  """Minimal TFDS DatasetBuilder for the Genomics OOD dataset."""
  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
  }

  def __init__(self, data_dir, data_mode, **kwargs):
    super(_GenomicsOodDatasetBuilder, self).__init__(
        data_dir=data_dir, **kwargs)
    # We have to override self._data_dir to prevent the parent class from
    # appending the class name and version.
    self._data_dir = data_dir
    self._data_mode = data_mode

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
    file_pattern = _tfrecord_filepattern(split, self._data_mode)
    file_list = tf.io.gfile.glob(os.path.join(self._data_dir, file_pattern))
    dataset = tf.data.TFRecordDataset(file_list)
    return dataset

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the `tfds.core.DatasetInfo` object."""
    features = {
        'seq': tfds.features.Tensor(shape=[], dtype=tf.string),
        'label': tfds.features.ClassLabel(num_classes=10),
        'seq_info': tfds.features.Tensor(shape=(None,), dtype=tf.string),
        'domain': tfds.features.Tensor(shape=(None,), dtype=tf.string),
    }
    info = tfds.core.DatasetInfo(
        builder=self,
        description='Genomics OOD dataset.',
        features=tfds.features.FeaturesDict(features),
        # Note that while metadata seems to be the most appropriate way to store
        # arbitrary info, it will not be printed when printing out the dataset
        # info.
        metadata=tfds.core.MetadataDict())
    # Instead of having a single element shard_lengths, we should really have a
    # list of the number of elements in each file shard in each split.
    split_infos = [
        tfds.core.SplitInfo(
            name=tfds.Split.VALIDATION,
            shard_lengths=[_NUM_VAL],
            num_bytes=0,
        ),
        tfds.core.SplitInfo(
            name=tfds.Split.TEST,
            shard_lengths=[_NUM_TEST],
            num_bytes=0,
        ),
        tfds.core.SplitInfo(
            name=tfds.Split.TRAIN,
            shard_lengths=[_NUM_TRAIN],
            num_bytes=0,
        ),
    ]
    split_dict = tfds.core.SplitDict(
        split_infos, dataset_name='__genomics_ood_dataset_builder')
    info.set_splits(split_dict)
    return info


class GenomicsOodDataset(base.BaseDataset):
  """Genomics OOD dataset builder class."""

  def __init__(
      self,
      split: str,
      shuffle_buffer_size: Optional[int] = None,
      num_parallel_parser_calls: int = 64,
      eval_filter_class_id: int = -1,
      data_mode: str = 'ind',
      data_dir: Optional[str] = None,
      is_training: Optional[bool] = None,
      **unused_kwargs: Dict[str, Any]):
    """Create an Genomics OOD tf.data.Dataset builder.

    Args:
      split: a dataset split, either a custom tfds.Split or one of the
        tfds.Split enums [TRAIN, VALIDAITON, TEST] or their lowercase string
        names.
      shuffle_buffer_size: the number of example to use in the shuffle buffer
        for tf.data.Dataset.shuffle().
      num_parallel_parser_calls: the number of parallel threads to use while
        preprocessing in tf.data.Dataset.map().
      eval_filter_class_id: evalulate inputs from a particular class only.
      data_mode: either 'ind' or 'ood' to decide whether to read in-distribution
        data or out-of-domain data.
      data_dir: path to a directory containing the Genomics OOD dataset, with
        filenames train-*-of-*', 'validate.tfr', 'test.tfr'.
      is_training: Whether or not the given `split` is the training split. Only
        required when the passed split is not one of ['train', 'validation',
        'test', tfds.Split.TRAIN, tfds.Split.VALIDATION, tfds.Split.TEST].
    """
    if data_dir is None:
      builder = tfds.builder('genomics_ood')
      data_dir = builder.data_dir

    super(GenomicsOodDataset, self).__init__(
        name='genomics_ood',
        dataset_builder=_GenomicsOodDatasetBuilder(data_dir, data_mode),
        split=split,
        is_training=is_training,
        shuffle_buffer_size=shuffle_buffer_size,
        num_parallel_parser_calls=num_parallel_parser_calls,
        download_data=False)

  def _create_process_example_fn(self) -> base.PreProcessFn:

    def _example_parser(example: Dict[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
      """A pre-process function to return ACGT strings."""
      # Read data from serialized examples.
      # seq: the input DNA sequence composed by {A, C, G, T}.
      # label_id: the predictive target, i.e., the index of bacteria class.
      # label_name: the name of the bacteria class corresponding to label_id
      # seq_info: the source of the DNA sequence, i.e., the genome name,
      # NCBI accession number, and the position where it was sampled from.
      # domain: if the bacteria is in-distribution (in), or OOD (ood)
      features = tf.io.parse_single_example(
          example['features'],
          features={
              'seq': tf.io.FixedLenFeature([], tf.string),
              'label': tf.io.FixedLenFeature([], tf.int64),
              'seq_info': tf.io.VarLenFeature(tf.string),
              'domain': tf.io.VarLenFeature(tf.string),
          })

      # Convert a input DNA sequence of type string into a list of integers
      # by replacing {A, C, G, T} with {0, 1, 2, 3}.
      # eg, 'CAGTA' (input) --> '10230' --> [1,0,2,3,0] (output)
      seq = tf.strings.regex_replace(features['seq'], 'A', '0')
      seq = tf.strings.regex_replace(seq, 'C', '1')
      seq = tf.strings.regex_replace(seq, 'G', '2')
      seq = tf.strings.regex_replace(seq, 'T', '3')
      seq_list = tf.strings.bytes_split(seq)
      seq = tf.strings.to_number(seq_list, out_type=tf.int32)
      return {
          'features': seq,
          'labels': tf.cast(features['label'], tf.int32),
      }

    return _example_parser

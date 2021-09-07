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

"""SMCalflow dataset builder.

 The SMCalFlow dataset is from the following paper:
   Task-Oriented Dialogue as Dataflow Synthesis (Andreas et al., 2020)

 The MultiWoz 2.1 dataset is the released version from the following paper:
   Task-Oriented Dialogue as Dataflow Synthesis (Andreas et al., 2020)

 The dataset is originally published at:
   MultiWOZ 2.1: A Consolidated Multi-Domain Dialogue Dataset with State
   Corrections and State Tracking Baselines (Eric et al., 2019)
 The released version is processed by:
   Transferable Multi-Domain State Generator for Task-Oriented Dialogue Systems
   (Wu et al., 2019)

 Processed following the directions in:
   https://github.com/microsoft/task_oriented_dialogue_as_dataflow_synthesis

"""

import os.path

from typing import Any, Dict, Optional, Type

import seqio
import t5.data

import tensorflow as tf
import tensorflow_datasets as tfds
from uncertainty_baselines.datasets import base


_NUM_TRAIN_SMCALFLOW = 121200
_NUM_VAL_SMCALFLOW = 13499

_NUM_TRAIN_MULTIWOZ = 56668
_NUM_VAL_MULTIWOZ = 7374
_NUM_TEST_MULTIWOZ = 7368

_FEATURES = [
    'encoder_input_tokens', 'decoder_target_tokens', 'decoder_input_tokens',
    'encoder_segment_ids', 'decoder_segment_ids'
]


def _get_num_examples(name: str) -> Dict[str, int]:
  """Retrieves the number of examples and filenames according to task name."""
  if name == 'smcalflow':
    num_examples = {
        tfds.Split.TRAIN: _NUM_TRAIN_SMCALFLOW,
        tfds.Split.VALIDATION: _NUM_VAL_SMCALFLOW,
    }
  elif name == 'multiwoz':
    num_examples = {
        tfds.Split.TRAIN: _NUM_TRAIN_MULTIWOZ,
        tfds.Split.VALIDATION: _NUM_VAL_MULTIWOZ,
        tfds.Split.TEST: _NUM_TEST_MULTIWOZ,
    }
  else:
    raise ValueError('"name" can only be one of "smcalflow" or "multiwoz". '
                     'Got "{}".'.format(name))

  return num_examples


def _has_test_split(name: str) -> bool:
  return name == 'multiwoz'


class _SMCalflowDatasetBuilder(tfds.core.DatasetBuilder):
  """Minimal TFDS DatasetBuilder for SMCalflow and MultiWoZ, does not support downloading."""
  VERSION = tfds.core.Version('0.0.0')

  def __init__(self, name: str, data_dir: str, max_seq_length: int,
               vocabulary: seqio.Vocabulary,
               feature_converter_cls: Type[seqio.FeatureConverter],
               **unused_kwargs: Dict[str, Any]):
    self._max_seq_length = max_seq_length
    self._task = self._build_task(name, data_dir, vocabulary)
    self._feature_converter = feature_converter_cls()

    super().__init__(
        data_dir=data_dir, **unused_kwargs)
    # We have to override self._data_dir to prevent the parent class from
    # appending the class name and version.
    self._data_dir = data_dir

  def _download_and_prepare(self, dl_manager, download_config=None):
    """Downloads and prepares dataset for reading."""
    raise NotImplementedError

  def _build_task(self, task_name: str, data_dir: str,
                  vocabulary: seqio.Vocabulary) -> seqio.Task:
    split_to_filepattern = {
        tfds.Split.TRAIN: os.path.join(data_dir, 'train.tfr*'),
        tfds.Split.VALIDATION: os.path.join(data_dir, 'valid.tfr*')
    }
    if _has_test_split(task_name):
      split_to_filepattern[tfds.Split.TEST] = os.path.join(
          data_dir, 'test.tfr*')

    source_features = {
        'inputs': tf.io.FixedLenFeature([], tf.string, ''),
        'targets': tf.io.FixedLenFeature([], tf.string, '')
    }
    data_source = seqio.TFExampleDataSource(
        split_to_filepattern=split_to_filepattern,
        feature_description=source_features,
        num_input_examples=_get_num_examples(task_name))

    output_features = {
        'inputs':
            seqio.Feature(vocabulary=vocabulary, add_eos=True, required=False),
        'targets':
            seqio.Feature(vocabulary=vocabulary, add_eos=True)
    }
    task = seqio.Task(
        name=task_name,
        source=data_source,
        output_features=output_features,
        preprocessors=[
            seqio.preprocessors.tokenize, seqio.preprocessors.append_eos
        ],
        shuffle_buffer_size=None  # disable shuffling.
    )
    return task

  def _as_dataset(self,
                  split: tfds.Split,
                  decoders=None,
                  read_config=None,
                  shuffle_files=False) -> tf.data.Dataset:
    """Constructs a `tf.data.Dataset`."""
    del decoders
    del read_config
    del shuffle_files
    task_feature_lengths = {
        'inputs': self._max_seq_length,
        'targets': self._max_seq_length
    }

    dataset = self._task.get_dataset(
        sequence_length=task_feature_lengths, split=split, shuffle=False)
    return self._feature_converter(dataset, task_feature_lengths)

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the `tfds.core.DatasetInfo` object."""
    features = {
        feature_name:
        tfds.features.Tensor(shape=[self._max_seq_length], dtype=tf.int32)
        for feature_name in _FEATURES
    }
    info = tfds.core.DatasetInfo(
        builder=self,
        description=self._task.name,
        features=tfds.features.FeaturesDict(features),
        metadata=None)
    # Instead of having a single element shard_lengths, we should really have a
    # list of the number of elements in each file shard in each split.
    splits = [tfds.Split.TRAIN, tfds.Split.VALIDATION]
    if _has_test_split(self._task.name):
      splits.append(tfds.Split.TEST)
    split_infos = []
    for split in splits:
      split_infos.append(
          tfds.core.SplitInfo(
              name=split,
              shard_lengths=[self._task.num_input_examples(split)],
              num_bytes=0,
          ))
    split_dict = tfds.core.SplitDict(split_infos, dataset_name=self.name)
    info.set_splits(split_dict)
    return info


class _SMCalflowDataset(base.BaseDataset):
  """SMCalflow dataset builder class."""

  def __init__(
      self,
      name: str,
      split: str,
      data_dir: Optional[str] = None,
      max_seq_length: int = 512,
      vocabulary: Optional[seqio.Vocabulary] = t5.data.get_default_vocabulary(),
      feature_converter_cls: Optional[Type[
          seqio.FeatureConverter]] = seqio.EncDecFeatureConverter,
      is_training: Optional[bool] = None,
      num_parallel_parser_calls: int = 64,
      shuffle_buffer_size: Optional[int] = None):
    """Create a SMCalflow tf.data.Dataset builder.

    Args:
      name: the name of this dataset.
      split: a dataset split, either a custom tfds.Split or one of the
        tfds.Split enums [TRAIN, VALIDAITON, TEST] or their lowercase string
        names.
      data_dir: path to a directory containing the Criteo datasets, with
        filenames train-*-of-*', 'validate.tfr', 'test.tfr'.
      max_seq_length: the maximum sequence length for the input and target of an
        example.
      vocabulary: the vocabulary used for tokenization. Must be a subclass of
        seqio.Vocabulary.
      feature_converter_cls: the type of the feature converter converting
        examples of {'input', 'target'} into model specific outputs. Must be a
        subclass of seqio.FeatureConverter.
      is_training: Whether or not the given `split` is the training split. Only
        required when the passed split is not one of ['train', 'validation',
        'test', tfds.Split.TRAIN, tfds.Split.VALIDATION, tfds.Split.TEST].
      num_parallel_parser_calls: the number of parallel threads to use while
        preprocessing in tf.data.Dataset.map().
      shuffle_buffer_size: the number of example to use in the shuffle buffer
        for tf.data.Dataset.shuffle().
    """
    super().__init__(
        name=name,
        dataset_builder=_SMCalflowDatasetBuilder(
            name=name,
            data_dir=data_dir,
            max_seq_length=max_seq_length,
            vocabulary=vocabulary,
            feature_converter_cls=feature_converter_cls),
        split=split,
        is_training=is_training,
        shuffle_buffer_size=shuffle_buffer_size,
        num_parallel_parser_calls=num_parallel_parser_calls)

  def _create_process_example_fn(self) -> base.PreProcessFn:
    """Create a pre-process function to return labels and sentence tokens."""

    def _example_parser(example: Dict[str, tf.Tensor]) -> Dict[str, Any]:
      """Parse sentences and labels from a serialized tf.train.Example."""
      return {feature: example[feature] for feature in _FEATURES}

    return _example_parser


class SMCalflowDataset(_SMCalflowDataset):
  """SMCalflow dataset builder class."""

  def __init__(self, data_dir: Optional[str] = None, **kwargs: Dict[str, Any]):
    super().__init__(
        name='smcalflow', data_dir=data_dir, **kwargs)


class MultiWoZDataset(_SMCalflowDataset):
  """MultiWoZ dataset builder class."""

  def __init__(self, data_dir: Optional[str] = None, **kwargs: Dict[str, Any]):
    super().__init__(
        name='multiwoz', data_dir=data_dir, **kwargs)

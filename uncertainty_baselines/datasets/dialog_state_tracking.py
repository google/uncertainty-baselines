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

"""Data loader for goal-oriented dialog state tracking datasets.

Dialog state tracking is a sequence prediction task that predicts the dialog
state label of each conversational turn in a given dialog. Currently, the
following datasets are supported.

   * Synthetic Task-oriented Dialog with Controllable Complexity (SimDial) [1]

## References
[1]: Zhao, Tiancheng and Eskenazi, Maxine. Zero-Shot Dialog Generation with
     Cross-Domain Latent Actions.
     In _Meeting of the Special Interest Group on Discourse and Dialogue_, 2018.
     https://www.aclweb.org/anthology/W18-5001/
"""

import json
import os

from typing import Dict, Tuple, Optional, Any
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds
from uncertainty_baselines.datasets import base


MAX_UTT_LEN = 40
MAX_DIALOG_LEN = 13

VOCAB_SIZE_UTT = 460
VOCAB_SIZE_LABEL = 48

USR_UTT_NAME = 'usr_utt'
SYS_UTT_NAME = 'sys_utt'
STATE_LABEL_NAME = 'label'
DIAL_LEN_NAME = 'dialog_len'

NUM_TRAIN = 9600
NUM_TEST = 2400

FILENAME_META = 'meta.json'
FILENAME_TOKENIZER = 'id_to_vocab.json'
FILENAME_TOKENIZER_LABEL = 'id_to_vocab_label.json'

FILENAME_TRAIN = 'train.tfrecord'
FILENAME_TEST = 'test.tfrecord'

# Use test as stand-in for val. In practice we never use this dataset.
NUM_VAL = NUM_TEST
FILENAME_VALID = FILENAME_TEST


def _build_dataset(glob_dir: str, is_training: bool) -> tf.data.Dataset:
  cycle_len = 10 if is_training else 1
  dataset = tf.data.Dataset.list_files(glob_dir, shuffle=is_training)
  dataset = dataset.interleave(tf.data.TFRecordDataset, cycle_length=cycle_len)
  return dataset


def _make_features_spec() -> Dict[str, tf.io.FixedLenFeature]:
  return {
      USR_UTT_NAME: tf.io.FixedLenFeature([], tf.string, default_value=''),
      SYS_UTT_NAME: tf.io.FixedLenFeature([], tf.string, default_value=''),
      STATE_LABEL_NAME: tf.io.FixedLenFeature([], tf.string, default_value=''),
      DIAL_LEN_NAME: tf.io.FixedLenFeature([], tf.int64, default_value=0)
  }


def _get_num_examples_and_filenames() -> Tuple[Dict[str, int], Dict[str, str]]:
  """Retrieves the number of examples and filenames according to data mode."""
  num_examples = {'train': NUM_TRAIN, 'validation': NUM_VAL, 'test': NUM_TEST}
  file_names = {
      'train': FILENAME_TRAIN,
      'validation': FILENAME_VALID,
      'test': FILENAME_TEST,
      'metadata': FILENAME_META
  }

  return num_examples, file_names


def load_json(json_dir: str) -> Dict[Any, Any]:
  with tf.io.gfile.GFile(json_dir) as json_file:
    return json.load(json_file)


_CITATION = """
@article{zhao2018zero,
  title={Zero-Shot Dialog Generation with Cross-Domain Latent Actions},
  author={Zhao, Tiancheng and Eskenazi, Maxine},
  journal={arXiv preprint arXiv:1805.04803},
  year={2018}
}
"""
_DESCRIPTION = (
    'Simulated goal-oriented conversations [1] generated for information '
    'requests in four domains: bus, restaurant, weather, and movie.')


class _SimDialDatasetBuilder(tfds.core.DatasetBuilder):
  """Minimal TFDS DatasetBuilder, does not support downloading."""
  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
  }

  def __init__(self, data_dir, **kwargs):
    self._num_examples, self._file_names = _get_num_examples_and_filenames()
    self._file_paths = self._get_file_paths(data_dir)

    super().__init__(data_dir=data_dir, **kwargs)
    # We have to reset self._data_dir since the parent class appends the class
    # name and version to dir name.
    self._data_dir = data_dir

  def _download_and_prepare(self, dl_manager, download_config=None):
    """Downloads and prepares dataset for reading."""
    raise NotImplementedError(
        'Must provide a data_dir with the files already downloaded to.')

  def _get_file_paths(self, data_dir) -> Dict[str, str]:
    """Returns the full path to file."""
    get_full_path = lambda name: os.path.join(data_dir, name)
    return {
        'train': get_full_path(self._file_names['train']),
        'validation': get_full_path(self._file_names['validation']),
        'test': get_full_path(self._file_names['test']),
        'metadata': get_full_path(self._file_names['metadata'])
    }

  def _as_dataset(self,
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
          glob_dir=self._file_paths['train'], is_training=True)
    elif split == tfds.Split.VALIDATION:
      return _build_dataset(
          glob_dir=self._file_paths['validation'], is_training=False)
    elif split == tfds.Split.TEST:
      return _build_dataset(
          glob_dir=self._file_paths['test'], is_training=False)
    raise ValueError('Unsupported split given: {}.'.format(split))

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the `tfds.core.DatasetInfo` object."""
    features = {
        USR_UTT_NAME: tfds.features.Tensor(shape=[], dtype=tf.string),
        SYS_UTT_NAME: tfds.features.Tensor(shape=[], dtype=tf.string),
        STATE_LABEL_NAME: tfds.features.Tensor(shape=[], dtype=tf.string),
        DIAL_LEN_NAME: tfds.features.Tensor(shape=[], dtype=tf.int64)
    }

    metadata_dict = load_json(self._file_paths['metadata'])

    info = tfds.core.DatasetInfo(
        builder=self,
        description=_DESCRIPTION,
        features=tfds.features.FeaturesDict(features),
        homepage='https://github.com/snakeztc/SimDial',
        citation=_CITATION,
        # Note that while metadata seems to be the most appropriate way to store
        # arbitrary info, it will not be printed when printing out the dataset
        # info.
        metadata=tfds.core.MetadataDict(**metadata_dict))

    # Instead of having a single element shard_lengths, we should really have a
    # list of the number of elements in each file shard in each split.
    split_infos = [
        tfds.core.SplitInfo(
            name=tfds.Split.VALIDATION,
            shard_lengths=[self._num_examples['validation']],
            num_bytes=0,
        ),
        tfds.core.SplitInfo(
            name=tfds.Split.TEST,
            shard_lengths=[self._num_examples['test']],
            num_bytes=0,
        ),
        tfds.core.SplitInfo(
            name=tfds.Split.TRAIN,
            shard_lengths=[self._num_examples['train']],
            num_bytes=0,
        ),
    ]
    split_dict = tfds.core.SplitDict(
        split_infos, dataset_name='__sim_dial_dataset_builder')
    info.set_splits(split_dict)
    return info


class SimDialDataset(base.BaseDataset):
  """SimDial dataset builder class."""

  def __init__(self,
               split: str,
               shuffle_buffer_size: Optional[int] = None,
               num_parallel_parser_calls: int = 64,
               data_dir: Optional[str] = None,
               download_data: bool = False,
               is_training: Optional[bool] = None):
    """Create a SimDial tf.data.Dataset builder.

    Args:
      split: a dataset split, either a custom tfds.Split or one of the
        tfds.Split enums [TRAIN, VALIDAITON, TEST] or their lowercase string
        names.
      shuffle_buffer_size: the number of example to use in the shuffle buffer
        for tf.data.Dataset.shuffle().
      num_parallel_parser_calls: the number of parallel threads to use while
        preprocessing in tf.data.Dataset.map().
      data_dir: path to a directory containing the tfrecord datasets.
      download_data: Whether or not to download data before loading. Currently
        unsupported.
      is_training: Whether or not the given `split` is the training split. Only
        required when the passed split is not one of ['train', 'validation',
        'test', tfds.Split.TRAIN, tfds.Split.VALIDATION, tfds.Split.TEST].
    """

    # Load vocab for dialog utterances and state labels.
    self.vocab_utter = load_json(os.path.join(data_dir, FILENAME_TOKENIZER))
    self.vocab_label = load_json(
        os.path.join(data_dir, FILENAME_TOKENIZER_LABEL))

    super().__init__(
        name='simdial',
        dataset_builder=_SimDialDatasetBuilder(data_dir),
        split=split,
        is_training=is_training,
        shuffle_buffer_size=shuffle_buffer_size,
        num_parallel_parser_calls=num_parallel_parser_calls,
        download_data=False)

  def _create_process_example_fn(self) -> base.PreProcessFn:

    def _example_parser(example: Dict[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
      """Parse features and labels from a serialized tf.train.Example."""
      features_spec = _make_features_spec()
      features = tf.io.parse_single_example(example['features'], features_spec)

      sys_utt = tf.io.parse_tensor(features[SYS_UTT_NAME], out_type=tf.int32)
      usr_utt = tf.io.parse_tensor(features[USR_UTT_NAME], out_type=tf.int32)
      state_label = tf.io.parse_tensor(
          features[STATE_LABEL_NAME], out_type=tf.int32)
      dialog_len = features[DIAL_LEN_NAME]

      return {
          SYS_UTT_NAME: sys_utt,
          USR_UTT_NAME: usr_utt,
          STATE_LABEL_NAME: state_label,
          DIAL_LEN_NAME: dialog_len
      }

    return _example_parser

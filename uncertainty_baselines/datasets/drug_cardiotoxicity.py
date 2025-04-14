# coding=utf-8
# Copyright 2024 The Uncertainty Baselines Authors.
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

"""Data loader for the Drug Cardiotoxicity dataset.

Drug Cardiotoxicity dataset [1-2] is a molecule classification task to detect
cardiotoxicity caused by binding hERG target, a protein associated with heart
beat rhythm. The data covers over 9000 molecules with hERG activity
(active/inactive).

Note:

1. The data is split into train-validation-test ratio of roughly 8:2:1.

2. The dataset is stored in TFRecord format. Each molecule is represented as
  a 2D graph - nodes are the atoms and edges are the bonds. Each atom is
  represented as a vector encoding basic atom information such as atom type.
  Similar logic applies to bonds.


## References
[1]: Vishal B. S. et al. Critical Assessment of Artificial Intelligence Methods
for Prediction of hERG Channel Inhibition in the Big Data Era.
     JCIM, 2020. https://pubs.acs.org/doi/10.1021/acs.jcim.0c00884

[2]: K. Han et al. Reliable Graph Neural Networks for Drug Discovery Under
Distributional Shift.
    NeurIPS DistShift Workshop 2021. https://arxiv.org/abs/2111.12951
"""
import os.path
from typing import Dict, Optional, Tuple

from absl import logging
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds
from uncertainty_baselines.datasets import base

# filenames for datasets
_FILENAME_TRAIN = 'cardiotox-train.tfrecord*'
_FILENAME_VAL = 'cardiotox-validation.tfrecord*'
_FILENAME_TEST = 'cardiotox-test.tfrecord*'
_FILENAME_TEST2 = 'cardiotox-test2.tfrecord*'

_NUM_TRAIN = 6523
_NUM_VAL = 1631
_NUM_TEST = 839
_NUM_TEST2 = 177

_LABEL_NAME = 'active'
_NODES_FEATURE_NAME = 'atoms'
_EDGES_FEATURE_NAME = 'pairs'
_NODE_MASK_FEATURE_NAME = 'atom_mask'
_EDGE_MASK_FEATURE_NAME = 'pair_mask'
_DISTANCE_TO_TRAIN_NAME = 'dist2topk_nbs'
_EXAMPLE_NAME = 'molecule_id'

_MAX_NODES = 60
_NODE_FEATURE_LENGTH = 27
_EDGE_FEATURE_LENGTH = 12
_NUM_CLASSES = 2


def _build_dataset(glob_dir: str, is_training: bool) -> tf.data.Dataset:
  cycle_len = 10 if is_training else 1
  dataset = tf.data.Dataset.list_files(glob_dir, shuffle=is_training)
  dataset = dataset.interleave(tf.data.TFRecordDataset, cycle_length=cycle_len)
  return dataset


def _make_features_spec() -> Dict[str, tf.io.FixedLenFeature]:
  return {
      _LABEL_NAME:
          tf.io.FixedLenFeature([_NUM_CLASSES], tf.int64),
      _NODES_FEATURE_NAME:
          tf.io.FixedLenFeature([_MAX_NODES, _NODE_FEATURE_LENGTH], tf.float32),
      _EDGES_FEATURE_NAME:
          tf.io.FixedLenFeature([_MAX_NODES, _MAX_NODES, _EDGE_FEATURE_LENGTH],
                                tf.float32),
      _NODE_MASK_FEATURE_NAME:
          tf.io.FixedLenFeature([_MAX_NODES], tf.float32),
      _EDGE_MASK_FEATURE_NAME:
          tf.io.FixedLenFeature([_MAX_NODES, _MAX_NODES], tf.float32),
      _DISTANCE_TO_TRAIN_NAME:
          tf.io.FixedLenFeature(shape=[1], dtype=tf.float32),
      _EXAMPLE_NAME:
          tf.io.FixedLenFeature([], tf.string)
  }


def _get_num_examples_and_filenames() -> Tuple[Dict[str, int], Dict[str, str]]:
  """Retrieves the number of examples and filenames according to data mode."""
  num_examples = {
      'train': _NUM_TRAIN,
      'validation': _NUM_VAL,
      'test': _NUM_TEST,
      'test2': _NUM_TEST2
  }
  file_names = {
      'train': _FILENAME_TRAIN,
      'validation': _FILENAME_VAL,
      'test': _FILENAME_TEST,
      'test2': _FILENAME_TEST2
  }

  return num_examples, file_names


_CITATION = """
@ARTICLE{Han2021-tu,
  title         = "Reliable Graph Neural Networks for Drug Discovery Under
                   Distributional Shift",
  author        = "Han, Kehang and Lakshminarayanan, Balaji and Liu, Jeremiah",
  month         =  nov,
  year          =  2021,
  archivePrefix = "arXiv",
  primaryClass  = "cs.LG",
  eprint        = "2111.12951"
}
"""
_DESCRIPTION = (
    'Drug Cardiotoxicity dataset [1-2] is a molecule classification task to '
    'detect cardiotoxicity caused by binding hERG target, a protein associated '
    'with heart beat rhythm.')


class _DrugCardiotoxicityDatasetBuilder(tfds.core.DatasetBuilder):
  """TFDS DatasetBuilder for Drug Cardiotoxicity, downloading not supported."""
  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
  }

  def __init__(self, data_dir, **kwargs):
    self._num_examples, self._file_names = _get_num_examples_and_filenames()
    super().__init__(data_dir=data_dir, **kwargs)
    # We have to override self._data_dir to prevent the parent class from
    # appending the class name and version.
    self._data_dir = data_dir

  def _download_and_prepare(self, dl_manager, download_config=None):
    """Downloads and prepares dataset for reading."""
    raise NotImplementedError(
        'Must provide a data_dir with the files already downloaded to.')

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
    elif split == tfds.Split('test2'):
      return _build_dataset(
          glob_dir=os.path.join(self._data_dir, self._file_names['test2']),
          is_training=False)
    raise ValueError('Unsupported split given: {}.'.format(split))

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the `tfds.core.DatasetInfo` object."""
    features = {
        _LABEL_NAME:
            tfds.features.ClassLabel(num_classes=_NUM_CLASSES),
        _NODES_FEATURE_NAME:
            tfds.features.Tensor(
                shape=[_MAX_NODES, _NODE_FEATURE_LENGTH], dtype=tf.float32),
        _EDGES_FEATURE_NAME:
            tfds.features.Tensor(
                shape=[_MAX_NODES, _MAX_NODES, _EDGE_FEATURE_LENGTH],
                dtype=tf.float32),
        _NODE_MASK_FEATURE_NAME:
            tfds.features.Tensor(shape=[_MAX_NODES], dtype=tf.float32),
        _EDGE_MASK_FEATURE_NAME:
            tfds.features.Tensor(
                shape=[_MAX_NODES, _MAX_NODES], dtype=tf.float32),
        _DISTANCE_TO_TRAIN_NAME:
            tfds.features.Tensor(shape=[1], dtype=tf.float32),
        _EXAMPLE_NAME:
            tfds.features.Tensor(shape=[], dtype=tf.string),
    }
    info = tfds.core.DatasetInfo(
        builder=self,
        description=_DESCRIPTION,
        features=tfds.features.FeaturesDict(features),
        homepage='https://www.tensorflow.org/datasets/catalog/cardiotox',
        citation=_CITATION,
        # Note that while metadata seems to be the most appropriate way to store
        # arbitrary info, it will not be printed when printing out the dataset
        # info.
        metadata=tfds.core.MetadataDict(
            max_nodes=_MAX_NODES,
            node_features=_NODE_FEATURE_LENGTH,
            edge_features=_EDGE_FEATURE_LENGTH))

    # Instead of having a single element shard_lengths, we should really have a
    # list of the number of elements in each file shard in each split.
    split_infos = [
        tfds.core.SplitInfo(
            name=tfds.Split.VALIDATION,
            shard_lengths=[self._num_examples['validation']],
            num_bytes=0,
            filename_template=tfds.core.filename_template_for(
                builder=self, split=tfds.Split.VALIDATION),
        ),
        tfds.core.SplitInfo(
            name=tfds.Split.TEST,
            shard_lengths=[self._num_examples['test']],
            num_bytes=0,
            filename_template=tfds.core.filename_template_for(
                builder=self, split=tfds.Split.TEST),
        ),
        tfds.core.SplitInfo(
            name=tfds.Split('test2'),
            shard_lengths=[self._num_examples['test2']],
            num_bytes=0,
            filename_template=tfds.core.filename_template_for(
                builder=self, split=tfds.Split('test2')),
        ),
        tfds.core.SplitInfo(
            name=tfds.Split.TRAIN,
            shard_lengths=[self._num_examples['train']],
            num_bytes=0,
            filename_template=tfds.core.filename_template_for(
                builder=self, split=tfds.Split.TRAIN),
        ),
    ]
    split_dict = tfds.core.SplitDict(
        split_infos, dataset_name='__drug_cardiotoxicity_dataset_builder')
    info.set_splits(split_dict)
    return info


class DrugCardiotoxicityDataset(base.BaseDataset):
  """Drug Cardiotoxicity dataset builder class."""

  def __init__(self,
               split: str,
               shuffle_buffer_size: Optional[int] = None,
               num_parallel_parser_calls: int = 64,
               download_data: bool = False,
               data_dir: Optional[str] = None,
               is_training: Optional[bool] = None,
               drop_remainder: bool = False):
    """Create a tf.data.Dataset builder.

    Args:
      split: a dataset split, either a custom tfds.Split or one of the
        tfds.Split enums [TRAIN, VALIDAITON, TEST] or their lowercase string
        names.
      shuffle_buffer_size: the number of example to use in the shuffle buffer
        for tf.data.Dataset.shuffle().
      num_parallel_parser_calls: the number of parallel threads to use while
        preprocessing in tf.data.Dataset.map().
      download_data: Whether or not to download data before loading. Currently
        unsupported.
      data_dir: Path to a directory containing the tfrecord datasets, with
        filenames train-*-of-*', 'validate.tfr', 'test.tfr'.
      is_training: Whether or not the given `split` is the training split. Only
        required when the passed split is not one of ['train', 'validation',
        'test', tfds.Split.TRAIN, tfds.Split.VALIDATION, tfds.Split.TEST].
      drop_remainder: whether or not to drop the last batch of data if the
        number of points is not exactly equal to the batch size. This option
        needs to be True for running on TPUs.
    """
    if data_dir is None:
      builder = tfds.builder('cardiotox')
      data_dir = builder.data_dir
    logging.info('CardioTox data dir: %s', data_dir)

    super().__init__(
        name='drug_cardiotoxicity',
        dataset_builder=_DrugCardiotoxicityDatasetBuilder(data_dir),
        split=split,
        is_training=is_training,
        shuffle_buffer_size=shuffle_buffer_size,
        num_parallel_parser_calls=num_parallel_parser_calls,
        drop_remainder=drop_remainder,
        download_data=False)

  def _create_process_example_fn(self) -> base.PreProcessFn:

    def _example_parser(example: Dict[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
      """Parse features and labels from a serialized tf.train.Example."""
      features_spec = _make_features_spec()
      features = tf.io.parse_example(example['features'], features_spec)
      labels = features.pop(_LABEL_NAME)

      return {'features': features, 'labels': labels}

    return _example_parser

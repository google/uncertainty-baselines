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

"""Minimal TFDS DatasetBuilder backed by SQL, does not support downloading."""

import os
import sqlite3
import tempfile
from typing import Mapping

from absl import logging
import pandas as pd
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_federated as tff


class TFDSBuilderFromSQLClientData(tfds.core.DatasetBuilder):
  """Minimal TFDS DatasetBuilder backed by SQL, does not support downloading."""
  VERSION = tfds.core.Version('0.0.0')

  def __init__(
      self,
      sql_database: str,
      tfds_features: tfds.features.FeaturesDict,
      element_spec: Mapping[str, tf.TensorSpec],
      **kwargs,
  ):

    self._cd = tff.simulation.datasets.load_and_parse_sql_client_data(
        sql_database, element_spec=element_spec)
    self._tfds_features = tfds_features
    self._sql_database = sql_database
    super().__init__(data_dir=sql_database, **kwargs)
    self._data_dir = sql_database

  def _download_and_prepare(self, dl_manager, download_config=None):
    """Downloads aren't supported."""
    raise NotImplementedError(
        'Must provide a data_dir with the files already downloaded to.')

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the `tfds.core.DatasetInfo` object."""
    info = tfds.core.DatasetInfo(
        builder=self,
        description='A minimal TFDS DatasetBuilder backed by SQL ClientData',
        features=self._tfds_features,
        homepage='N/A',
        citation='N/A',
        metadata=None)
    df = _load_sql_client_data_metadata(self._sql_database)

    split_infos = list()

    for client_id in self._cd.client_ids:
      split_infos.append(
          tfds.core.SplitInfo(
              name=client_id,
              shard_lengths=[
                  int(df[df['client_id'] == client_id]['num_examples'])
              ],
              num_bytes=0))

    split_dict = tfds.core.SplitDict(
        split_infos, dataset_name='tfds_builder_by_sql_client_data')
    info.set_splits(split_dict)
    return info

  def _as_dataset(self,
                  split: tfds.Split,
                  decoders=None,
                  read_config=None,
                  shuffle_files=False) -> tf.data.Dataset:
    """Constructs a `tf.data.Dataset`.

    Args:
      split: `tfds.Split` which subset of the data to read.
      decoders: Unused.
      read_config: Unused.
      shuffle_files: Unused.

    Returns:
      A `tf.data.Dataset`.
    """
    del decoders  # Image will not be decoded
    del read_config
    del shuffle_files

    if hasattr(split, 'split'):
      client_id = split.split
    else:
      client_id = str(split)

    return self._cd.create_tf_dataset_for_client(client_id)

  @property
  def client_ids(self):
    return self._cd.client_ids


def _load_sql_client_data_metadata(database_filepath: str) -> pd.DataFrame:
  """Load the metadata from a SqlClientData database.

  This function will first fetch the SQL database to a local temporary
  directory if `database_filepath` is a remote directory.

  Args:
    database_filepath: A `str` filepath of the SQL database.

  Returns:
    A pandas.DataFrame containing the metadata.

  Raises:
    FileNotFoundError: if database_filepath does not exist.
  """

  if not tf.io.gfile.exists(database_filepath):
    raise FileNotFoundError(f'No such file or directory: {database_filepath}')
  elif not os.path.exists(database_filepath):
    logging.info('Starting fetching SQL database to local.')
    tmp_dir = tempfile.mkdtemp()
    tmp_database_filepath = tf.io.gfile.join(
        tmp_dir, os.path.basename(database_filepath))
    tf.io.gfile.copy(database_filepath, tmp_database_filepath, overwrite=True)
    database_filepath = tmp_database_filepath
    logging.info('Finished fetching SQL database to local.')

  con = sqlite3.connect(database_filepath)
  return pd.read_sql_query('SELECT * from client_metadata', con)

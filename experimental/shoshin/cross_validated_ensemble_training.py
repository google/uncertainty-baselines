# coding=utf-8
# Copyright 2022 The Uncertainty Baselines Authors.
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

"""Train a cross-validated ensemble.

Split the data k fold and train each model on a separate slice.
"""

import getpass
from typing import Sequence

from absl import app
from absl import flags
from ml_collections import config_flags
import tensorflow_datasets as tfds
import training_library  # local file import from experimental.shoshin

from google3.learning.deepmind.researchdata import datatables
from google3.learning.deepmind.xmanager2.client import xmanager_api

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file(
    'config',
    default='configs/config_tf_launch.py',
    help_string='config file',
    lock_config=False)

_WORKDIR = flags.DEFINE_string(
    'workdir', None, 'Work unit directory.', required=True)
flags.mark_flags_as_required(['config', 'workdir'])


def create_datasets_iterators(num_splits, seed, index, batch_size,
                              eval_batch_size):
  """Create dataset iterators."""
  # Create train and validation splits
  read_config = tfds.ReadConfig()
  read_config.add_tfds_id = True  # Set `True` to return the 'tfds_id' key

  split_size_in_pct = int(100 / num_splits)
  vals_ds = tfds.load(
      'celeb_a',
      read_config=read_config,
      split=[
          f'train[{k}%:{k+split_size_in_pct}%]'
          for k in range(0, 100, split_size_in_pct)
      ])[index]
  trains_ds = tfds.load(
      'celeb_a',
      read_config=read_config,
      split=[
          f'train[:{k}%]+train[{k+split_size_in_pct}%:]'
          for k in range(0, 100, split_size_in_pct)
      ])[index]

  # Create dataset iterators
  train_ds_iterator = iter(
      tfds.as_numpy(
          trains_ds.batch(batch_size).shuffle(10 * batch_size,
                                              seed=seed).cache().repeat()))
  val_ds_iterator = iter(
      tfds.as_numpy(vals_ds.batch(eval_batch_size).cache().repeat()))
  return trains_ds, vals_ds, train_ds_iterator, val_ds_iterator


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  config = FLAGS.config
  xm_client = xmanager_api.XManagerApi(xm_deployment_env='alphabet')
  experiment = xm_client.get_current_experiment()
  wid = xm_client.get_current_work_unit().id
  if experiment.id > 0:
    workdir = _WORKDIR.value.format(
        username=getpass.getuser(), xid=experiment.id, wid=wid)
  else:
    workdir = _WORKDIR.value
  config.train.output_dir = workdir
  acls = datatables.DatatableACLs(
      owners=(getpass.getuser(),),
      readers=('all-person-users',),
      writers=(getpass.getuser(),))
  table = (f'/datatable/xid/{experiment.id}/predictions'
           if experiment.id > 0 else config.datatable)
  if config.train.train_bias:
    writer = None
  else:
    writer = datatables.Writer(
        table,
        keys=[('index', int), ('id', str)],
        fixed_key_values=[
            config.index,
        ],
        options=datatables.WriterOptions(acls=acls))
  if experiment.id > 0:
    config.train.bias_id = experiment.id
  training_library.load_data_train_model(config.dataset, config.train,
                                         config.train.model, config.index,
                                         writer)


if __name__ == '__main__':
  app.run(main)

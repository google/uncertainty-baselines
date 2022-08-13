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

import functools
import getpass
from typing import Sequence

from absl import app
from absl import flags
from ml_collections import config_flags
import tensorflow as tf
import tensorflow_datasets as tfds
import training  # local file import from experimental.shoshin

from google3.learning.deepmind.researchdata import datatables
from google3.learning.deepmind.xmanager2.client import xmanager_api

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file(
    'config', default='config.py', help_string='config file', lock_config=False)

_WORKDIR = flags.DEFINE_string(
    'workdir', None, 'Work unit directory.', required=True)
flags.mark_flags_as_required(['config', 'workdir'])


def preprocess_batch(input_shape, batch):
  image = tf.image.resize(batch['image'], input_shape[:2])
  image = tf.keras.applications.resnet50.preprocess_input(image)
  label = batch['attributes']['Male'].astype(int)
  return image.numpy(), label


def write_predictions(predictor, data_iterator, writer, in_sample):
  """Writer predictions of predictor on data from data_iterator to writer."""
  for batch in data_iterator:
    predictions = predictor(batch)
    # TODO(dvij): Remove hard coding here
    label = batch['attributes']['Male'].astype(int)
    label_attr = batch['attributes']['Blond_Hair'].astype(int)
    for j in range(predictions.shape[0]):
      measures = {'id': str(batch['tfds_id'][j]), 'in_sample': in_sample,
                  'label': label[j].item(), 'label_attr': label_attr[j].item(),
                  'prediction': predictions[j].item()}
      writer.write(measures)
  return


def create_datasets_iterators(
    num_splits, seed, index, batch_size, eval_batch_size):
  """Create dataset iterators."""
  # Create train and validation splits
  read_config = tfds.ReadConfig()
  read_config.add_tfds_id = True  # Set `True` to return the 'tfds_id' key

  split_size_in_pct = int(100/num_splits)
  vals_ds = tfds.load(
      'celeb_a', read_config=read_config,
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
      tfds.as_numpy(trains_ds.batch(batch_size).shuffle(
          10 * batch_size, seed=seed).cache().repeat()))
  val_ds_iterator = iter(
      tfds.as_numpy(vals_ds.batch(
          eval_batch_size).cache().repeat()))
  return trains_ds, vals_ds, train_ds_iterator, val_ds_iterator


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  config = FLAGS.config
  batch_size = config.train.train_batch_size
  xm_client = xmanager_api.XManagerApi(xm_deployment_env='alphabet')
  experiment = xm_client.get_current_experiment()
  wid = xm_client.get_current_work_unit().id
  workdir = (
      f'/xfile/{getpass.getuser()}/{experiment.id}/{wid}/' if experiment.id > 0
      else FLAGS.workdir)

  acls = datatables.DatatableACLs(
      owners=(getpass.getuser(),),
      readers=('all-person-users',),
      writers=(getpass.getuser(),))
  table = (f'/datatable/xid/{experiment.id}/predictions' if experiment.id > 0
           else config.datatable)
  with datatables.Writer(
      table,
      keys=[('index', int), ('id', str)],
      fixed_key_values=[
          config.index,
      ],
      options=datatables.WriterOptions(acls=acls)) as writer:

    trains_ds, vals_ds, train_ds_iterator, val_ds_iterator = (
        create_datasets_iterators(config.num_splits, config.dataset_seed,
                                  config.index, batch_size,
                                  config.train.eval_batch_size))
    predictor = training.train_loop(
        config.train, workdir, train_ds_iterator, val_ds_iterator,
        functools.partial(preprocess_batch, config.train.input_shape))
    write_predictions(
        predictor, tfds.as_numpy(trains_ds.batch(batch_size)), writer, True)
    write_predictions(
        predictor, tfds.as_numpy(
            vals_ds.batch(config.train.eval_batch_size)), writer, False)


if __name__ == '__main__':
  app.run(main)

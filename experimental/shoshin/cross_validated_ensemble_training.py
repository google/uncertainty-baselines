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

from typing import Sequence

from absl import app
from absl import flags
from ml_collections import config_flags
import tensorflow_datasets as tfds

import training  # local file import from experimental.shoshin

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file(
    'config', default='config.py', help_string='config file', lock_config=False)

_WORKDIR = flags.DEFINE_string(
    'workdir', None, 'Work unit directory.', required=True)
flags.mark_flags_as_required(['config', 'workdir'])


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  config = FLAGS.config
  batch_size = config.train.train_batch_size
  index = config.index  # index of CV fold to be used for training/validation

  # TODO(dvij): Remove hardcording of splits here
  # Create train and validation splits
  vals_ds = tfds.load(
      'cifar10',
      split=[f'train[{k}%:{k+10}%]' for k in range(0, 100, 10)])[index]
  trains_ds = tfds.load(
      'cifar10',
      split=[f'train[:{k}%]+train[{k+10}%:]' for k in range(0, 100, 10)])[index]

  # Create dataset iterators
  train_ds = iter(
      tfds.as_numpy(trains_ds.cache().repeat().shuffle(
          10 * batch_size, seed=0).batch(batch_size)))
  val_ds = iter(
      tfds.as_numpy(vals_ds.cache().repeat().batch(
          config.train.eval_batch_size)))

  training.train_loop(config.train, FLAGS.workdir, train_ds, val_ds)


if __name__ == '__main__':
  app.run(main)

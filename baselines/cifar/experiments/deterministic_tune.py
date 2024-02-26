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

r"""Deterministic baseline for Diabetic Retinopathy Detection.

"""

import datetime
import getpass
import os.path
import random

from ml_collections import config_dict


def get_config():
  """Returns the configuration for this experiment."""
  config = config_dict.ConfigDict()
  config.user = getpass.getuser()
  config.priority = 'prod'
  config.platform = 'gpu'
  config.gpu_type = 't4'
  config.num_gpus = 1
  config.experiment_name = (
      os.path.splitext(os.path.basename(__file__))[0] + '_' +
      datetime.datetime.today().strftime('%Y-%m-%d-%H-%M-%S'))
  output_dir = 'gs://launcher-beta-test-bucket/{}'.format(
      config.experiment_name)
  config.args = {
      'train_epochs': 90,
      'per_core_batch_size': 64,
      'checkpoint_interval': -1,
      'data_dir': output_dir,
      'output_dir': output_dir,
      'download_data': True,
      'train_proportion': 0.9,
      'eval_on_ood': True,
      'ood_dataset': 'cifar100,svhn_cropped',
      # If drop_remainder=false, it will cause the issue of
      # `TPU has inputs with dynamic shapes` for sngp.py
      # To make the evaluation comparable, we set true for deterministic.py too.
      'drop_remainder_for_eval': True,
  }
  return config


def get_sweep(hyper):
  return hyper.product([
      hyper.sweep('dempster_shafer_ood', hyper.categorical([False, True])),
      hyper.sweep('seed', hyper.discrete(random.sample(range(1, int(1e10)),
                                                       5))),
  ])

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

r"""Deterministic baseline for Diabetic Retinopathy Detection.

Refining based off
baselines/diabetic_retinopathy_detection/initial_tuning/experiments/deterministic_tune.py.

"""

import datetime
import getpass
import os.path

from ml_collections import config_dict


def get_config(launch_on_gcp):
  """Returns the configuration for this experiment."""
  config = config_dict.ConfigDict()
  config.user = getpass.getuser()
  config.priority = 'prod'
  config.platform = 'tpu-v3'
  config.tpu_topology = '2x2'
  config.experiment_name = (
      os.path.splitext(os.path.basename(__file__))[0] + '_' +
      datetime.datetime.today().strftime('%Y-%m-%d-%H-%M-%S'))
  output_dir = 'gs://launcher-beta-test-bucket/diabetic_retinopathy_detection/{}'.format(
      config.experiment_name)
  data_dir = 'gs://ub-data/retinopathy'
  config.args = {
      'train_epochs': 90,
      'use_gpu': False,  # Use TPU.
      'train_batch_size': 64,
      'eval_batch_size': 64,
      'output_dir': output_dir,
      'checkpoint_interval': -1,
      'lr_schedule': 'linear',
      'data_dir': data_dir,
  }
  return config


def get_sweep(hyper):
  num_trials = 50
  return hyper.zipit([
      hyper.loguniform('base_learning_rate', hyper.interval(0.03, 0.5)),
      hyper.uniform('final_decay_factor', hyper.discrete([1e-3, 1e-2, 0.1])),
      hyper.loguniform('one_minus_momentum', hyper.interval(5e-3, 0.05)),
      hyper.loguniform('l2', hyper.interval(1e-6, 2e-4)),
  ], length=num_trials)

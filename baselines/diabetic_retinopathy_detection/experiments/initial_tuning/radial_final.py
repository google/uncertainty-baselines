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

r"""Radial BNN baseline for Diabetic Retinopathy Detection.

10 seeds using second-best hyperparameters from
baselines/diabetic_retinopathy_detection/initial_tuning/experiments/radial_tune_final.py.

We had to use the second-best hyperparameter point because the best point was
very unstable, 11/20 runs diverged (dropped to 0 test AUC).
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
      'batch_size': 32,
      'output_dir': output_dir,
      # Checkpoint every eval to get the best checkpoints via early stopping.
      'checkpoint_interval': 1,
      # Second-best hparams.
      'base_learning_rate': 0.84557,
      'one_minus_momentum': 0.023980,
      'l2': 0.00019403,
      'stddev_mean_init': 0.000018096,
      'stddev_stddev_init': 0.067054,
      'use_validation': False,
      'data_dir': data_dir,
  }
  return config


def get_sweep(hyper):
  num_trials = 10
  return hyper.sweep('seed', hyper.discrete(range(42, 42 + num_trials)))

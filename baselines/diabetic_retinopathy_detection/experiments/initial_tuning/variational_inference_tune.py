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

r"""Variational_inference baseline for Diabetic Retinopathy Detection.

"""

import datetime
import getpass
import os.path

from ml_collections import config_dict


def get_config(launch_on_gcp):
  """Returns the configuration for this experiment."""
  config = config_dict.ConfigDict()
  config.user = getpass.getuser()
  config.priority = 'freebie'
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
      'checkpoint_interval': -1,
      'data_dir': data_dir,
  }
  return config


def get_sweep(hyper):
  num_trials = 50
  return hyper.zipit([
      hyper.loguniform('base_learning_rate', hyper.interval(1e-3, 0.1)),
      hyper.loguniform('one_minus_momentum', hyper.interval(1e-2, 0.1)),
      hyper.loguniform('l2', hyper.interval(1e-5, 1e-3)),
      hyper.loguniform('stddev_mean_init', hyper.interval(1e-5, 1e-1)),
      hyper.loguniform('stddev_stddev_init', hyper.interval(1e-2, 1.0)),
  ], length=num_trials)

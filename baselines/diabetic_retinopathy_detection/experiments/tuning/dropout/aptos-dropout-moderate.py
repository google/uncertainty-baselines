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

r"""MC Dropout baseline for Diabetic Retinopathy Detection, evaluated on the OOD APTOS Indian retinopathy dataset with moderate decision threshold.
"""

import datetime
import getpass
import os.path

from ml_collections import config_dict


launch_on_gcp = True


def get_config():
  """Returns the configuration for this experiment."""
  config = config_dict.ConfigDict()
  config.user = getpass.getuser()
  config.priority = 'prod'
  config.platform = 'tpu-v2'
  config.tpu_topology = '2x2'
  config.experiment_name = (
      os.path.splitext(os.path.basename(__file__))[0] + '_' +
      datetime.datetime.today().strftime('%Y-%m-%d-%H-%M-%S'))
  output_dir = 'gs://drd-dropout-aptos-results/{}'.format(
      config.experiment_name)
  config.args = {
      'per_core_batch_size': 16,
      'num_dropout_samples_eval': 5,
      'train_epochs': 90,
      'num_cores': 8,
      'class_reweight_mode': 'minibatch',
      'dr_decision_threshold': 'moderate',
      'distribution_shift': 'aptos',
      'checkpoint_interval': 1,
      'output_dir': output_dir,
      'data_dir': 'gs://ub-data/retinopathy',
  }
  return config


def get_sweep(hyper):
  num_trials = 32
  return hyper.zipit([
      hyper.loguniform('base_learning_rate', hyper.interval(1e-3, 0.5)),
      hyper.loguniform('one_minus_momentum', hyper.interval(5e-3, 0.05)),
      hyper.loguniform('l2', hyper.interval(1e-6, 2e-4)),
      hyper.uniform('dropout_rate', hyper.interval(0.05, 0.2)),
  ],
                     length=num_trials)

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

r"""
MC Dropout baseline for Diabetic Retinopathy Detection,
evaluated on the Severity Shift with mild decision threshold.
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
  config.platform = 'gpu'
  config.experiment_name = ('nband' + '_' +
      os.path.splitext(os.path.basename(__file__))[0] + '_' +
      datetime.datetime.today().strftime('%Y-%m-%d-%H-%M-%S'))
  output_dir = 'gs://drd-dropout-severity-results/{}'.format(
    config.experiment_name)
  config.args = {
      # 'per_core_batch_size': 64,
      'epochs': 90,
      'num_cores': 4,
      'dr_decision_threshold': 'moderate',
      'distribution_shift': 'severity',
      'checkpoint_interval': 1,
      'output_dir': output_dir,
      'data_dir': 'gs://ub-data/retinopathy',
      'layer_to_linearize': 1,
      'per_core_batch_size': 64,
  }
  return config

def get_sweep(hyper):
  num_trials = 50
  return hyper.zipit([
    hyper.uniform('prior_cov', hyper.interval(1, 20)),
    hyper.uniform('n_inducing_inputs', hyper.discrete([10, 20, 50])),
    hyper.uniform('base_learning_rate', hyper.interval(0.020824, 0.031448)),
    hyper.loguniform('one_minus_momentum', hyper.interval(5e-3, 0.05)),
    hyper.loguniform('l2', hyper.interval(1e-6, 2e-4)),
    hyper.uniform('loss_type', hyper.discrete([3, 5])),
  ], length=num_trials)

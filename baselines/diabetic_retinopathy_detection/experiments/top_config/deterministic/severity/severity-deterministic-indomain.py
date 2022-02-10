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

r"""Top hyperparameter configuration of the deterministic baseline for Diabetic Retinopathy Detection, tuning on in-domain validation AUC, evaluated on the Severity Shift with moderate decision threshold.
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
  output_dir = 'gs://drd-deterministic-severity-finetune/indomain/{}'.format(
      config.experiment_name)
  config.args = {
      'per_core_batch_size': 32,
      'train_epochs': 90,
      'num_cores': 8,
      'class_reweight_mode': 'minibatch',
      'dr_decision_threshold': 'moderate',
      'distribution_shift': 'severity',
      'checkpoint_interval': 1,
      'lr_schedule': 'linear',
      'output_dir': output_dir,
      'data_dir': 'gs://ub-data/retinopathy',

      # Hypers
      'final_decay_factor': 0.010000,
      'one_minus_momentum': 0.015731,
      'l2': 0.0000031828,
      'base_learning_rate': 0.058737
  }
  return config


def get_sweep(hyper):
  num_trials = 6
  return hyper.sweep('seed', hyper.discrete(range(42, 42 + num_trials)))

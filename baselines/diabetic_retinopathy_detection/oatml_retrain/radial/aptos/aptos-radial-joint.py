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
Radial baseline for Diabetic Retinopathy Detection,
evaluated on APTOS.
"""

import datetime
import getpass
import os.path

from ml_collections import config_dict


launch_on_gcp = True

# Run on A100s

def get_config():
  """Returns the configuration for this experiment."""
  config = config_dict.ConfigDict()
  config.user = getpass.getuser()
  config.priority = 'prod'
  config.platform = 'tpu-v2'
  config.tpu_topology = '2x2'
  # config.platform = 'gpu'
  # config.gpu_type = 'a100'
  # config.num_gpus = 4

  config.experiment_name = ('nband' + '_' +
      os.path.splitext(os.path.basename(__file__))[0] + '_' +
      datetime.datetime.today().strftime('%Y-%m-%d-%H-%M-%S'))
  output_dir = 'gs://drd-radial-aptos-finetune/joint/{}'.format(
    config.experiment_name)
  config.args = {
      'batch_size': 16,
      'num_mc_samples_train': 1,
      'num_mc_samples_eval': 5,
      'train_epochs': 90,
      'num_cores': 8,
      'class_reweight_mode': 'minibatch',
      'dr_decision_threshold': 'moderate',
      'distribution_shift': 'aptos',
      'checkpoint_interval': 1,
      'output_dir': output_dir,
      'data_dir': 'gs://ub-data/retinopathy',

      # https://wandb.ai/uncertainty-baselines/aptos-radial-moderate-aug24-tpuv2/runs/yvohv18x?workspace=user-nband
      'l2': 0.00014935598488986335,
      'one_minus_momentum': 0.03291582226615088,
      'stddev_stddev_init': 0.06782455683568875,
      'stddev_mean_init': 2.140984173642608e-05,
      'base_learning_rate': 0.15606291288576823
  }
  return config


def get_sweep(hyper):
  num_trials = 6
  return hyper.sweep('seed', hyper.discrete(range(42, 42 + num_trials)))

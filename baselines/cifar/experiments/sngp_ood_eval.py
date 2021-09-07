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

r"""SNGP OOD evaluation for CIFAR10 baselines.


"""

import datetime
import getpass
import os.path

from ml_collections import config_dict


def get_config():
  """Returns the configuration for this experiment."""
  config = config_dict.ConfigDict()
  config.user = getpass.getuser()
  config.priority = 'prod'
  config.platform = 'gpu'

  config.experiment_name = (
      os.path.splitext(os.path.basename(__file__))[0] + '_' +
      datetime.datetime.today().strftime('%Y-%m-%d-%H-%M-%S'))
  config.args = {
      'eval_only': True,
      'run_ood': True,
      'ood_dataset':
          'cifar100,svhn_cropped',
  }
  return config


def get_sweep(hyper):
  num_trials = 10
  saved_model_dirs = [
      os.path.join(base_model_dir, '{}-seed:{}'.format(i + 1, i))
      for i in range(num_trials)
  ]
  return hyper.sweep('saved_model_dir', saved_model_dirs)

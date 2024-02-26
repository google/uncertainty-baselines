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

r"""Ensemble WideResNet CIFAR-100.

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
  config.gpu_type = 'v100'
  config.num_gpus = 1
  config.experiment_name = (
      os.path.splitext(os.path.basename(__file__))[0] + '_' +
      datetime.datetime.today().strftime('%Y-%m-%d-%H-%M-%S'))
  config.args = {
      'dataset':
          'cifar100',
      'eval_on_ood':
          True,
      'ood_dataset':
          'cifar10,svhn_cropped',
      # If drop_remainder=false, it will cause the issue of
      # `TPU has inputs with dynamic shapes` for sngp.py
      # To make the evaluation comparable, we set true for deterministic.py too.
      'drop_remainder_for_eval':
          True,
  }
  return config


def get_sweep(hyper):
  return hyper.product([
      hyper.sweep('seed', list(range(10))),
      hyper.sweep('use_spec_norm', hyper.categorical([False, True])),
      hyper.sweep('use_gp_layer', hyper.categorical([False, True])),
      hyper.sweep('gp_mean_field_factor_ensemble', hyper.discrete([5])),
      hyper.sweep('dempster_shafer_ood', hyper.categorical([False, True])),
  ])

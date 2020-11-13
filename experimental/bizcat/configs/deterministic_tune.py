# coding=utf-8
# Copyright 2020 The Uncertainty Baselines Authors.
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

r"""Vizier for deterministic baseline for Bizcat.

gxm
third_party/py/uncertainty_baselines/baselines/xm_launcher.py \
--binary=//third_party/py/uncertainty_baselines/experimental/bizcat/\
deterministic.py \
--config=//third_party/py/uncertainty_baselines/experimental/bizcat/configs/\
deterministic_tune.py
"""

from ml_collections import config_dict
import numpy as np

NUM_RUN = 25


def get_config():
  """Returns the configuration for this experiment."""
  config = config_dict.ConfigDict()
  config.cell = 'iz'
  config.priority = 'prod'
  config.platform = 'jf'
  config.tpu_topology = '4x4'
  config.max_task_failures = 5
  config.ttl = -1
  config.num_runs = NUM_RUN
  config.args = {}
  return config


def get_sweep(hyper):
  num_runs = NUM_RUN
  domain = [
      hyper.sweep('base_learning_rate',
                  hyper.discrete(np.logspace(-2, -1, num=num_runs).tolist())),
  ]
  sweep = hyper.product(domain)
  return sweep

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

r"""Template of linear VRNN for SGDDataset."""
import os

from psl import constants_dstc as psl_config  # local file import from experimental.language_structure
from uncertainty_baselines.experimental.language_structure.vrnn.experiments.linear_vrnn import default_config


_DATASET = 'sgd'


def add_psl_config(config):
  config.psl_constraint_rule_names = psl_config.RULE_NAMES
  config.psl_constraint_rule_weights = psl_config.RULE_WEIGHTS
  config.psl = psl_config.DATA_CONFIG
  config.psl['batch_size'] = config.train_batch_size


def get_config(**kwargs):
  """Returns the configuration for this experiment."""
  config = default_config.get_config(_DATASET, **kwargs)

  config.has_ood = True

  config.train_epochs = 60

  config.train_batch_size = 16
  config.eval_batch_size = 16
  config.model.vae_cell.encoder_hidden_size = 128

  config.patience = -1
  config.platform = 'pf'
  config.tpu_topology = '2x2x2'


  add_psl_config(config)

  return config

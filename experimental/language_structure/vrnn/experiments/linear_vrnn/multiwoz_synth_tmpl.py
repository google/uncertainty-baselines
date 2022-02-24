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

r"""Template of linear VRNN for MultiWoZSynthDataset."""

import os

import default_config  # local file import from experimental.language_structure.vrnn.experiments.linear_vrnn

_DATASET = 'multiwoz_synth'


def add_psl_config(config):
  config.psl_constraint_rule_names = [
      'rule_%d' % rule_id for rule_id in range(1, 13)
  ]
  config.psl_constraint_rule_weights = [
      1.0, 20.0, 5.0, 5.0, 5.0, 10.0, 5.0, 20.0, 5.0, 5.0, 5.0, 10.0
  ]
  config.psl_config_file = os.path.join(
      default_config.get_config_dir(_DATASET), 'psl_config.json')


def get_config(**kwargs):
  """Returns the configuration for this experiment."""
  config = default_config.get_config(_DATASET, **kwargs)

  config.train_epochs = 100


  add_psl_config(config)

  return config

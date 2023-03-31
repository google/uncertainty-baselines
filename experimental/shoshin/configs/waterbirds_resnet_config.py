# coding=utf-8
# Copyright 2023 The Uncertainty Baselines Authors.
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

"""Configuration file for experiment with Waterbirds data and ResNet model."""

import ml_collections
from configs import base_config  # local file import from experimental.shoshin


def get_config() -> ml_collections.ConfigDict:
  """Get mlp config."""
  config = base_config.get_config()

  # Consider landbirds on water and waterbirds on land as subgroups.
  config.data.subgroup_ids = ()  # ('0_1', '1_0')
  config.data.subgroup_proportions = ()  # (0.04, 0.012)
  config.data.initial_sample_proportion = .25

  config.active_sampling.num_samples_per_round = 500
  config.num_rounds = 4

  data = config.data
  data.name = 'waterbirds'
  data.num_classes = 2

  model = config.model
  model.name = 'resnet'
  model.dropout_rate = 0.2

  # Set to 0 to compute introspection signal based on the best epoch.
  config.eval.num_signal_ckpts = 0
  return config

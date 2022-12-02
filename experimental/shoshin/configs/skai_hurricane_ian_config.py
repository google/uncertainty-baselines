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

"""Configuration file for experiment with SKAI data and ResNet model."""

import ml_collections
from configs import base_config  # local file import from experimental.shoshin


def get_config() -> ml_collections.ConfigDict:
  """Get mlp config."""
  config = base_config.get_config()

  config.train_bias = False
  config.num_rounds = 1
  config.round_idx = 0

  data = config.data
  data.name = 'skai'
  data.num_classes = 2
  # TODO(jihyeonlee): Determine what are considered subgroups in SKAI domain
  # and add support for identifying by ID.
  data.subgroup_ids = ()
  data.subgroup_proportions = ()
  data.initial_sample_proportion = 1.
  data.labeled_train_pattern = ''
  data.unlabeled_train_pattern = ''
  data.validation_pattern = ''
  data.use_post_disaster_only = False

  model = config.model
  model.name = 'resnet'
  model.num_channels = 6

  return config

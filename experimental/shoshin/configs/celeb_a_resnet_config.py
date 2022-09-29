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

"""Configuration file for experiment with Cardiotoxicity data and MLP model."""

import ml_collections
from configs import base_config  # local file import from experimental.shoshin


def get_config() -> ml_collections.ConfigDict:
  """Get mlp config."""
  config = base_config.get_config()

  config.data.subgroup_ids = (
      'Blond_Hair',
  )  # ('Blond_Hair') Currently only the use ofa single attribute supported
  config.data.subgroup_proportions = (0.01,)

  data = config.data
  data.name = 'local_celeb_a'
  data.num_classes = 2

  model = config.model
  model.name = 'resnet'
  model.dropout_rate = 0.2

  return config

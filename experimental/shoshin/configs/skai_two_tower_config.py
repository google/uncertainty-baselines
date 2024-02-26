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

r"""Configuration file for experiment with SKAI data and TwoTower model.

"""


import ml_collections
from configs import base_config  # local file import from experimental.shoshin


def get_config() -> ml_collections.ConfigDict:
  """Get two tower config."""
  config = base_config.get_config()

  config.train_bias = False
  config.num_rounds = 1
  config.round_idx = 0
  config.train_stage_2_as_ensemble = False
  config.save_train_ids = False

  data = config.data
  data.name = 'skai'
  data.num_classes = 2
  data.subgroup_ids = ()
  data.subgroup_proportions = ()
  data.initial_sample_proportion = 1.
  data.tfds_dataset_name = 'skai_dataset'
  data.tfds_data_dir = '/tmp/skai_dataset'
  data.labeled_train_pattern = ''
  data.unlabeled_train_pattern = ''
  data.validation_pattern = ''
  data.use_post_disaster_only = False
  data.batch_size = 32

  model = config.model
  model.load_pretrained_weights = True
  model.name = 'two_tower'
  model.num_channels = 6

  config.optimizer.learning_rate = 1e-4
  config.optimizer.type = 'adam'

  config.training.num_epochs = 100

  return config

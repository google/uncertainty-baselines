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

"""Config for cross validated ensemble training."""
from ml_collections import config_dict


def get_config():
  """Get config."""
  config = config_dict.ConfigDict()
  config.train = get_train_config()
  config.index = 0
  return config


def get_train_config():
  """Get training config."""
  config = config_dict.ConfigDict()
  config.train_seed = 0
  config.train_batch_size = 128
  config.eval_batch_size = 128
  config.input_shape = (224, 224, 3)

  config.logging_frequency = 1000
  config.checkpoint_every = 10000

  config.optimizer = get_optimizer_config()
  config.model = get_model_config([100, 100], 10)

  return config


def get_optimizer_config():
  """Get optimizer config."""
  config = config_dict.ConfigDict()
  config.learning_rate = 1e-3
  config.num_steps = 100000
  return config


def get_model_config(hidden_sizes, output_size):
  """Get model config."""
  config = config_dict.ConfigDict()
  config.hidden_sizes = hidden_sizes
  config.output_size = output_size
  return config

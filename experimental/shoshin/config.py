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
  config.index = 0
  config.train_batch_size = 128
  config.eval_batch_size = 128
  config.hidden_sizes = [200, 200]
  config.learning_rate = 1e-3
  config.num_steps = 100000
  config.logging_frequency = 1000
  config.checkpoint_every = 10000
  return config

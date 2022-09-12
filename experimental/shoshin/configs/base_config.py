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

"""Base configuration file.

Serves as base config for custom configs, which will specify the model and
dataloader to use as well as experiment-level arguments, like whether or not to
generate a bias table or train the stage 2 model as an ensemble.
"""

import ml_collections


def check_flags(config: ml_collections.ConfigDict):
  """Checks validity of certain config values."""
  if not config.output_dir:
    raise ValueError('output_dir has to be specified.')
  if not config.data.name:
    raise ValueError('config.data.name has to be specified.')
  if not config.model.name:
    raise ValueError('config.model.name has to be specified.')
  if 100 % config.data.num_splits != 0:
    raise ValueError('100 should be divisible by config.data.num_splits ',
                     'because we use TFDS split by percent feature.')
  if config.bias_percentile_threshold < 0. or config.bias_percentile_threshold > 1.:
    raise ValueError(
        'config.bias_percentile_threshold must be between 0. and 1.')
  if config.bias_value_threshold and (config.bias_value_threshold < 0. or
                                      config.bias_value_threshold > 1.):
    raise ValueError('config.bias_value_threshold must be between 0. and 1.')


def get_config() -> ml_collections.ConfigDict:
  """Get config."""
  config = ml_collections.ConfigDict()

  config.output_dir = ''

  # Number of rounds of active sampling to conduct.
  config.num_rounds = 3

  # Threshold to generate bias labels. Can be specified as percentile or value.
  config.bias_percentile_threshold = 0.2
  config.bias_value_threshold = None
  config.save_bias_table = True
  # Path to existing bias table to use in training the bias head. If
  # unspecified, generates new one.
  config.path_to_existing_bias_table = ''

  config.train_bias = True
  # When True, does not conduct multiple rounds of active sampling and simply
  # trains a single model.
  config.train_single_model = False
  # When True, trains the stage 2 model (stage 1 is calculating bias table)
  # as an ensemble of models. When True and only a single model is being
  # trained, trains that model as an ensemble.
  config.train_stage_2_as_ensemble = False

  config.data = ml_collections.ConfigDict()
  config.data.name = ''
  config.data.num_classes = 2
  config.data.batch_size = 64
  # Number of slices into which train and val will be split.
  config.data.num_splits = 5
  # Ratio of splits that will be considered out-of-distribution from each
  # combination, e.g. when num_splits == 5 and ood_ratio == 0.4, 2 out 5
  # slices will be excluded for every combination of training data.
  config.data.ood_ratio = 0.4
  # Indices of data splits to include in training. All by default.
  config.data.included_splits_idx = (0, 1, 2, 3, 4)

  config.training = ml_collections.ConfigDict()
  config.training.num_epochs = 10
  config.training.save_model_checkpoints = True
  # TODO(jihyeonlee): Allow user to specify early stopping patience.
  # When True, stops training when val AUC does not improve after 3 epochs.
  config.training.early_stopping = True

  config.optimizer = ml_collections.ConfigDict()
  config.optimizer.learning_rate = 1e-4
  config.optimizer.type = 'adam'

  config.model = ml_collections.ConfigDict()
  config.model.name = ''
  config.model.hidden_sizes = None

  return config

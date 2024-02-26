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
  if config.bias_percentile_threshold < 0 or config.bias_percentile_threshold > 100:
    raise ValueError(
        'config.bias_percentile_threshold must be between 0 and 100.')
  if config.bias_value_threshold and (config.bias_value_threshold < 0. or
                                      config.bias_value_threshold > 1.):
    raise ValueError('config.bias_value_threshold must be between 0. and 1.')


def get_data_config():
  """Get dataset config."""
  config = ml_collections.ConfigDict()
  config.name = ''
  config.num_classes = 2
  config.batch_size = 64
  # Number of slices into which train and val will be split.
  config.num_splits = 5
  # Ratio of splits that will be considered out-of-distribution from each
  # combination, e.g. when num_splits == 5 and ood_ratio == 0.4, 2 out 5
  # slices will be excluded for every combination of training data.
  config.ood_ratio = 0.4
  # Indices of data splits to include in training. All by default.
  config.included_splits_idx = (0, 1, 2, 3, 4)
  # Subgroup IDs. Specify them in an experiment config. For example, for
  # Waterbirds, the subgroup IDs might be ('0_1', '1_0') for landbirds on water
  # and waterbirds on land, respectively.
  config.subgroup_ids = ()
  # Subgroup proportions. Specify them in an experiment config. For example, for
  # Waterbirds, the subgroup proportions might be (0.05, 0.05), meaning each
  # subgroup will represent 5% of the dataset.
  config.subgroup_proportions = ()

  # Proportion of training set to sample initially. Rest is considered the pool
  # for active sampling.
  config.initial_sample_proportion = 0.5
  # Whether to use data splits for the creation of an ensemble or filtering.
  # When filtering is used instead of creating combinations of splits each
  # model is trained on a random subsample of the dataset. Split guarantees
  # each point to be in the exact number of splits defined by the ood ratio.
  # Filtering only guarantees this in expectation.
  config.use_splits = True
  config.use_filtering = False

  # The following arguments are only used when use_filtering=True
  # The sum of split seed and split_id form the sampling seed for subset
  # selection.
  config.split_seed = 0
  config.split_id = 0
  # Seed for initial sample selection when filitering is used.
  config.initial_sample_seed = 0
  # Proportion of split to size of initial training set (similar to ood_ratio
  # but can have arbitrary value between 0 and 1.)
  config.split_proportion = 0.7

  # Leave one out training
  config.loo_id = ''
  config.loo_training = False
  # Correlation strength of the minority vs majority group. This is equivalent
  # to the proportion of majority group examples in the data.
  config.corr_strength = 0.95
  # Specify whether to load small images or not.
  config.load_small_images = True
  return config


def get_training_config():
  """Get training config."""
  config = ml_collections.ConfigDict()
  config.num_epochs = 60
  config.save_model_checkpoints = False
  config.save_best_model = True
  # TODO(jihyeonlee): Allow user to specify early stopping patience.
  # When True, stops training when val AUC does not improve after 3 epochs.
  config.early_stopping = False
  config.log_to_xm = True
  return config


def get_optimizer_config():
  """Get optimizer config."""
  config = ml_collections.ConfigDict()
  # With Adam, use lr 1e-4.
  config.learning_rate = 1e-5
  config.type = 'sgd'
  return config


def get_model_config():
  """Get model config."""
  config = ml_collections.ConfigDict()
  config.name = ''
  config.hidden_sizes = None
  config.num_channels = 3
  config.l2_regularization_factor = 0.5
  # TODO(jihyeonlee): Debug why loading ImageNet weights causes model train/val
  # acc to decrease rather than improving over time.
  config.load_pretrained_weights = False
  # If ResNet, use config.use_pytorch_style_resnet will make adjustments to the
  # TF ResNet model to match the PyTorch implementation, such as using He normal
  # initialization for convolution layers.
  config.use_pytorch_style_resnet = True
  return config


def get_active_sampling_config():
  """Get model config."""
  config = ml_collections.ConfigDict()
  config.sampling_score = 'ensemble_uncertainty'
  config.num_samples_per_round = 50
  return config


def get_reweighting_config():
  """Get config for performing reweighting during training."""
  config = ml_collections.ConfigDict()
  config.do_reweighting = False
  config.signal = 'bias'  # Options are bias, error.
  # Weight that underrepresented group examples will receive. Between 0 and 1.
  config.lambda_value = 0.
  config.error_percentile_threshold = 0.2
  return config


def get_upsampling_config():
  """Get config for performing upsampling during training."""
  config = ml_collections.ConfigDict()
  config.do_upsampling = False
  # TODO(jihyeonlee): Add support for upsampling signal being bias or error.
  config.signal = 'subgroup_label'
  # Lambda determines how much each example of the group to be upsampled is
  # repeated in the dataset.
  config.lambda_value = 60
  return config


def get_evaluation_config():
  """Get config for performing introspection signal computation."""
  config = ml_collections.ConfigDict()
  # A iterable tuple of epochs to compute checkpoint for.
  config.signal_ckpt_epochs = ()
  # Number of training epochs to check for computing introspection signals.
  # Used if `signal_ckpt_epochs` is empty. If 0 then compute signals based
  # on the latest epoch using `tf.train.latest_checkpoint`.
  config.num_signal_ckpts = 0
  return config


def get_config() -> ml_collections.ConfigDict:
  """Get config."""
  config = ml_collections.ConfigDict()

  config.output_dir = ''
  config.save_dir = ''
  config.ids_dir = ''

  config.eval_splits = ('val', 'test')

  # Number of rounds of active sampling to conduct.
  config.num_rounds = 4

  # Threshold to generate bias labels. Can be specified as percentile or value.
  config.bias_percentile_threshold = 80
  config.bias_value_threshold = None
  config.tracin_percentile_threshold = 80
  config.tracin_value_threshold = None
  config.save_bias_table = True
  # Path to existing bias table to use in training the bias head. If
  # unspecified, generates new one.
  config.path_to_existing_bias_table = ''
  # The signal used to train the bias head.
  config.bias_head_prediction_signal = 'bias_label'

  config.train_bias = True
  # When True, trains the stage 2 model (stage 1 is calculating bias table)
  # as an ensemble of models. When True and only a single model is being
  # trained, trains that model as an ensemble.
  config.train_stage_2_as_ensemble = True

  # Combo index to train
  config.combo_index = 0

  # Round of acitve sampling being performed
  config.round_idx = -1

  # Whether to generate bias table (from stage one models) or prediction table
  #    (from stage two models)
  config.generate_bias_table = True

  # Whether or not to do introspective training
  config.introspective_training = True

  # Whether to save the ids used during training (for bias estimation)
  config.save_train_ids = True

  config.data = get_data_config()
  config.training = get_training_config()
  config.optimizer = get_optimizer_config()
  config.model = get_model_config()
  config.active_sampling = get_active_sampling_config()
  config.reweighting = get_reweighting_config()
  config.upsampling = get_upsampling_config()
  config.eval = get_evaluation_config()
  return config

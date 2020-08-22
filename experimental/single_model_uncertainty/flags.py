# coding=utf-8
# Copyright 2020 The Uncertainty Baselines Authors.
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

# Lint as: python3
"""Common flags."""

from typing import Any, Dict, List
from absl import flags

from uncertainty_baselines.datasets import datasets
from uncertainty_baselines.models import models


FLAGS = flags.FLAGS


def serialize_flags(flag_list: Dict[str, Any]) -> str:
  string = ''
  for flag_name, flag_value in flag_list.items():
    string += '--{}={}\n'.format(flag_name, flag_value)
  # Remove the final trailing newline.
  return string[:-1]


def define_flags() -> List[str]:
  """Define common flags."""
  predefined_flags = set(FLAGS)

  flags.DEFINE_string('experiment_name', None, 'Name of this experiment.')

  # TPU Job flags.
  flags.DEFINE_string('master', '', 'Name of the TPU to use.')
  flags.DEFINE_enum(
      'mode',
      'train_and_eval',
      ['train', 'eval', 'train_and_eval'],
      'Whether to execute train and/or eval.')
  flags.DEFINE_bool('run_ood', False, 'Whether to run OOD jobs with eval job.')
  flags.DEFINE_bool('use_tpu', False, 'Whether to run on CPU or TPU.')

  # Train/eval loop flags.
  flags.DEFINE_integer(
      'checkpoint_step', -1, 'Step of the checkpoint to restore from.')
  flags.DEFINE_enum(
      'dataset_name',
      None,
      datasets.get_dataset_names(),
      'Name of the dataset to use.')
  flags.DEFINE_enum(
      'ood_dataset_name',
      None,
      datasets.get_dataset_names(),
      'Name of the OOD dataset to use for evaluation.')
  flags.DEFINE_integer(
      'eval_frequency',
      None,
      'How many steps between evaluating on the (validation and) test set.')
  flags.DEFINE_string('model_dir', None, 'Base output directory.')
  flags.DEFINE_enum(
      'model_name',
      None,
      models.get_model_names(),
      'Name of the model to use.')
  flags.DEFINE_integer(
      'log_frequency',
      100,
      'How many steps between logging the metrics.')
  flags.DEFINE_integer('train_steps', None, 'How many steps to train for.')

  # Hyperparamater flags.
  flags.DEFINE_integer('batch_size', None, 'Training batch size.')
  flags.DEFINE_integer('eval_batch_size', None, 'Validation/test batch size.')
  flags.DEFINE_float('learning_rate', None, 'Learning rate.')
  flags.DEFINE_string(
      'learning_rate_schedule',
      'constant',
      'Learning rate schedule to use.')
  flags.DEFINE_integer('schedule_hparams_warmup_epochs', 1,
                       'Number of epochs for a linear warmup to the initial '
                       'learning rate. Use 0 to do no warmup.')
  flags.DEFINE_float('schedule_hparams_decay_ratio', 0.2,
                     'Amount to decay learning rate.')
  flags.DEFINE_list('schedule_hparams_decay_epochs', ['60', '120', '160'],
                    'Epochs to decay learning rate by.')
  flags.DEFINE_string('optimizer', 'adam', 'Optimizer to use.')
  flags.DEFINE_float('optimizer_hparams_momentum', 0.9, 'SGD momentum.')
  flags.DEFINE_float('optimizer_hparams_beta_1', 0.9, 'Adam beta_1.')
  flags.DEFINE_float('optimizer_hparams_beta_2', 0.999, 'Adam beta_2.')
  flags.DEFINE_float('optimizer_hparams_epsilon', 1e-7, 'Adam epsilon.')
  flags.DEFINE_float('weight_decay', 0.0, 'Weight decay for optimizer.')
  flags.DEFINE_float('l2_regularization', 1e-4, 'L2 regularization for models.')
  flags.DEFINE_integer('seed', 42, 'Random seed.')
  flags.DEFINE_float(
      'validation_percent',
      0.0,
      'Percent of training data to hold out and use as a validation set.')
  flags.DEFINE_integer(
      'shuffle_buffer_size', 16384, 'Dataset shuffle buffer size.')

  # Model flags, Wide Resnet
  flags.DEFINE_integer('wide_resnet_depth', 28,
                       'Depth of wide resnet model.')
  flags.DEFINE_integer('wide_resnet_width_multiplier', 10,
                       'Width multiplier for wide resnet model.')

  # Flags relating to genomics_cnn model
  flags.DEFINE_integer('len_seqs', 250,
                       'Sequence length, only used for genomics dataset.')
  flags.DEFINE_integer('num_motifs', 1024,
                       'Number of motifs, only used for the genomics dataset.')
  flags.DEFINE_integer('len_motifs', 20,
                       'Length of motifs, only used for the genomics dataset.')
  flags.DEFINE_integer('num_denses', 128,
                       'Number of denses, only used for the genomics dataset.')

  # Flags relating to SNGP model
  flags.DEFINE_float('dropout_rate', 0.1, 'Dropout rate for dropout layers.')
  flags.DEFINE_bool(
      'before_conv_dropout', False,
      'Whether to use filter wise dropout before convolutionary layers. ')
  flags.DEFINE_bool(
      'use_mc_dropout', False,
      'Whether to use Monte Carlo dropout for the hidden layers.')
  flags.DEFINE_bool('use_spec_norm', False,
                    'Whether to apply spectral normalization.')
  flags.DEFINE_bool('use_gp_layer', False,
                    'Whether to use Gaussian process as the output layer.')
  # Model flags, Spectral Normalization.
  flags.DEFINE_integer(
      'spec_norm_iteration', 1,
      'Number of power iterations to perform for estimating '
      'the spectral norm of weight matrices.')
  flags.DEFINE_float('spec_norm_bound', 6.,
                     'Upper bound to spectral norm of weight matrices.')

  # Model flags, Gaussian Process layer.
  flags.DEFINE_float('gp_bias', 0., 'The bias term for GP layer.')
  flags.DEFINE_float(
      'gp_scale', 2.,
      'The length-scale parameter for the RBF kernel of the GP layer.')
  flags.DEFINE_integer(
      'gp_input_dim', 128,
      'The dimension to reduce the neural network input to for the GP layer '
      '(via random Gaussian projection which preserves distance by the '
      ' Johnson-Lindenstrauss lemma). If -1 the no dimension reduction.')
  flags.DEFINE_integer(
      'gp_hidden_dim', 1024,
      'The hidden dimension of the GP layer, which corresponds to the number '
      'of random features used to for the approximation ')
  flags.DEFINE_bool(
      'gp_input_normalization', True,
      'Whether to normalize the input using LayerNorm for GP layer.'
      'This is similar to automatic relevance determination (ARD) in the '
      'classic GP learning.')
  flags.DEFINE_float(
      'gp_cov_ridge_penalty', 1e-3,
      'The Ridge penalty parameter for GP posterior covariance.')
  flags.DEFINE_float(
      'gp_cov_discount_factor', 0.999,
      'The discount factor to compute the moving average of '
      'precision matrix.')
  flags.DEFINE_float(
      'gp_mean_field_factor', 0.001,
      'The tunable multiplicative factor used in the mean-field approximation '
      'for the posterior mean of softmax Gaussian process. If -1 then use '
      'posterior mode instead of posterior mean. See [2] for detail.')

  flags.mark_flag_as_required('dataset_name')
  flags.mark_flag_as_required('experiment_name')
  flags.mark_flag_as_required('model_name')

  # Flags relating to OOD metrics
  flags.DEFINE_list('sensitivity_thresholds',
                    ['0.1', '0.8', '10'],
                    'List of sensitivities at which to calculate specificity.'
                    ' The list should contains '
                    '[lower bound, upper bound, num_elements]')
  flags.DEFINE_list('specificity_thresholds',
                    ['0.1', '0.8', '10'],
                    'List of specificities at which to calculate sensitivity.'
                    ' The list should contains '
                    '[lower bound, upper bound, num_elements]')
  flags.DEFINE_list('precision_thresholds',
                    ['0.1', '0.8', '10'],
                    'List of precisions at which to calculate recall.'
                    ' The list should contains '
                    '[lower bound, upper bound, num_elements]')
  flags.DEFINE_list('recall_thresholds',
                    ['0.1', '0.8', '10'],
                    'List of recalls at which to calculate precision.'
                    ' The list should contains '
                    '[lower bound, upper bound, num_elements]')

  all_flags = set(FLAGS)
  program_flag_names = sorted(list(all_flags - predefined_flags))
  return program_flag_names

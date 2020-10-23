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

  # Flags relating to setting up the job.
  flags.DEFINE_string('tpu', None, 'Name of the TPU to use.')
  flags.DEFINE_enum(
      'mode',
      'train_and_eval',
      ['train', 'eval', 'train_and_eval'],
      'Whether to execute train and/or eval.')
  flags.DEFINE_integer(
      'num_cores', None, 'How many TPU cores or GPUs the job is running on.')
  flags.DEFINE_bool('use_cpu', False, 'Whether to run on CPU.')
  flags.DEFINE_bool('use_gpu', False, 'Whether to run on GPU or TPU.')

  # Flags relating to the training/eval loop.
  flags.DEFINE_integer(
      'checkpoint_step', -1, 'Step of the checkpoint to restore from.')
  flags.DEFINE_enum(
      'dataset_name',
      None,
      datasets.get_dataset_names(),
      'Name of the dataset to use.')
  flags.DEFINE_integer(
      'eval_frequency',
      None,
      'How many steps between evaluating on the (validation and) test set.')
  flags.DEFINE_string(
      'data_dir',
      None,
      'Directory where the dataset is stored to be loaded via tfds.load. '
      'Optional, useful for loading datasets stored on GCS.')
  flags.DEFINE_string('output_dir', None, 'Base output directory.')
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

  # Hyperparamter flags.
  flags.DEFINE_integer('batch_size', None, 'Training batch size.')
  flags.DEFINE_integer('eval_batch_size', None, 'Validation/test batch size.')
  flags.DEFINE_float('learning_rate', None, 'Learning rate.')
  flags.DEFINE_string(
      'learning_rate_schedule',
      'constant',
      'Learning rate schedule to use. See `get_learning_rate_schedule` in '
      'uncertainty_baselines/optimizers.py for more details.')
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
  flags.DEFINE_float('weight_decay', 0.0, 'Weight decay.')
  flags.DEFINE_integer('seed', 42, 'Random seed.')
  flags.DEFINE_float(
      'validation_percent',
      0.0,
      'Percent of training data to hold out and use as a validation set.')
  flags.DEFINE_integer(
      'shuffle_buffer_size', 16384, 'Dataset shuffle buffer size.')

  # Model flags for the Wide Resnet model.
  flags.DEFINE_integer('wide_resnet_depth', 28,
                       'Depth of wide resnet model.')
  flags.DEFINE_integer('wide_resnet_width_multiplier', 10,
                       'Width multiplier for wide resnet model.')
  flags.DEFINE_integer('wide_resnet_version', 2,
                       'Version of the wide resnet model.')
  flags.DEFINE_float('l2_regularization', 1e-4, 'L2 regularization for models.')

  # Flags relating to the genomics_cnn model.
  flags.DEFINE_integer('num_motifs', 1024,
                       'Number of motifs, only used for the genomics dataset.')
  flags.DEFINE_integer('len_motifs', 20,
                       'Length of motifs, only used for the genomics dataset.')
  flags.DEFINE_integer('num_denses', 128,
                       'Number of denses, only used for the genomics dataset.')

  flags.mark_flag_as_required('dataset_name')
  flags.mark_flag_as_required('experiment_name')
  flags.mark_flag_as_required('model_name')
  flags.mark_flag_as_required('batch_size')

  all_flags = set(FLAGS)
  program_flag_names = sorted(list(all_flags - predefined_flags))
  return program_flag_names

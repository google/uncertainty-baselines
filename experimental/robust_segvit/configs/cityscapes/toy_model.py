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

# pylint: disable=line-too-long
r"""Train toy segmenter model on cityscapes.

"""
# pylint: enable=line-too-long

import ml_collections


batch_size = 128
_CITYSCAPES_TRAIN_SIZE_SPLIT = 146

# Model spec.
STRIDE = 4
mlp_dim = 2
num_heads = 1
num_layers = 1
hidden_size = 1
target_size = (128, 128)


def get_config(runlocal=''):
  """Returns the configuration for Cityscapes segmentation."""

  runlocal = bool(runlocal)

  config = ml_collections.ConfigDict()
  config.experiment_name = 'cityscapes_segmenter_toy_model'

  # Dataset.
  config.dataset_name = 'cityscapes'
  config.dataset_configs = ml_collections.ConfigDict()
  config.dataset_configs.target_size = target_size
  config.dataset_configs.train_split = 'train[:5%]'
  config.dataset_configs.dataset_name = ''  # name of ood dataset to evaluate

  # Model.
  config.model_name = 'segvit'
  config.model = ml_collections.ConfigDict()

  config.model.patches = ml_collections.ConfigDict()
  config.model.patches.size = (STRIDE, STRIDE)

  config.model.backbone = ml_collections.ConfigDict()
  config.model.backbone.type = 'vit'
  config.model.backbone.mlp_dim = mlp_dim
  config.model.backbone.num_heads = num_heads
  config.model.backbone.num_layers = num_layers
  config.model.backbone.hidden_size = hidden_size
  config.model.backbone.dropout_rate = 0.1
  config.model.backbone.attention_dropout_rate = 0.0
  config.model.backbone.classifier = 'gap'

  # Decoder
  config.model.decoder = ml_collections.ConfigDict()
  config.model.decoder.type = 'linear'

  # Training.
  config.trainer_name = 'segvit_trainer'
  config.optimizer = 'adam'
  config.optimizer_configs = ml_collections.ConfigDict()
  config.l2_decay_factor = 0.0
  config.max_grad_norm = 1.0
  config.label_smoothing = None
  config.num_training_epochs = ml_collections.FieldReference(2)
  config.batch_size = batch_size
  config.rng_seed = 0
  config.focal_loss_gamma = 0.0

  # Learning rate.
  config.steps_per_epoch = _CITYSCAPES_TRAIN_SIZE_SPLIT // config.get_ref(
      'batch_size')
  # setting 'steps_per_cycle' to total_steps basically means non-cycling cosine.
  config.lr_configs = ml_collections.ConfigDict()
  config.lr_configs.learning_rate_schedule = 'compound'
  config.lr_configs.factors = 'constant * cosine_decay * linear_warmup'
  config.lr_configs.warmup_steps = 0
  config.lr_configs.steps_per_cycle = config.get_ref(
      'num_training_epochs') * config.get_ref('steps_per_epoch')
  config.lr_configs.base_learning_rate = 1e-4

  # model and data dtype
  config.model_dtype_str = 'float32'
  config.data_dtype_str = 'float32'

  # init not included

  # Logging.
  config.write_summary = True
  config.write_xm_measurements = True  # write XM measurements
  config.xprof = False  # Profile using xprof.
  config.checkpoint = True  # Do checkpointing.
  config.checkpoint_steps = 5 * config.get_ref('steps_per_epoch')

  config.debug_train = False  # Debug mode during training.
  config.debug_eval = False  # Debug mode during eval.
  config.log_eval_steps = 1 * config.get_ref('steps_per_epoch')

  # Evaluation.
  config.eval_mode = False
  config.eval_configs = ml_collections.ConfigDict()
  config.eval_configs.mode = 'standard'
  config.eval_covariate_shift = True
  config.eval_label_shift = True

  if runlocal:
    config.count_flops = False

  return config


def get_sweep(hyper):
  return hyper.product([])

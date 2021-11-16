# coding=utf-8
# Copyright 2021 The Uncertainty Baselines Authors.
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
r"""Segmenter + cityscapes.

"""
# pylint: enable=line-too-long

import ml_collections
#import get_fewshot  # local file import

_CITYSCAPES_TRAIN_SIZE = 2975
DEBUG = True


def get_config():
  """Config for training a patch-transformer on JFT."""
  config = ml_collections.ConfigDict()

  config.experiment_name = 'cityscapes_segvit_ub'

  config.dataset_name = 'cityscapes'
  config.dataset_configs = ml_collections.ConfigDict()
  config.dataset_configs.target_size = (512, 512)

  # config following scenic
  config.num_classes = 19

  config.patches = ml_collections.ConfigDict()
  config.patches.size = [4, 4]

  config.backbone_configs = ml_collections.ConfigDict()
  config.backbone_configs.type = 'vit'
  config.backbone_configs.attention_dropout_rate = 0.
  config.backbone_configs.dropout_rate = 0.
  config.backbone_configs.classifier = 'gap'

  if DEBUG:
    config.backbone_configs.mlp_dim = 2
    config.backbone_configs.num_heads = 1
    config.backbone_configs.num_layers = 1
    config.backbone_configs.hidden_size = 1
  else:
    config.backbone_configs.mlp_dim = 3072
    config.backbone_configs.num_heads = 12
    config.backbone_configs.num_layers = 12
    config.backbone_configs.hidden_size = 768

  config.decoder_configs = ml_collections.ConfigDict()
  config.decoder_configs.type = 'linear'

  # training
  config.optimizer = 'adam'
  config.optimizer_configs = ml_collections.ConfigDict()
  config.l2_decay_factor = 0.0
  config.max_grad_norm = 1.0
  config.label_smoothing = None
  num_training_epochs = 1 # ml_collections.FieldReference(100)
  config.num_training_epochs = num_training_epochs
  config.batch_size = 1
  config.rng_seed = 0
  config.focal_loss_gamma = 0.0

  # learning rate
  steps_per_epoch = _CITYSCAPES_TRAIN_SIZE // config.batch_size
  # setting 'steps_per_cycle' to total_steps basically means non-cycling cosine.
  config.lr_configs = ml_collections.ConfigDict()
  config.lr_configs.learning_rate_schedule = 'compound'
  config.lr_configs.factors = 'constant * cosine_decay * linear_warmup'
  config.lr_configs.warmup_steps = 1 * steps_per_epoch
  config.lr_configs.steps_per_cycle = num_training_epochs * steps_per_epoch
  config.lr_configs.base_learning_rate = 1e-4

  # model and data dtype
  config.model_dtype_str = 'float32'
  config.data_dtype_str = 'float32'

  #logging
  config.write_summary = True  # write TB and/or XM summary
  config.write_xm_measurements = True  # write XM measurements
  #config.xprof = False  # Profile using xprof
  config.checkpoint = True  # do checkpointing
  config.checkpoint_steps = 5 * steps_per_epoch

  config.debug_train = True  # debug mode during training
  config.debug_eval = True  # debug mode during eval
  config.log_eval_steps = 200

  # extra
  config.args = {}

  return config


def get_sweep(hyper):
  return hyper.product([])

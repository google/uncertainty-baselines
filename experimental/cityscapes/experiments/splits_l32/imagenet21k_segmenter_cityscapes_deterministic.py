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

include wandb parameters
"""
# pylint: enable=line-too-long

import ml_collections
#import get_fewshot  # local file import

_CITYSCAPES_TRAIN_SIZE = 2975
DEBUG = 0

STRIDE = 32
target_size=(512, 512)

batch_size = 8
number_train_examples_debug = 2975
num_training_epochs = ml_collections.FieldReference(100)

mlp_dim = 4096
num_heads = 16
num_layers = 24
hidden_size = 1024
train_split = 'train[:1%]'

LOAD_PRETRAINED_BACKBONE=True

if DEBUG ==1:
  STRIDE = 4
  target_size = (128, 128)

  batch_size = 1
  number_train_examples_debug = 29
  num_training_epochs = ml_collections.FieldReference(1)

  mlp_dim = 2
  num_heads = 1
  num_layers = 1
  hidden_size = 1
  train_split = 'train[:1%]'


def get_config():
  """Config for training a patch-transformer on JFT."""
  config = ml_collections.ConfigDict()

  config.experiment_name = 'cityscapes_segvit_ub'

  config.dataset_name = 'cityscapes'
  config.dataset_configs = ml_collections.ConfigDict()
  config.dataset_configs.target_size = target_size
  config.dataset_configs.train_split = train_split
  # flags to debug scenic
  config.dataset_configs.number_train_examples_debug = number_train_examples_debug
  #config.dataset_configs.number_eval_examples_debug = number_train_examples_debug

  # config following scenic
  config.num_classes = 19

  config.patches = ml_collections.ConfigDict()
  config.patches.size = (STRIDE, STRIDE)

  config.backbone_configs = ml_collections.ConfigDict()
  config.backbone_configs.type = 'vit'
  config.backbone_configs.attention_dropout_rate = 0.
  config.backbone_configs.dropout_rate = 0.
  config.backbone_configs.classifier = 'gap'

  config.backbone_configs.mlp_dim = mlp_dim
  config.backbone_configs.num_heads = num_heads
  config.backbone_configs.num_layers = num_layers
  config.backbone_configs.hidden_size = hidden_size

  config.decoder_configs = ml_collections.ConfigDict()
  config.decoder_configs.type = 'linear'

  # training
  config.trainer_name = 'segvit_trainer'
  config.optimizer = 'adam'
  config.optimizer_configs = ml_collections.ConfigDict()
  config.l2_decay_factor = 0.0
  config.max_grad_norm = 1.0
  config.label_smoothing = None
  config.num_training_epochs = num_training_epochs
  config.batch_size = batch_size
  config.rng_seed = 0
  config.focal_loss_gamma = 0.0

  # learning rate
  # setting 'steps_per_cycle' to total_steps basically means non-cycling cosine.
  config.steps_per_epoch = config.dataset_configs.get_ref('number_train_examples_debug') // config.get_ref('batch_size')
  config.lr_configs = ml_collections.ConfigDict()
  config.lr_configs.learning_rate_schedule = 'compound'
  config.lr_configs.factors = 'constant * cosine_decay * linear_warmup'
  config.lr_configs.warmup_steps = 1 * config.get_ref('steps_per_epoch')
  config.lr_configs.steps_per_cycle = config.get_ref('num_training_epochs') * config.get_ref('steps_per_epoch')
  config.lr_configs.base_learning_rate = 1e-4

  # model and data dtype
  config.model_dtype_str = 'float32'
  config.data_dtype_str = 'float32'

  # load pretrained backbone
  config.upstream_model = 'deterministic'
  config.load_pretrained_backbone = LOAD_PRETRAINED_BACKBONE
  config.pretrained_backbone_configs = get_pretrained_backbone_config(config)

  #logging
  config.write_summary = True  # write TB and/or XM summary
  config.write_xm_measurements = True  # write XM measurements
  #config.xprof = False  # Profile using xprof
  config.checkpoint = True  # do checkpointing
  config.checkpoint_steps = 5 * config.get_ref('steps_per_epoch')

  config.debug_train = True  # debug mode during training
  config.debug_eval = True  # debug mode during eval
  config.log_eval_steps = 1 * config.get_ref('steps_per_epoch')

  # wabdb
  config.use_wandb=True
  config.wandb_dir= 'wandb'
  config.wandb_project = 'rdl-visual'  # Wandb project name.
  config.wandb_exp_name = None  # Give experiment a name.
  config.wandb_exp_group = None  # Give experiment a group name.

  config.early_stopping_patience = 3  # number of epochs to wait before stopping training
  return config


def get_pretrained_backbone_config(config):
  if not config.load_pretrained_backbone:
    return None
  pretrained_backbone_configs = ml_collections.ConfigDict()
  pretrained_backbone_configs.checkpoint_format = "ub"
  pretrained_backbone_configs.type = 'base'

  pretrained_backbone_configs.checkpoint_path = "gs://ub-checkpoints/ImageNet21k_ViT-L32/1/checkpoint.npz"
  pretrained_backbone_configs.checkpoint_cfg = "https://github.com/google/uncertainty-baselines/blob/4097549f62ca5e209c6f1ca244fe178b53b6cff4/baselines/jft/experiments/jft300m_vit_l32_finetune.py"

  return pretrained_backbone_configs


def get_sweep(hyper):
  return hyper.product([])

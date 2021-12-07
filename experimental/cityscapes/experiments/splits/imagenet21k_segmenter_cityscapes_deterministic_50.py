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
_CITYSCAPES_TRAIN_SIZE = 2975
DEBUG = 0

TRAIN_PROP=50

# we will have 4 version of train split
if TRAIN_PROP == 100:
  _CITYSCAPES_TRAIN_SIZE_SPLIT = _CITYSCAPES_TRAIN_SIZE
  train_split = 'train'
elif TRAIN_PROP == 75:
  _CITYSCAPES_TRAIN_SIZE_SPLIT = 2231
  train_split = 'train[:75%]'
elif TRAIN_PROP == 50:
  _CITYSCAPES_TRAIN_SIZE_SPLIT = 1488
  train_split = 'train[:50%]'
elif TRAIN_PROP == 25:
  _CITYSCAPES_TRAIN_SIZE_SPLIT = 744
  train_split = 'train[:25%]'
elif TRAIN_PROP == 10:
  _CITYSCAPES_TRAIN_SIZE_SPLIT = 298
  train_split = 'train[:10%]'

target_size = (512, 512)
LOAD_PRETRAINED_BACKBONE = True
PRETRAIN_BACKBONE_TYPE = 'base'

STRIDE=16
batch_size=8
num_training_epochs = 100  # ml_collections.FieldReference(100)
log_eval_steps = 200

mlp_dim = 3072
num_heads = 12
num_layers = 12
hidden_size = 768

if DEBUG ==5:
  number_train_examples_debug = 16

def get_config():
  """Config for cityscapes segmentation."""
  config = ml_collections.ConfigDict()

  config.experiment_name = 'cityscapes_segvit_ub_init'

  #dataset
  config.dataset_name = 'cityscapes'
  config.dataset_configs = ml_collections.ConfigDict()
  config.dataset_configs.target_size = target_size
  config.dataset_configs.train_split = train_split

  # flags to debug scenic on mac
  #config.dataset_configs.number_train_examples_debug = number_train_examples_debug
  #config.dataset_configs.number_eval_examples_debug = number_train_examples_debug

  # config following scenic
  # model
  config.model_name = 'segmenter_pretrained_mini'
  config.model = ml_collections.ConfigDict()

  config.patches = ml_collections.ConfigDict()
  config.patches.size = (STRIDE, STRIDE)

  config.backbone_configs = ml_collections.ConfigDict()
  config.backbone_configs.type = 'vit'
  config.backbone_configs.classifier = 'gap'
  #config.backbone_configs.grid_size
  config.backbone_configs.hidden_size = hidden_size
  #config.backbone_configs.patches
  #config.backbone_configs.representation_size = None

  config.backbone_configs.attention_dropout_rate = 0.
  config.backbone_configs.dropout_rate = 0.
  config.backbone_configs.mlp_dim = mlp_dim
  config.backbone_configs.num_heads = num_heads
  config.backbone_configs.num_layers = num_layers

  #decoder
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
  steps_per_epoch = _CITYSCAPES_TRAIN_SIZE_SPLIT // config.batch_size
  #steps_per_epoch = number_train_examples_debug // config.batch_size

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

  # load pretrained backbone
  config.load_pretrained_backbone = LOAD_PRETRAINED_BACKBONE
  config.pretrained_backbone_configs = get_pretrained_backbone_config(config)

  #logging
  config.write_summary = True  # write TB and/or XM summary
  config.write_xm_measurements = True  # write XM measurements
  #config.xprof = False  # Profile using xprof
  config.checkpoint = True  # do checkpointing
  config.checkpoint_steps = 5 * steps_per_epoch

  config.debug_train = True  # debug mode during training
  config.debug_eval = True  # debug mode during eval
  config.log_eval_steps = 1 * steps_per_epoch  #log_eval_steps  # 200

  # extra
  config.args = {}

  return config


def get_pretrained_backbone_config(config):
  if not config.load_pretrained_backbone:
    return None
  pretrained_backbone_configs = ml_collections.ConfigDict()
  pretrained_backbone_configs.checkpoint_format = "ub"
  pretrained_backbone_configs.type = PRETRAIN_BACKBONE_TYPE

  if PRETRAIN_BACKBONE_TYPE == 'base':
    pretrained_backbone_configs.checkpoint_path = "gs://ub-checkpoints/ImageNet21k_ViT-B16/ImagetNet21k_ViT-B:16_28592399.npz"
    pretrained_backbone_configs.checkpoint_cfg = "https://github.com/google/uncertainty-baselines/blob/main/baselines/jft/experiments/imagenet21k_vit_base16.py"
  elif PRETRAIN_BACKBONE_TYPE == 'gp':
    pretrained_backbone_configs.checkpoint_path = "gs://ub-checkpoints/ImageNet21k_ViT-B16-GP/ImageNet21k_ViT-B:16-GP_29240948.npz"
    pretrained_backbone_configs.checkpoint_cfg = "https://github.com/google/uncertainty-baselines/blob/main/baselines/jft/experiments/imagenet21k_vit_base16_sngp.py"
  else:
    raise NotImplementedError("")

  return pretrained_backbone_configs


def get_sweep(hyper):
  return hyper.product([])

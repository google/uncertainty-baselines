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

# pylint: disable=line-too-long
r"""Train toy model on toy ade20k dataset calling the robust_segvit codebase.

"""
# pylint: enable=line-too-long

import ml_collections

_CITYSCAPES_FINE_TRAIN_SIZE = 2975
_CITYSCAPES_COARSE_TRAIN_SIZE = 19998

_ADE20K_TRAIN_SIZE = 20210
_PASCAL_VOC_TRAIN_SIZE = 10582
_PASCAL_CONTEXT_TRAIN_SIZE = 4998

TRAIN_SIZES = {
    'cityscapes': _CITYSCAPES_FINE_TRAIN_SIZE,
    'ade20k': _ADE20K_TRAIN_SIZE,
    'ade20k_ind': _ADE20K_TRAIN_SIZE,
    'pascal_voc': _PASCAL_VOC_TRAIN_SIZE,
    'pascal_context': _PASCAL_CONTEXT_TRAIN_SIZE
}

# Model spec.
STRIDE = 4
mlp_dim = 2
num_heads = 1
num_layers = 1
hidden_size = 1
target_size = (512, 512)

TRAIN_SAMPLES = 32


def get_config(runlocal=''):
  """Returns the configuration for ADE20k_ind segmentation."""

  runlocal = bool(runlocal)

  config = ml_collections.ConfigDict()
  config.experiment_name = 'ade20k_segmenter_ind_toy_model'

  # Dataset.
  config.dataset_name = 'robust_segvit_segmentation'
  config.dataset_configs = ml_collections.ConfigDict()
  config.dataset_configs.target_size = target_size
  config.dataset_configs.train_target_size = config.dataset_configs.get_ref(
      'target_size')
  config.dataset_configs.denoise = None
  config.dataset_configs.use_timestep = 0
  config.dataset_configs.train_split = 'train'
  config.dataset_configs.validation_split = 'validation'
  config.dataset_configs.name = 'ade20k_ind'
  config.dataset_configs.dataset_name = ''  # ood name flag to write in eval.

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
  config.model.backbone.classifier = 'token'

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
  config.batch_size = 32
  config.rng_seed = 0
  config.focal_loss_gamma = 0.0

  # Learning rate.
  config.num_train_examples = TRAIN_SIZES.get(config.dataset_configs.name)
  config.steps_per_epoch = config.get_ref(
      'num_train_examples') // config.get_ref('batch_size')
  # setting 'steps_per_cycle' to total_steps basically means non-cycling cosine.
  config.lr_configs = ml_collections.ConfigDict()
  config.lr_configs.learning_rate_schedule = 'compound'
  config.lr_configs.factors = 'constant * cosine_decay * linear_warmup'
  config.lr_configs.warmup_steps = 0
  config.lr_configs.steps_per_cycle = config.get_ref(
      'num_training_epochs') * config.get_ref('steps_per_epoch')
  config.lr_configs.base_learning_rate = 5e-4

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
  config.eval_covariate_shift = False
  config.eval_label_shift = False

  if runlocal:
    config.count_flops = False
    config.batch_size = 8
    config.dataset_configs.train_split = f'train[:{TRAIN_SAMPLES}]'
    config.dataset_configs.validation_split = f'validation[:{TRAIN_SAMPLES}]'
    config.num_train_examples = TRAIN_SAMPLES
  return config


def get_sweep(hyper):
  return hyper.product([])

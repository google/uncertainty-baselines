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
r"""Finetune a pretrained ViT-L/32 on EyePACS.

This config is used for models pretrained on either JFT-300M or ImageNet-21K.

"""
# pylint: enable=line-too-long

import ml_collections


def get_config():
  """Config for finetuning."""
  config = ml_collections.ConfigDict()

  # model_init should be modified per experiment
  config.model_init = '/path/to/pretrained_model_ckpt.npz'
  config.data_dir = 'gs://ub-data/retinopathy'
  config.num_classes = 2

  # 512 fits in memory on TPUv3-64, but we use TPUv3-8 for external reasons.
  config.batch_size = 128  # using TPUv3-8
  config.total_steps = 10_000  # set in sweep

  pp_input_res = 512
  pp_common = f'diabetic_retinopathy_preprocess({pp_input_res})'
  pp_common += f'|onehot({config.num_classes})'
  config.pp_train = pp_common
  config.pp_eval = pp_common
  config.shuffle_buffer_size = 15_000  # Per host, so small-ish is ok.

  config.log_training_steps = 100
  config.log_eval_steps = 1000
  config.checkpoint_steps = 5000
  config.checkpoint_timeout = 1
  config.use_validation = True
  config.use_test = True

  config.prefetch_to_device = 2
  config.trial = 0

  # OOD evaluation
  config.distribution_shift = 'aptos'  # set in sweep

  # External Wandb usage
  config.use_wandb = False
  config.wandb_dir = 'wandb'  # Directory where wandb logs go.
  config.project = 'ub-debug'  # Wandb project name.
  config.exp_name = None  # Give experiment a name.
  config.exp_group = None  # Give experiment a group name.

  # Model section
  config.model = ml_collections.ConfigDict()
  config.model.patches = ml_collections.ConfigDict()
  config.model.patches.size = [32, 32]
  config.model.hidden_size = 1024
  config.model.transformer = ml_collections.ConfigDict()
  config.model.transformer.mlp_dim = 4096
  config.model.transformer.num_heads = 16
  config.model.transformer.num_layers = 24
  config.model.transformer.attention_dropout_rate = 0.
  config.model.transformer.dropout_rate = 0.
  config.model.classifier = 'token'
  # This is "no head" fine-tuning, which we use by default.
  config.model.representation_size = None

  # Optimizer section
  config.optim_name = 'Momentum'
  config.optim = ml_collections.ConfigDict()
  config.grad_clip_norm = 1.0
  config.weight_decay = None
  config.loss = 'softmax_xent'

  config.lr = ml_collections.ConfigDict()
  config.lr.base = 0.01  # set in sweep
  config.lr.warmup_steps = 500  # set in sweep
  config.lr.decay_type = 'cosine'
  return config


def get_sweep(hyper):
  """Sweeps over hyperparameters."""
  # Try sweep settings based on CIFAR finetuning sweep or older sweeps on
  # ViT-B/16 for EyePACS.
  step_sweep = hyper.zipit([
      # hyper.sweep('config.total_steps',
      #             [int(40_000 * x) for x in [0.5, 1.0, 1.5, 2.0]]),
      # hyper.sweep('config.lr.warmup_steps',
      #             [int(2000 * x) for x in [0.5, 1.0, 1.5, 2.0]]),
      hyper.sweep('config.total_steps',
                  [10_000, 15_000, 20_000, 25_000]),
      hyper.sweep('config.lr.warmup_steps',
                  [500, 750, 1000, 1250]),
  ])
  return hyper.product([
      # hyper.sweep('config.lr.base', [0.03/4, 0.01/4, 0.003/4, 0.001/4]),
      hyper.sweep('config.lr.base', [0.05, 0.03, 0.01, 0.005]),
      step_sweep,
      hyper.sweep('config.lr.decay_type', ['cosine', 'linear']),
      hyper.sweep('config.grad_clip_norm', [1.0, 2.5]),
      # Sweep on just APTOS as Severity has similar in-dist results.
      hyper.sweep('config.distribution_shift', ['aptos']),
      # hyper.sweep('config.distribution_shift', ['aptos', 'severity']),
  ])

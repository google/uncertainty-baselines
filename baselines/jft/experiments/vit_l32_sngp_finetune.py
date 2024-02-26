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
r"""Finetune a pretrained ViT-L/32-SNGP on CIFAR-10/100 and ImageNet.

This config is used for models pretrained on either JFT-300M or ImageNet-21K.

"""
# pylint: enable=line-too-long

import ml_collections
from experiments import sweep_utils  # local file import from baselines.jft


def get_config():
  """Config for finetuning."""
  config = ml_collections.ConfigDict()

  config.model_init = ''  # set in sweep
  config.dataset = ''  # set in sweep
  config.data_dir = None
  config.test_split = ''  # set in sweep
  config.val_split = ''  # set in sweep
  config.train_split = ''  # set in sweep
  config.num_classes = None  # set in sweep

  config.batch_size = 512
  config.total_steps = None  # set in sweep

  config.pp_train = ''  # set in sweep
  config.pp_eval = ''  # set in sweep
  config.shuffle_buffer_size = 50_000  # Per host, so small-ish is ok.

  config.log_training_steps = 100
  config.log_eval_steps = 1000
  config.checkpoint_steps = 5000
  config.checkpoint_timeout = 1

  config.prefetch_to_device = 2
  config.trial = 0

  # Subpopulation shift evaluation. Parameters set in the sweep. If
  # `config.subpopl_cifar_data_file` is None, this evaluation is skipped.
  config.subpopl_cifar_data_file = None
  config.pp_eval_subpopl_cifar = None

  # OOD evaluation. They're all set in the sweep.
  config.ood_datasets = []
  config.ood_num_classes = []
  config.ood_split = ''
  config.ood_methods = []
  config.pp_eval_ood = []
  config.eval_on_cifar_10h = False
  config.pp_eval_cifar_10h = ''
  config.eval_on_imagenet_real = False
  config.pp_eval_imagenet_real = ''

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

  # Gaussian process layer section
  config.gp_layer = ml_collections.ConfigDict()
  config.gp_layer.ridge_penalty = 1.
  # Disable momentum in order to use exact covariance update for finetuning.
  config.gp_layer.covmat_momentum = -1.  # Disable to allow exact cov update.
  config.gp_layer.mean_field_factor = 5.

  # Optimizer section
  config.optim_name = 'Momentum'
  config.optim = ml_collections.ConfigDict()
  config.grad_clip_norm = 1.
  config.weight_decay = None  # No explicit weight decay
  config.loss = 'softmax_xent'

  config.lr = ml_collections.ConfigDict()
  config.lr.base = 0.001
  config.lr.warmup_steps = 500
  config.lr.decay_type = 'cosine'
  return config


def get_sweep(hyper):
  """Sweeps over datasets."""
  checkpoints = ['/path/to/pretrained_model_ckpt.npz']
  checkpoints = [checkpoints[0]]

  cifar10_sweep = hyper.product([
      hyper.chainit([
          hyper.product(sweep_utils.cifar10(
              hyper, steps=int(10_000 * s), warmup=int(500 * s)))
          for s in [0.5, 1.0, 1.5, 2.0]
      ]),
      hyper.sweep('config.lr.base', [0.03, 0.01, 0.003, 0.001]),
      hyper.sweep('config.gp_layer.mean_field_factor',
                  [-1., 0.1, 0.2, 0.3, 0.5, 1., 2., 3., 5., 10., 20]),
  ])

  cifar100_sweep = hyper.product([
      hyper.chainit([
          hyper.product(sweep_utils.cifar100(
              hyper, steps=int(10_000 * s), warmup=int(500 * s)))
          for s in [0.5, 1.0, 1.5, 2.0]
      ]),
      hyper.sweep('config.lr.base', [0.03, 0.01, 0.003, 0.001]),
      hyper.sweep('config.gp_layer.mean_field_factor',
                  [-1., 0.1, 0.2, 0.3, 0.5, 1., 2., 3., 5., 10., 20]),
  ])

  imagenet_sweep = hyper.product([
      hyper.chainit([
          hyper.product(sweep_utils.imagenet(
              hyper, steps=int(20_000 * s), warmup=int(500 * s)))
          for s in [0.5, 1.0, 1.5, 2.0]
      ]),
      hyper.sweep('config.lr.base', [0.1, 0.06, 0.03, 0.01]),
      hyper.sweep('config.gp_layer.mean_field_factor',
                  [-1., 0.1, 0.2, 0.3, 0.5, 1., 2., 3., 5., 10., 20]),
  ])

  imagenet_1shot_sweep = hyper.product([
      hyper.chainit([
          hyper.product(sweep_utils.imagenet_fewshot(
              hyper, fewshot='1shot', steps=200, warmup=s,
              log_eval_steps=20)) for s in [1, 5, 10, 20, 30, 40, 50]
      ]),
      hyper.sweep('config.lr.base', [0.06, 0.05, 0.04, 0.03, 0.02, 0.01]),
      hyper.sweep('config.gp_layer.mean_field_factor',
                  [-1, 1e-4, 1e-3, 1e-2, 0.1, 0.5, 1, 2]),
  ])

  imagenet_5shot_sweep = hyper.product([
      hyper.chainit([
          hyper.product(sweep_utils.imagenet_fewshot(
              hyper, fewshot='5shot', steps=1000, warmup=s,
              log_eval_steps=100)) for s in [1, 5, 10, 20, 30, 40, 50]
      ]),
      hyper.sweep('config.lr.base', [0.06, 0.05, 0.04, 0.03, 0.02, 0.01]),
      hyper.sweep('config.gp_layer.mean_field_factor',
                  [-1, 1e-4, 1e-3, 1e-2, 0.1, 0.5, 1, 2]),
  ])

  imagenet_10shot_sweep = hyper.product([
      hyper.chainit([
          hyper.product(sweep_utils.imagenet_fewshot(
              hyper, fewshot='10shot', steps=2000, warmup=s,
              log_eval_steps=200)) for s in [1, 5, 10, 20, 30, 40, 50]
      ]),
      hyper.sweep('config.lr.base', [0.06, 0.05, 0.04, 0.03, 0.02, 0.01]),
      hyper.sweep('config.gp_layer.mean_field_factor',
                  [-1, 1e-4, 1e-3, 1e-2, 0.1, 0.5, 1, 2]),
  ])

  return hyper.product([
      hyper.chainit([
          cifar10_sweep,
          cifar100_sweep,
          imagenet_sweep,
          imagenet_1shot_sweep,
          imagenet_5shot_sweep,
          imagenet_10shot_sweep,
      ]),
      hyper.sweep('config.model_init', checkpoints),
  ])

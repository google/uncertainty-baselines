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
r"""Active learn a pretrained ViT-L/32 on CIFAR-10.

Based on: vit_l32_finetune.py and sweep_utils.py

"""
# pylint: enable=line-too-long

import ml_collections
import sweep_utils  # local file import from baselines.jft.experiments


def get_config():
  """Config for finetuning."""
  config = ml_collections.ConfigDict()

  config.seed = 0

  # Active learning section
  config.acquisition_method = ''  # set in sweep
  config.max_training_set_size = 200
  config.initial_training_set_size = 0
  config.acquisition_batch_size = 10
  config.early_stopping_patience = 64

  # Dataset section
  config.dataset = ''  # set in sweep
  config.test_split = ''  # set in sweep
  config.val_split = ''  # set in sweep
  config.train_split = ''  # set in sweep
  config.num_classes = None  # set in swee

  config.batch_size = 256  # half of config's 512 - due to memory issues
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

  # Model section
  config.model_init = ''  # set in sweep
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
  config.lr.base = 0.003  # set in sweep
  # Turned off for active learning:
  config.lr.warmup_steps = 0
  config.lr.decay_type = 'cosine'
  config.eval_on_cifar_10h = False
  # OOD evaluation. Not used but needed for sweep_utils.
  config.ood_datasets = []
  config.ood_num_classes = []
  config.ood_split = ''
  config.ood_methods = []
  config.pp_eval_ood = []
  config.eval_on_cifar_10h = False
  config.pp_eval_cifar_10h = ''
  config.eval_on_imagenet_real = False
  config.pp_eval_imagenet_real = ''

  return config


def get_sweep(hyper):
  """Sweeps over datasets."""
  # Adapted the sweep over checkpoints from vit_l32_finetune.py.
  checkpoints = ['/path/to/pretrained_model_ckpt.npz']
  use_jft = True  # whether to use JFT-300M or ImageNet-21K settings
  sweep_lr = True  # whether to sweep over learning rates
  acquisition_methods = ['uniform', 'entropy', 'margin', 'density']
  if use_jft:
    cifar10_sweep = sweep_utils.cifar10(hyper)
    cifar10_sweep.append(hyper.fixed('config.lr.base', 0.01, length=1))
    cifar10_sweep = hyper.product(cifar10_sweep)
  else:
    cifar10_sweep = sweep_utils.cifar10(hyper)
    cifar10_sweep.append(hyper.fixed('config.lr.base', 0.003, length=1))
    cifar10_sweep = hyper.product(cifar10_sweep)
  if sweep_lr:
    # Apply a learning rate sweep following Table 4 of Vision Transformer paper.
    checkpoints = [checkpoints[0]]
    cifar10_sweep = hyper.chainit([
        hyper.product(sweep_utils.cifar10(
            hyper, steps=int(10_000 * s), warmup=int(500 * s)))
        for s in [0.5, 1.0, 1.5, 2.0]
    ])
    cifar10_sweep = hyper.product([
        cifar10_sweep,
        hyper.sweep('config.lr.base', [0.03, 0.01, 0.003, 0.001])])

  return hyper.product([
      cifar10_sweep,
      hyper.sweep('config.model_init', checkpoints),
      hyper.sweep('config.acquisition_method', acquisition_methods),
  ])

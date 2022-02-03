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
r"""Finetune a pretrained ViT-L/32 on CIFAR-10/100 and ImageNet.

This config is used for models pretrained on either JFT-300M or ImageNet-21K.

"""
# pylint: enable=line-too-long

import ml_collections
import sweep_utils  # local file import from baselines.jft.experiments


def get_config():
  """Config for finetuning."""
  config = ml_collections.ConfigDict()

  config.model_init = ''  # set in sweep
  config.dataset = ''  # set in sweep
  config.test_split = ''  # set in sweep
  config.val_split = ''  # set in sweep
  config.test_split = ''  # set in sweep
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

  # Heteroscedastic
  config.model.use_het = True
  # TODO(trandustin): Enable finetuning jobs with multi-class HetSNGP layer.
  config.model.multiclass = True
  config.model.temperature = 0.2
  config.model.mc_samples = 1000
  config.model.num_factors = 0  # set in sweep
  config.model.param_efficient = True

  # BatchEnsemble
  config.model.use_be = True
  config.model.transformer.be_layers = (21, 23)
  config.model.transformer.ens_size = 3
  config.model.transformer.random_sign_init = 0.5

  # GP
  config.model.use_gp = False
  # Use momentum-based (i.e., non-exact) covariance update for pre-training.
  # This is because the exact covariance update can be unstable for pretraining,
  # since it involves inverting a precision matrix accumulated over 300M data.
  config.model.covmat_momentum = .999
  config.model.ridge_penalty = 1.
  # No need to use mean field adjustment for pretraining.
  config.model.mean_field_factor = -1.

  # Optimizer section
  config.optim_name = 'Momentum'
  config.optim = ml_collections.ConfigDict()
  config.grad_clip_norm = 1.0
  config.weight_decay = None
  config.loss = 'softmax_xent'

  config.lr = ml_collections.ConfigDict()
  config.lr.base = 0.001  # set in sweep
  config.lr.warmup_steps = 0  # set in sweep
  config.lr.decay_type = 'cosine'
  return config


def get_sweep(hyper):
  """Sweeps over datasets."""
  checkpoints = ['/path/to/pretrained_model_ckpt.npz']
  # Apply a learning rate sweep following Table 4 of Vision Transformer paper.
  cifar10_sweep = sweep_utils.cifar10(hyper, val_split='train[98%:]')
  cifar10_sweep.extend([
      hyper.sweep('config.model.num_factors', [3]),
      hyper.sweep('config.lr.base', [0.03, 0.01, 0.003, 0.001]),
  ])
  cifar10_sweep = hyper.product(cifar10_sweep)

  cifar100_sweep = sweep_utils.cifar100(hyper, val_split='train[98%:]')
  cifar100_sweep.extend([
      hyper.sweep('config.model.num_factors', [10]),
      hyper.sweep('config.lr.base', [0.03, 0.01, 0.003, 0.001]),
  ])
  cifar100_sweep = hyper.product(cifar100_sweep)

  imagenet_sweep = sweep_utils.imagenet(hyper, val_split='train[99%:]')
  imagenet_sweep.extend([
      hyper.sweep('config.model.num_factors', [15]),
      hyper.sweep('config.lr.base', [0.06, 0.03, 0.01, 0.003]),
  ])
  imagenet_sweep = hyper.product(imagenet_sweep)

  return hyper.product([
      hyper.chainit([
          cifar10_sweep,
          cifar100_sweep,
          imagenet_sweep,
      ]),
      hyper.sweep('config.model.temperature', [0.35, 1.0, 2.0]),
      hyper.sweep('config.model_init', checkpoints),
  ])

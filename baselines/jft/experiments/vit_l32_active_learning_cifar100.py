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
r"""Active learn a pretrained (BE) ViT-L/32 on CIFAR-100.

Based on: vit_l32_finetune.py and sweep_utils.py

"""

import ml_collections


def get_config():
  """Config for finetuning."""
  config = ml_collections.ConfigDict()

  config.model_init = '/cns/tp-d/home/trandustin/baselines-jft-0209_205214/1/checkpoint.npz'  # pass as parameter to script
  # pylint: enable=line-too-long
  config.seed = 0

  n_cls = 100
  size = 384

  # AL section:
  config.model_type = 'deterministic'  # 'batchensemble'
  config.acquisition_method = 'margin'
  config.max_training_set_size = 200
  config.initial_training_set_size = 0
  config.acquisition_batch_size = 10
  config.early_stopping_patience = 64

  # Dataset section:
  config.dataset = 'cifar100'
  config.val_split = 'train[98%:]'
  config.train_split = 'train[:98%]'
  config.test_split = 'test'
  config.num_classes = n_cls

  config.batch_size = 256  # half of config's 512 - due to memory issues
  config.total_steps = 1024

  pp_common = '|value_range(-1, 1)'
  pp_common += f'|onehot({n_cls}, key="label", key_result="labels")'
  pp_common += '|keep(["image", "labels", "id"])'
  config.pp_train = f'decode|inception_crop({size})|flip_lr' + pp_common
  config.pp_eval = f'decode|resize({size})' + pp_common

  config.shuffle_buffer_size = 50_000  # Per host, so small-ish is ok.

  config.log_training_steps = 100
  config.log_eval_steps = 1000
  config.checkpoint_steps = 5000
  config.checkpoint_timeout = 1

  config.prefetch_to_device = 2
  config.trial = 0

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
  # BatchEnsemble parameters.
  config.model.transformer.be_layers = (21, 22, 23)
  config.model.transformer.ens_size = 3
  config.model.transformer.random_sign_init = -0.5
  config.fast_weight_lr_multiplier = 1.0
  # This is "no head" fine-tuning, which we use by default.
  config.model.representation_size = None

  # Optimizer section
  config.optim_name = 'Momentum'
  config.optim = ml_collections.ConfigDict()
  config.grad_clip_norm = 1.0
  config.weight_decay = None
  config.loss = 'softmax_xent'

  config.lr = ml_collections.ConfigDict()
  config.lr.base = 0.003
  # Turned off for active learning:
  # config.lr.warmup_steps = 0
  # config.lr.decay_type = 'cosine'
  return config


def get_sweep(hyper):
  """Sweeps over datasets."""
  # Apply a learning rate sweep following Table 4 of Vision Transformer paper.
  return hyper.product(
      [hyper.sweep('config.lr.base', [0.03])])

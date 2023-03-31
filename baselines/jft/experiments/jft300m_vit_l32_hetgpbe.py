# coding=utf-8
# Copyright 2023 The Uncertainty Baselines Authors.
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
r"""Heteroscedastic GP BatchEnsemble L/32.

"""
# pylint: enable=line-too-long

import ml_collections
from experiments import common_fewshot  # local file import from baselines.jft


def get_config():
  """Config."""
  config = ml_collections.ConfigDict()

  config.dataset = 'jft/entity:1.0.0'
  config.val_split = 'test[:49511]'  # aka tiny_test/test[:5%] in task_adapt
  config.train_split = 'train'  # task_adapt used train+validation so +64167
  config.num_classes = 18291
  config.init_head_bias = -10.0    # ~= ln(1/18k) ~= ln(1/num_classes)

  config.trial = 0
  config.batch_size = 4096
  config.num_epochs = 7

  pp_common = '|value_range(-1, 1)'
  pp_common += f'|onehot({config.num_classes})'
  # To use ancestor 'smearing', use this line instead:
  # pp_common += f'|onehot({config.num_classes}, key='labels_extended', key_result='labels')  # pylint: disable=line-too-long
  pp_common += '|keep(["image", "labels"])'
  config.pp_train = 'decode_jpeg_and_inception_crop(224)|flip_lr' + pp_common
  config.pp_eval = 'decode|resize_small(256)|central_crop(224)' + pp_common
  config.shuffle_buffer_size = 250_000  # Per host, so small-ish is ok.

  config.log_training_steps = 5000
  config.log_eval_steps = 10000
  config.checkpoint_steps = 15000
  config.checkpoint_timeout = 10

  # Model section
  config.model = ml_collections.ConfigDict()
  config.model.patches = ml_collections.ConfigDict()
  config.model.patches.size = [32, 32]
  config.model.hidden_size = 1024
  config.model.transformer = ml_collections.ConfigDict()
  config.model.transformer.attention_dropout_rate = 0.
  config.model.transformer.dropout_rate = 0.
  config.model.transformer.mlp_dim = 4096
  config.model.transformer.num_heads = 16
  config.model.transformer.num_layers = 24
  config.model.classifier = 'token'  # Or 'gap'
  config.model.representation_size = 1024

  # Heteroscedastic
  config.model.multiclass = False
  config.model.temperature = 0.2
  config.model.mc_samples = 100
  config.model.num_factors = 50
  config.model.param_efficient = True

  # BatchEnsemble
  config.model.transformer.be_layers = (21, 22, 23)
  config.model.transformer.ens_size = 3
  config.model.transformer.random_sign_init = -0.5

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
  config.optim_name = 'Adam'
  config.optim = ml_collections.ConfigDict()
  config.optim.weight_decay = 0.1
  config.optim.beta1 = 0.9
  config.optim.beta2 = 0.999
  config.weight_decay = None  # No explicit weight decay
  config.grad_clip_norm = 1.0  # setting from mark's rec
  config.lr = ml_collections.ConfigDict()
  config.lr.base = 6e-4  # LR has to be lower for larger models!
  config.lr.warmup_steps = 10_000
  config.lr.decay_type = 'linear'
  config.lr.linear_end = 1e-5

  # Few-shot eval section
  config.fewshot = common_fewshot.get_fewshot()
  config.fewshot.representation_layer = 'pre_ens_logits'
  config.fewshot.log_steps = 100_000
  return config


def get_sweep(hyper):
  return hyper.product([
      hyper.chainit([
          hyper.product([
              hyper.sweep('config.model.use_gp', [False]),
              hyper.sweep('config.model.transformer.random_sign_init',
                          [0.5, -0.5]),
              hyper.sweep('config.model.mc_samples', [100, 500]),
              hyper.sweep('config.grad_clip_norm', [1.0, 2.5]),
              hyper.sweep('config.model.temperature', [0.2, 0.35, 1.0]),
          ]),
          # hyper.product([
          #     hyper.sweep('config.model.use_gp', [True]),
          #     hyper.sweep('config.lr.base', [3e-4, 4e-4, 6e-4]),
          #     hyper.sweep('config.model.transformer.random_sign_init',
          #                 [0.5, -0.5]),
          #     hyper.sweep('config.model.temperature', [0.2, 0.35, 1.0]),
          # ]),
      ])
  ])

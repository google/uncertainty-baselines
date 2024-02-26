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
r"""ViT + BatchEnsemble.

"""
# pylint: enable=line-too-long

import ml_collections
from experiments import common_fewshot  # local file import from baselines.jft


def get_config():
  """Config."""
  config = ml_collections.ConfigDict()

  config.seed = 0

  # JFT parameters.
  config.dataset = 'jft/entity:1.0.0'
  config.val_split = 'test[:49511]'  # aka tiny_test/test[:5%] in task_adapt
  config.train_split = 'train'  # task_adapt used train+validation so +64167
  config.num_classes = 18291
  config.init_head_bias = -10.0    # ~= ln(1/18k) ~= ln(1/num_classes)

  pp_common = '|value_range(-1, 1)'
  pp_common += f'|onehot({config.num_classes})'
  # To use ancestor 'smearing', use this line instead:
  # pp_common += f'|onehot({config.num_classes}, key='labels_extended', key_result='labels')  # pylint: disable=line-too-long
  pp_common += '|keep(["image", "labels"])'
  config.pp_train = 'decode_jpeg_and_inception_crop(224)|flip_lr' + pp_common
  config.pp_eval = 'decode|resize_small(256)|central_crop(224)' + pp_common
  config.shuffle_buffer_size = 250_000  # Per host, so small-ish is ok.

  # Model parameters.
  config.model = ml_collections.ConfigDict()
  config.model.patches = ml_collections.ConfigDict()
  config.model.patches.size = [32, 32]
  config.model.hidden_size = 1024
  config.model.representation_size = 1024
  config.model.classifier = 'token'
  config.model.transformer = ml_collections.ConfigDict()
  config.model.transformer.num_layers = 24
  config.model.transformer.dropout_rate = 0.0
  config.model.transformer.mlp_dim = 4096
  config.model.transformer.num_heads = 16
  config.model.transformer.attention_dropout_rate = 0.0

  # BatchEnsemble parameters.
  config.model.transformer.be_layers = (22, 23)
  config.model.transformer.ens_size = 3
  config.model.transformer.random_sign_init = -0.5
  # TODO(trandustin): Remove `ensemble_attention` hparam once we no longer
  # need checkpoints that only apply BE on the FF block.
  config.model.transformer.ensemble_attention = True
  config.fast_weight_lr_multiplier = 1.0

  # Optimizer parameters.
  config.optim_name = 'Adam'
  config.optim = ml_collections.ConfigDict()
  config.optim.beta1 = 0.9
  config.optim.beta2 = 0.999
  # TODO(trandustin): Note BE uses config.weight_decay instead of
  # config.optim.weight_decay as in typical ViT. xid/34376832 shows
  # config.weight_decay is better for BE. Still need to sweep over LR though.
  config.weight_decay = 0.1

  config.lr = ml_collections.ConfigDict()
  config.lr.base = 4e-4  # LR likely has to be lower for larger models!
  config.lr.warmup_steps = 10_000
  config.lr.decay_type = 'linear'
  config.lr.linear_end = 1e-5
  config.disable_preemption_reproducibility = True

  config.batch_size = 4096         # Global batch size.
  config.num_epochs = 7

  config.log_training_steps = 50
  config.log_eval_steps = 1000

  config.checkpoint_steps = 5000
  config.checkpoint_timeout = 10

  config.prefetch_to_device = 2
  config.trial = 0

  # Few-shot eval section
  config.fewshot = common_fewshot.get_fewshot()
  config.fewshot.log_steps = 25_000
  return config


def get_sweep(hyper):
  # Use this as a sensible sweep over hyperparameters.
  return hyper.product([
      hyper.chainit([
          hyper.product([
              hyper.sweep('config.model.transformer.ens_size', [2]),
              hyper.sweep('config.model.transformer.be_layers',
                          [(20, 21, 22, 23), (21, 22, 23)]),
          ]),
          hyper.product([
              hyper.sweep('config.model.transformer.ens_size', [3]),
              hyper.sweep('config.model.transformer.be_layers',
                          [(21, 22, 23), (22, 23)]),
          ]),
      ]),
      hyper.sweep('config.lr.base', [4e-4, 6e-4]),
      hyper.sweep('config.model.transformer.random_sign_init',
                  [-0.5, 0.5]),
  ])

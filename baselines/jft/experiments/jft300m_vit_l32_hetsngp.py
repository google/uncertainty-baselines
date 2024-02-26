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
r"""Heteroscedastic ViT-L/32.

"""
# pylint: enable=line-too-long

import ml_collections
# TODO(dusenberrymw): Open-source remaining imports.
from experiments import common_fewshot  # local file import from baselines.jft


def get_config():
  """Config for training a patch-transformer on JFT."""
  config = ml_collections.ConfigDict()

  config.seed = 0

  # Directory for the version de-dup'd from BiT downstream test-sets.
  config.dataset = 'jft/entity:1.0.0'
  config.val_split = 'test[:49511]'  # aka tiny_test/test[:5%] in task_adapt
  config.train_split = 'train'  # task_adapt used train+validation so +64167
  config.num_classes = 18291
  config.init_head_bias = -10.0

  config.trial = 0
  config.batch_size = 4096
  config.num_epochs = 7

  pp_common = '|value_range(-1, 1)'
  pp_common += f'|onehot({config.num_classes})'
  # To use ancestor "smearing", use this line instead:
  # pp_common += f'|onehot({config.num_classes}, key="labels_extended", key_result="labels")  # pylint: disable=line-too-long
  pp_common += '|keep(["image", "labels"])'
  config.pp_train = 'decode_jpeg_and_inception_crop(224)|flip_lr' + pp_common
  config.pp_eval = 'decode|resize_small(256)|central_crop(224)' + pp_common
  config.shuffle_buffer_size = 250_000  # Per host, so small-ish is ok.

  config.log_training_steps = 10000
  config.log_eval_steps = 73230  # ~= steps_per_epoch
  # NOTE: Save infrequently to prevent crowding the disk space.
  config.checkpoint_steps = 17250
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
  config.het = ml_collections.ConfigDict()
  config.het.multiclass = False
  config.het.temperature = 1.5
  config.het.mc_samples = 1000
  config.het.num_factors = 50
  config.het.param_efficient = True

  # Gaussian process layer section
  config.gp_layer = ml_collections.ConfigDict()
  # Use momentum-based (i.e., non-exact) covariance update for pre-training.
  # This is because the exact covariance update can be unstable for pretraining,
  # since it involves inverting a precision matrix accumulated over 300M data.
  config.gp_layer.covmat_momentum = .999
  config.gp_layer.ridge_penalty = 1.
  # No need to use mean field adjustment for pretraining.
  config.gp_layer.mean_field_factor = -1.

  # Optimizer section
  config.optim_name = 'Adam'
  config.optim = ml_collections.ConfigDict()
  config.optim.weight_decay = 0.1
  config.optim.beta1 = 0.9
  config.optim.beta2 = 0.999
  config.weight_decay = None  # No explicit weight decay
  config.grad_clip_norm = 1.0

  # TODO(lbeyer): make a mini-language like preprocessings.
  config.lr = ml_collections.ConfigDict()
  config.lr.base = 6e-4  # LR has to be lower for larger models!
  config.lr.warmup_steps = 10_000
  config.lr.decay_type = 'linear'
  config.lr.linear_end = 1e-5

  # Few-shot eval section
  config.fewshot = common_fewshot.get_fewshot()
  config.fewshot.log_steps = 50_000
  return config


def get_sweep(hyper):
  return hyper.product([
      hyper.sweep('config.seed', [0]),
      hyper.sweep('config.het.temperature',
                  [0.25, 0.4, 0.55, 0.7, 0.85, 1.0])
  ])

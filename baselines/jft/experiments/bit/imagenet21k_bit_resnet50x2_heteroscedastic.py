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
r"""BiT ResNet-50x2.

"""
# pylint: enable=line-too-long

import ml_collections
from experiments.common_fewshot import get_fewshot  # local file import from baselines.jft


def get_config():
  """Config for training a BiT ResNet-50x2 on ImageNet21k."""
  config = ml_collections.ConfigDict()

  # Directory for the version de-dup'd from BiT downstream test-sets.
  config.dataset = 'imagenet21k'
  config.train_split = 'full[204800:]'
  config.val_split = 'full[102400:204800]'
  config.test_split = 'full[:102400]'  # former val when using 2-fold split
  config.num_classes = 21843
  config.init_head_bias = -10.0

  config.trial = 0
  config.seed = 0
  config.batch_size = 256
  config.num_epochs = 90

  pp_common = '|value_range(-1, 1)'
  config.pp_train = 'decode_jpeg_and_inception_crop(224)|flip_lr' + pp_common
  config.pp_train += f'|onehot({config.num_classes}, on=0.9999, off=0.0001)'
  config.pp_eval = 'decode|resize_small(256)|central_crop(224)' + pp_common
  config.pp_eval += f'|onehot({config.num_classes})'
  config.shuffle_buffer_size = 250_000  # Per host, so small-ish is ok.

  config.log_training_steps = 10000
  config.log_eval_steps = 50000
  # NOTE: Save infrequently to prevent crowding the disk space.
  config.checkpoint_steps = 17250
  config.checkpoint_timeout = 10

  # Model section
  config.model = ml_collections.ConfigDict()
  config.model.num_layers = 50
  config.model.width_factor = 2

  # Heteroscedastic
  config.model.multiclass = False
  config.model.temperature = 0.4
  config.model.temperature_lower_bound = 0.05
  config.model.temperature_upper_bound = 5.0
  config.model.mc_samples = 1000
  config.model.num_factors = 50
  config.model.param_efficient = True
  config.model.latent_het = False

  # Using the same hyperparameters as the ResNet50x2 baseline in the ViT paper
  # (https://arxiv.org/abs/2010.11929), see Table 3.
  # Optimizer section
  config.optim_name = 'Adam'
  config.optim = ml_collections.ConfigDict()
  config.optim.weight_decay = 0.03

  # TODO(lbeyer): make a mini-language like preprocessings.
  config.lr = ml_collections.ConfigDict()
  config.lr.base = 0.001
  config.lr.warmup_steps = 10_000
  config.lr.decay_type = 'linear'
  config.lr.linear_end = 1e-5

  # Few-shot eval section
  config.fewshot = get_fewshot()
  config.fewshot.log_steps = 50_000

  # Disable unnecessary CNS TTLs.
  config.ttl = 0
  return config


def get_sweep(hyper):
  return hyper.product([
      hyper.sweep('config.seed', [0]),
      hyper.sweep('config.num_epochs', [7]),
      hyper.sweep('config.model.temperature',
                  [-1, 0.05, 0.1, 0.2, 0.4, 0.8, 1.5, 3.0, 5.0]),
      hyper.sweep('config.model.latent_het', [True])
  ])

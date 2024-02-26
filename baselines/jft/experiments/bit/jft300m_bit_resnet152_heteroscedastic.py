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
r"""BiT ResNet152x{1,2} Heteroscedastic.

"""
# pylint: enable=line-too-long

import ml_collections


def get_config():
  """Config for training a BiT ResNet-152x{1,2} on JFT."""
  config = ml_collections.ConfigDict()

  # Directory for the version de-dup'd from BiT downstream test-sets.
  config.dataset = 'jft/entity:1.0.0'
  config.val_split = 'test[:49511]'  # aka tiny_test/test[:5%] in task_adapt
  config.train_split = 'train'  # task_adapt used train+validation so +64167
  config.num_classes = 18291
  config.init_head_bias = -10.0

  config.trial = 0
  config.seed = 0
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

  config.log_training_steps = 10000
  config.log_eval_steps = 50000
  # NOTE: Save infrequently to prevent crowding the disk space.
  config.checkpoint_steps = 17250
  config.checkpoint_timeout = 10

  # Model section
  config.model = ml_collections.ConfigDict()
  config.model.num_layers = 152
  config.model.width_factor = 1

  # Heteroscedastic
  config.model.multiclass = False
  config.model.temperature = 0.4
  config.model.temperature_lower_bound = 0.05
  config.model.temperature_upper_bound = 5.0
  config.model.mc_samples = 1000
  config.model.num_factors = 50
  config.model.param_efficient = True
  config.model.latent_het = False

  # Using the same hyperparameters as the ResNet152x2 baseline in the ViT paper
  # (https://arxiv.org/abs/2010.11929), see Table 3.

  # Optimizer section
  config.optim_name = 'Adam'
  config.optim = ml_collections.ConfigDict()
  config.optim.weight_decay = 0.1
  config.optim.beta1 = 0.9
  config.optim.beta2 = 0.999
  config.weight_decay = None  # No explicit weight decay.

  config.lr = ml_collections.ConfigDict()
  config.lr.base = 6e-4
  config.lr.warmup_steps = 10_000
  config.lr.decay_type = 'linear'
  config.lr.linear_end = 1e-5

  # Disable unnecessary CNS TTLs.
  config.ttl = 0
  return config


def get_sweep(hyper):
  return hyper.product([
      hyper.sweep('config.seed', [0]),
      hyper.sweep('config.num_epochs', [7]),
      hyper.sweep('config.model.mc_samples', [1000]),
      hyper.sweep('config.model.width_factor', [1]),
      hyper.sweep('config.model.temperature', [-1.0]),
      hyper.sweep('config.model.latent_het', [True])
  ])

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
r"""ViT-B/16.

"""
# pylint: enable=line-too-long

import ml_collections
from experiments import common_fewshot  # local file import from baselines.jft


def get_config():
  """Config for training a patch-transformer on JFT."""
  config = ml_collections.ConfigDict()

  # Directory for the version de-dup'd from BiT downstream test-sets.
  config.dataset = 'imagenet21k'
  config.val_split = 'full[:102400]'
  config.train_split = 'full[102400:]'
  config.num_classes = 21843
  config.init_head_bias = -10.0

  config.trial = 0
  config.batch_size = 1024
  config.num_epochs = 90

  pp_common = '|value_range(-1, 1)'
  config.pp_train = 'decode_jpeg_and_inception_crop(224)|flip_lr' + pp_common
  config.pp_train += f'|onehot({config.num_classes}, on=0.9999, off=0.0001)'
  config.pp_eval = 'decode|resize_small(256)|central_crop(224)' + pp_common
  config.pp_eval += f'|onehot({config.num_classes})'
  config.shuffle_buffer_size = 250_000  # Per host, so small-ish is ok.

  config.log_training_steps = 1000
  config.log_eval_steps = 10000
  # NOTE: eval is very fast O(seconds) so it's fine to run it often.
  config.checkpoint_steps = 17250
  config.checkpoint_timeout = 10

  # Model section
  config.model = ml_collections.ConfigDict()
  config.model.patches = ml_collections.ConfigDict()
  config.model.patches.size = [16, 16]
  config.model.hidden_size = 768
  config.model.transformer = ml_collections.ConfigDict()
  config.model.transformer.attention_dropout_rate = 0.
  config.model.transformer.dropout_rate = 0.1
  config.model.transformer.mlp_dim = 3072
  config.model.transformer.num_heads = 12
  config.model.transformer.num_layers = 12
  config.model.classifier = 'token'  # Or 'gap'
  config.model.representation_size = 768
  # Optimizer section
  config.optim_name = 'Adam'
  config.optim = ml_collections.ConfigDict()
  config.optim.weight_decay = 0.03

  # TODO(lbeyer): make a mini-language like preprocessings.
  config.lr = ml_collections.ConfigDict()
  config.lr.base = 0.001  # LR has to be lower for larger models!
  config.lr.warmup_steps = 10_000
  config.lr.decay_type = 'linear'
  config.lr.linear_end = 1e-5

  # Few-shot eval section
  config.fewshot = common_fewshot.get_fewshot()
  config.fewshot.log_steps = 10_000
  return config


def get_sweep(hyper):
  return hyper.product([])

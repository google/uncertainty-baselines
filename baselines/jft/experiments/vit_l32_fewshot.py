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
r"""Fewshot experiments for a pretrained ViT single models (Det, GP, Het).

This config is used for models pretrained on either JFT-300M or ImageNet-21K.

"""

import ml_collections
from experiments import common_fewshot  # local file import from baselines.jft


def get_config():
  """Config for finetuning."""
  config = ml_collections.ConfigDict()
  # Dataset.
  config.dataset = 'imagenet2012'
  config.train_split = 'train'
  config.val_split = 'validation'
  config.num_classes = 1000

  # ViT-L32 i21k: Det.
  config.model_init = ''

  # Model section
  config.model_family = 'single'
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
  config.model.representation_size = 1024

  # Few-shot eval section
  config.fewshot = common_fewshot.get_fewshot()
  config.fewshot.log_steps = 50_000
  return config


def get_sweep(hyper):
  """Sweeps over datasets."""
  checkpoints = ['/path/to/pretrained_model_ckpt.npz']

  return hyper.product([
      hyper.sweep('config.model_init', checkpoints),
  ])

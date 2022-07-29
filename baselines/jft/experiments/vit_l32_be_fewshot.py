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
r"""Fewshot experiments for a pretrained ViT (S32, B32, L32) BE.

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
  config.model_family = 'batchensemble'
  config.model = ml_collections.ConfigDict()
  config.model.patches = ml_collections.ConfigDict()
  config.model.patches.size = [32, 32]
  config.model.hidden_size = 512
  config.model.transformer = ml_collections.ConfigDict()
  config.model.transformer.mlp_dim = 2048
  config.model.transformer.num_heads = 8
  config.model.transformer.num_layers = 8
  config.model.transformer.attention_dropout_rate = 0.
  config.model.transformer.dropout_rate = 0.
  config.model.classifier = 'token'
  config.model.representation_size = 512

  # BatchEnsemble parameters.
  config.model.transformer.be_layers = (5, 6, 7)
  config.model.transformer.ens_size = 2
  config.model.transformer.random_sign_init = -0.5
  # TODO(trandustin): Remove `ensemble_attention` hparam once we no longer
  # need checkpoints that only apply BE on the FF block.
  config.model.transformer.ensemble_attention = False

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

# coding=utf-8
# Copyright 2021 The Uncertainty Baselines Authors.
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
r"""ViT-B/16 finetuning on CIFAR.

"""
# pylint: enable=line-too-long

import ml_collections
# TODO(dusenberrymw): Open-source remaining imports.


def get_sweep(hyper):
  return hyper.product([])


def get_config():
  """Config for training a patch-transformer on JFT."""
  config = ml_collections.ConfigDict()

  # Fine-tuning dataset
  config.dataset = 'cifar10'
  config.val_split = 'test[:50%]'
  config.train_split = 'train'
  config.num_classes = 10

  config.trial = 0
  BATCH_SIZE = 512  # pylint: disable=invalid-name
  config.batch_size = BATCH_SIZE
  config.total_steps = 10_000

  pp_common = '|value_range(-1, 1)'
  # pp_common += f'|onehot({config.num_classes})'
  # To use ancestor 'smearing', use this line instead:
  pp_common += f'|onehot({config.num_classes}, key="label", key_result="labels")'  # pylint: disable=line-too-long
  pp_common += '|keep("image", "labels")'
  config.pp_train = 'decode|inception_crop(224)|flip_lr' + pp_common
  config.pp_eval = 'decode|resize(224)' + pp_common
  config.shuffle_buffer_size = 50_000  # Per host, so small-ish is ok.

  config.log_training_steps = 10
  config.log_eval_steps = 100
  # NOTE: eval is very fast O(seconds) so it's fine to run it often.
  config.checkpoint_steps = 1000

  # Model section
  # pre-trained model ckpt file
  # !!!  The below section should be modified per experiment
  config.model_init = '/path/to/pretrained_model_ckpt.npz'
  # Model definition to be copied from the pre-training config
  config.model = ml_collections.ConfigDict()
  config.model.patches = ml_collections.ConfigDict()
  config.model.patches.size = [16, 16]
  config.model.hidden_size = 768
  config.model.transformer = ml_collections.ConfigDict()
  config.model.transformer.attention_dropout_rate = 0.
  config.model.transformer.dropout_rate = 0.
  config.model.transformer.mlp_dim = 3072
  config.model.transformer.num_heads = 12
  config.model.transformer.num_layers = 12
  config.model.classifier = 'token'  # Or 'gap'

  # This is "no head" fine-tuning, which we use by default
  config.model.representation_size = None

  # Optimizer section
  config.optim_name = 'Adam'  # TODO(jjren)  finetune uses 'momentum_hp'
  config.optim = ml_collections.ConfigDict()
  config.optim.weight_decay = 0.1
  config.optim.beta1 = 0.9
  config.optim.beta2 = 0.999
  config.weight_decay = None  # No explicit weight decay

  # TODO(lbeyer): make a mini-language like preprocessings.
  config.lr = ml_collections.ConfigDict()
  config.lr.base = 0.001
  config.lr.warmup_steps = 500
  config.lr.decay_type = 'cosine'
  config.lr.linear_end = 1e-5

  config.args = {}
  return config

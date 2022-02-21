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
r"""ViT + BatchEnsemble.

"""
# pylint: enable=line-too-long

import ml_collections
import common_fewshot  # local file import from baselines.jft.experiments


def get_config():
  """Config."""
  config = ml_collections.ConfigDict()

  config.seed = 0

  config.dataset = 'jft/entity:1.0.0'
  config.val_split = 'test[:49511]'  # aka tiny_test/test[:5%] in task_adapt
  config.train_split = 'train'  # task_adapt used train+validation so +64167
  config.num_classes = 18291
  config.init_head_bias = -10.0

  config.resume = '/cns/tp-d/home/trandustin/baselines-jft-0211_032549/1/checkpoint.npz'
  config.trial = 0
  config.batch_size = 4096
  config.num_epochs = 14
  config.prefetch_to_device = 2
  # TODO(trandustin): To resume properly, I removed this setting. Not sure what
  # it's doing.
  # config.disable_preemption_reproducibility = True

  pp_common = '|value_range(-1, 1)'
  pp_common += f'|onehot({config.num_classes})'
  pp_common += '|keep(["image", "labels"])'
  config.pp_train = 'decode_jpeg_and_inception_crop(224)|flip_lr' + pp_common
  config.pp_eval = 'decode|resize_small(256)|central_crop(224)' + pp_common
  config.shuffle_buffer_size = 250_000  # Per host, so small-ish is ok.

  config.log_training_steps = 50
  config.log_eval_steps = 1000
  config.checkpoint_steps = 5000
  config.checkpoint_timeout = 10

  # Model section
  config.model = ml_collections.ConfigDict()
  config.model.patches = ml_collections.ConfigDict()
  config.model.patches.size = [14, 14]
  config.model.hidden_size = 1280
  config.model.transformer = ml_collections.ConfigDict()
  config.model.transformer.attention_dropout_rate = 0.
  config.model.transformer.dropout_rate = 0.
  config.model.transformer.mlp_dim = 5120
  config.model.transformer.num_heads = 16
  config.model.transformer.num_layers = 32
  config.model.classifier = 'token'  # Or 'gap'
  config.model.representation_size = 1280

  # BatchEnsemble section
  # Using last n=5 layers was chosen somewhat arbitrarily, >3 from L/32.
  config.model.transformer.be_layers = (27, 28, 29, 30, 31)
  config.model.transformer.ens_size = 3
  config.model.transformer.random_sign_init = 0.5
  config.fast_weight_lr_multiplier = 1.0

  # Optimizer section
  # We use Adam HP to lower memory.
  config.optim_name = 'adam_hp'
  config.optim = ml_collections.ConfigDict()
  config.optim.beta1 = 0.9
  config.optim.beta2 = 0.999
  config.weight_decay = 0.1
  config.grad_clip_norm = 10.0

  config.lr = ml_collections.ConfigDict()
  # Note original ViT-H/14 uses 4e-4 and no grad clip norm until 130K steps,
  # then 3e-4 and grad_clip_norm=10.0 after.
  config.lr.base = 3e-4  # LR has to be lower for larger models!
  config.lr.warmup_steps = 10_000
  config.lr.decay_type = 'linear'
  config.lr.linear_end = 1e-5

  # Few-shot eval section
  config.fewshot = common_fewshot.get_fewshot()
  config.fewshot.log_steps = 25_000
  return config


def get_sweep(hyper):
  return hyper.product([])

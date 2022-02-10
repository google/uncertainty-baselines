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
r"""MIMO ViT-L/32.

"""
# pylint: enable=line-too-long

import ml_collections


def get_config():
  """Config for training on JFT300M. Batch size 4096 fits on DF4x4."""
  config = ml_collections.ConfigDict()

  config.seed = 0

  # Directory for the version de-dup'd from BiT downstream test-sets.
  config.dataset = 'jft/entity:1.0.0'
  config.val_split = 'test[:49511]'  # aka tiny_test/test[:5%] in task_adapt
  config.train_split = 'train'  # task_adapt used train+validation so +64167
  config.num_classes = 18291
  config.init_head_bias = -10.0    # ~= ln(1/18k) ~= ln(1/num_classes)

  config.trial = 0
  config.batch_size = 4096         # Global batch size.
  config.batch_size_eval = 4096    # Global batch size.
  config.num_epochs = 7

  pp_common = '|value_range(-1, 1)'
  pp_common += f'|onehot({config.num_classes})'
  # To use ancestor 'smearing', use this line instead:
  # pp_common += f'|onehot({config.num_classes}, key='labels_extended', key_result='labels')  # pylint: disable=line-too-long
  pp_common += '|keep(["image", "labels"])'
  config.pp_train = 'decode_jpeg_and_inception_crop(224)|flip_lr' + pp_common
  config.pp_eval = 'decode|resize_small(256)|central_crop(224)' + pp_common
  config.shuffle_buffer_size = 250_000  # Per host, so small-ish is ok.

  config.log_training_steps = 50
  config.log_eval_steps = 1000
  config.checkpoint_steps = 1000

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

  # MIMO section
  config.model.ensemble_size = 2
  config.mimo = ml_collections.ConfigDict()
  config.mimo.batch_repetitions = 2
  config.mimo.input_repetition_prob = 0.9

  # Optimizer section
  config.optim_name = 'Adam'
  config.optim = ml_collections.ConfigDict()
  config.optim.weight_decay = 0.1
  config.optim.beta1 = 0.9
  config.optim.beta2 = 0.999

  config.lr = ml_collections.ConfigDict()
  config.lr.base = 6e-4  # LR has to be lower for larger models!
  config.lr.warmup_steps = 10_000
  config.lr.decay_type = 'linear'
  config.lr.linear_end = 1e-5

  # Few-shot eval section
  # TODO(trandustin): Reenable after implementing.
  # config.fewshot = common_fewshot.get_fewshot()
  # config.fewshot.log_steps = 25_000

  return config


def get_sweep(hyper):
  return hyper.product([])
  # Use these sweeps for sensible hparam values.
  # return hyper.chainit([
  #     hyper.product([
  #         hyper.sweep('config.model.ensemble_size', [1]),
  #         hyper.sweep('config.mimo.batch_repetitions', [1]),
  #         hyper.sweep('config.mimo.input_repetition_prob', [0.]),
  #     ]),
  #     hyper.product([
  #         hyper.sweep('config.model.ensemble_size', [2]),
  #         hyper.sweep('config.mimo.batch_repetitions', [2]),
  #         hyper.sweep('config.mimo.input_repetition_prob',
  #                     [0.6, 0.8, 0.9]),
  #         hyper.sweep('config.model.transformer.mlp_dim', [4096]),
  #     ]),
  #     hyper.product([
  #         hyper.sweep('config.model.ensemble_size', [4]),
  #         hyper.sweep('config.mimo.batch_repetitions', [1, 2, 4]),
  #         hyper.sweep('config.mimo.input_repetition_prob', [0.]),
  #     ]),
  # ])

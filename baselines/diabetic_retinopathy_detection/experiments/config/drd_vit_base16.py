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

r"""Training ViT-B/16 from scratch on a Diabetic Retinopathy Detection dataset.

"""
# pylint: enable=line-too-long

import ml_collections
# TODO(dusenberrymw): Open-source remaining imports.


def get_sweep(hyper):
  return hyper.product([])


def get_config():
  """Config for training a patch-transformer on JFT."""
  config = ml_collections.ConfigDict()

  # Training dataset (no pretraining in this configuration,
  # we train from scratch on a Diabetic Retinopathy Detection dataset)
  config.data_dir = 'gs://ub-data/retinopathy'

  # TODO(nband): fix issue with sigmoid loss.
  config.num_classes = 2

  # Input resolution of each retina image. (Default: 512)
  config.pp_input_res = 512  # pylint: disable=invalid-name
  pp_common = f'|onehot({config.num_classes})'
  config.pp_train = (
      f'diabetic_retinopathy_preprocess({config.pp_input_res})' + pp_common)
  config.pp_eval = (
      f'diabetic_retinopathy_preprocess({config.pp_input_res})' + pp_common)
  config.shuffle_buffer_size = 15_000  # Per host, so small-ish is ok.

  config.log_training_steps = 100
  config.log_eval_steps = 1000
  # NOTE: eval is very fast O(seconds) so it's fine to run it often.
  config.checkpoint_steps = 1000
  config.checkpoint_timeout = 1

  config.prefetch_to_device = 2
  config.trial = 0

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
  config.optim_name = 'Momentum'
  config.optim = ml_collections.ConfigDict()
  config.loss = 'softmax_xent'  # or 'sigmoid_xent'
  # config.loss = 'sigmoid_xent'

  config.args = {}
  return config

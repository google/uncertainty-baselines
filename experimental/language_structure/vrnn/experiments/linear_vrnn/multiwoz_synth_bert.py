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

r"""Vizier for linear VRNN for MultiWoZSynthDataset.

"""

import multiwoz_synth_tmpl  # local file import from experimental.language_structure.vrnn.experiments.linear_vrnn as tmpl


def get_config():
  """Returns the configuration for this experiment."""
  config = tmpl.get_config(
      shared_bert_embedding=True, bert_embedding_type='base')

  config.platform = 'df'
  config.tpu_topology = '4x2'

  config.max_task_failures = -1
  config.max_per_task_failures = 10


  return config


def get_sweep(hyper):
  """Returns hyperparameter sweep."""
  domain = [
      hyper.sweep('config.word_weights_file_weight',
                  hyper.discrete([0.25 * i for i in range(5)])),
      hyper.sweep('config.psl_constraint_learning_weight',
                  hyper.discrete([0., 0.001, 0.005, 0.01, 0.05, 0.1])),
      hyper.sweep('config.model.vae_cell.encoder_hidden_size',
                  hyper.discrete([200, 300, 400])),
      hyper.sweep('config.base_learning_rate',
                  hyper.discrete([3e-5, 5e-5, 1e-4, 3e-4]))
  ]
  sweep = hyper.product(domain)
  return sweep

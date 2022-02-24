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

"""VRNN model config."""

from ml_collections import config_dict

VanillaLinearVAECellConfig = config_dict.ConfigDict
VanillaLinearVRNNConfig = config_dict.ConfigDict


def vanilla_linear_vae_cell_config(**kwargs) -> VanillaLinearVAECellConfig:
  """Creates model config for VanillaLinearVAECell."""
  config = config_dict.ConfigDict()

  config.dropout = kwargs.get('dropout', 0.5)

  config.max_seq_length = kwargs.get('max_seq_length', 40)

  config.vocab_size = kwargs.get('vocab_size', 500)
  config.embed_size = kwargs.get('embed_size', 300)
  config.trainable_embedding = kwargs.get('trainable_embedding', True)
  config.shared_embedding = kwargs.get('shared_embedding', True)
  config.word_embedding_path = kwargs.get('word_embedding_path')
  config.shared_bert_embedding = kwargs.get('shared_bert_embedding', False)
  config.shared_bert_embedding_ckpt_dir = kwargs.get(
      'shared_bert_embedding_ckpt_dir', '')
  config.shared_bert_embedding_config = kwargs.get(
      'shared_bert_embedding_config', {})

  config.encoder_hidden_size = kwargs.get('encoder_hidden_size', 400)
  config.encoder_cell_type = kwargs.get('encoder_cell_type', 'lstm')
  config.num_ecnoder_rnn_layers = kwargs.get('num_ecnoder_rnn_layers', 1)
  config.encoder_projection_sizes = kwargs.get('encoder_projection_sizes',
                                               (400, 200))

  config.temperature = kwargs.get('temperature', 5)
  config.sampler_post_processor_output_sizes = kwargs.get(
      'sampler_post_processor_output_sizes', (200, 200))

  config.num_states = kwargs.get('num_states', 10)
  config.state_updater_cell_type = kwargs.get('state_updater_cell_type', 'lstm')

  config.decoder_hidden_size = (
      config.num_states + config.sampler_post_processor_output_sizes[-1])
  config.decoder_cell_type = kwargs.get('decoder_cell_type', 'lstm')

  config.gumbel_softmax_label_adjustment_multiplier = kwargs.get(
      'gumbel_softmax_label_adjustment_multiplier', 0)

  return config


def vanilla_linear_vrnn_config(**kwargs) -> VanillaLinearVRNNConfig:
  """Creates model config for VanillaLinearVRNN."""
  config = config_dict.ConfigDict()

  vae_cell_data = kwargs.get('vae_cell', kwargs)
  config.vae_cell = vanilla_linear_vae_cell_config(**vae_cell_data)

  config.max_dialog_length = kwargs.get('max_dialog_length', 13)

  config.dropout = config.vae_cell.dropout
  config.num_states = config.vae_cell.num_states
  config.vocab_size = config.vae_cell.vocab_size

  config.prior_latent_state_updater_hidden_size = kwargs.get(
      'prior_latent_state_updater_hidden_size', (100, 100))
  config.with_direct_transition = kwargs.get('with_direct_transition', False)
  config.with_bow = kwargs.get('with_bow', True)
  config.bow_hidden_sizes = kwargs.get('bow_hidden_sizes', (400,))

  return config

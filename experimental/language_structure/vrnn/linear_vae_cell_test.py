# coding=utf-8
# Copyright 2023 The Uncertainty Baselines Authors.
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

"""Tests for linear_vae_cell."""

from absl.testing import absltest
import tensorflow as tf
import tensorflow_hub as hub
from vrnn import linear_vae_cell  # local file import from experimental.language_structure
from vrnn import model_config  # local file import from experimental.language_structure

from official.nlp.bert import configs


class LinearVaeCellTest(absltest.TestCase):

  def _build_embedding_layer(self, vocab_size, embed_size, max_seq_length):
    return linear_vae_cell._Embedding(
        linear_vae_cell.INPUT_ID_NAME,
        vocab_size,
        embed_size,
        input_length=max_seq_length,
        trainable=True)

  def test_dual_lstm_encoder_shape(self):
    vocab_size = 8
    embed_size = 6
    max_seq_length = 4
    hidden_size = 10
    num_layers = 2
    test_model = linear_vae_cell.DualRNNEncoder(
        hidden_size=hidden_size,
        num_layers=num_layers,
        embedding_layer=self._build_embedding_layer(vocab_size, embed_size,
                                                    max_seq_length),
        return_state=True)

    inputs = [{
        'input_word_ids': tf.keras.Input(shape=(max_seq_length,)),
        'input_mask': tf.keras.Input(shape=(max_seq_length,)),
    }, {
        'input_word_ids': tf.keras.Input(shape=(max_seq_length,)),
    }, [tf.keras.Input(shape=(hidden_size)) for _ in range(num_layers)]]
    output_1, output_2, state_1, state_2 = test_model(*inputs)

    for output in [output_1, output_2]:
      self.assertEqual([None, hidden_size], output.shape.as_list())

    for output in [state_1, state_2]:
      self.assertLen(output, num_layers)
      for state in output:
        self.assertEqual([None, hidden_size], state.shape.as_list())

  def test_dual_gru_encoder_shape(self):
    vocab_size = 8
    embed_size = 6
    max_seq_length = 4
    hidden_size = 10
    num_layers = 3
    test_model = linear_vae_cell.DualRNNEncoder(
        hidden_size=hidden_size,
        num_layers=num_layers,
        embedding_layer=self._build_embedding_layer(vocab_size, embed_size,
                                                    max_seq_length),
        cell_type='gru',
        return_state=True)

    inputs = [{
        'input_word_ids': tf.keras.Input(shape=(max_seq_length,)),
        'input_mask': tf.keras.Input(shape=(max_seq_length,)),
    }, {
        'input_word_ids': tf.keras.Input(shape=(max_seq_length,)),
        'input_mask': tf.keras.Input(shape=(max_seq_length,)),
    }, [tf.keras.Input(shape=(hidden_size)) for _ in range(num_layers)]]
    output_1, output_2, state_1, state_2 = test_model(*inputs)

    for output in [output_1, output_2]:
      self.assertEqual([None, hidden_size], output.shape.as_list())

    for output in [state_1, state_2]:
      self.assertLen(output, num_layers)
      for state in output:
        self.assertEqual([None, hidden_size], state.shape.as_list())

  def test_dual_lstm_decoder_shape(self):
    vocab_size = 8
    embed_size = 6
    max_seq_length = 4
    hidden_size = 10
    num_layers = 1
    test_model = linear_vae_cell.DualRNNDecoder(
        hidden_size=hidden_size,
        num_layers=num_layers,
        embedding_layer=self._build_embedding_layer(vocab_size, embed_size,
                                                    max_seq_length),
        return_state=True)

    inputs = [{
        'input_word_ids': tf.keras.Input(shape=(max_seq_length,)),
        'input_mask': tf.keras.Input(shape=(max_seq_length,)),
    }, {
        'input_word_ids': tf.keras.Input(shape=(max_seq_length,)),
        'input_mask': tf.keras.Input(shape=(max_seq_length,)),
    }, [tf.keras.Input(shape=(hidden_size)) for _ in range(num_layers)]]
    output_1, output_2, state_1, state_2 = test_model(*inputs)

    expected_output_shape = [None, max_seq_length, vocab_size]

    for output in [output_1, output_2]:
      self.assertEqual(expected_output_shape, output.shape.as_list())

    for output in [state_1, state_2]:
      self.assertLen(output, num_layers)

    for state in state_1:
      self.assertEqual([None, hidden_size], state.shape.as_list())

    for state in state_2:
      self.assertEqual([None, hidden_size * 2], state.shape.as_list())

  def test_dual_gru_decoder_shape(self):
    vocab_size = 8
    embed_size = 6
    max_seq_length = 4
    hidden_size = 10
    test_model = linear_vae_cell.DualRNNDecoder(
        hidden_size=hidden_size,
        embedding_layer=self._build_embedding_layer(vocab_size, embed_size,
                                                    max_seq_length),
        cell_type='gru',
        return_state=True)

    inputs = [{
        'input_word_ids': tf.keras.Input(shape=(max_seq_length,)),
    }, {
        'input_word_ids': tf.keras.Input(shape=(max_seq_length,)),
        'input_mask': tf.keras.Input(shape=(max_seq_length,)),
    }, [tf.keras.Input(shape=(hidden_size))]]
    output_1, output_2, state_1, state_2 = test_model(*inputs)

    expected_output_shape = [None, max_seq_length, vocab_size]

    for output in [output_1, output_2]:
      self.assertEqual(expected_output_shape, output.shape.as_list())

    for output in [state_1, state_2]:
      self.assertLen(output, 1)

    for state in state_1:
      self.assertEqual([None, hidden_size], state.shape.as_list())

    for state in state_2:
      self.assertEqual([None, hidden_size * 2], state.shape.as_list())

  def test_vanilla_vae_cell_shape(self):
    config = model_config.vanilla_linear_vae_cell_config(
        num_ecnoder_rnn_layers=2)
    test_model = linear_vae_cell.VanillaLinearVAECell(config)

    inputs = [{
        'input_word_ids':
            tf.keras.Input(shape=(config.max_seq_length,), dtype=tf.int32),
        'input_mask':
            tf.keras.Input(shape=(config.max_seq_length,), dtype=tf.int32),
    }] * 4 + [
        (tf.keras.Input(shape=(config.num_states,)),
         tf.keras.Input(shape=(config.num_states,))),
        tf.keras.Input(shape=(config.num_states,), dtype=tf.int32),
        tf.keras.Input(shape=(), dtype=tf.int32),
    ]
    outputs = test_model(inputs, return_states=True, return_samples=True)
    self.assertLen(outputs, 9)
    (sampler_inputs, decoder_outputs_1, decoder_outputs_2, next_state,
     latent_state, decoder_initial_state, decoder_state_1, decoder_state_2,
     samples) = outputs

    self.assertEqual([None, config.num_states], sampler_inputs.shape.as_list())
    self.assertEqual([None, config.num_states], samples.shape.as_list())
    self.assertEqual([None, config.encoder_projection_sizes[-1]],
                     latent_state.shape.as_list())

    self.assertEqual(
        [None, config.max_seq_length - 1, config.decoder_embedding.vocab_size],
        decoder_outputs_1.shape.as_list())
    self.assertEqual(
        [None, config.max_seq_length - 1, config.decoder_embedding.vocab_size],
        decoder_outputs_2.shape.as_list())

    self.assertEqual([None, config.num_states], next_state[0].shape.as_list())
    self.assertEqual([None, config.num_states], next_state[1].shape.as_list())
    self.assertEqual([None, config.decoder_hidden_size],
                     decoder_initial_state.shape.as_list())
    self.assertEqual([None, config.decoder_hidden_size],
                     decoder_state_1.shape.as_list())
    self.assertEqual([None, config.decoder_hidden_size * 2],
                     decoder_state_2.shape.as_list())

  def test_bert_shape(self):
    max_seq_length = 10
    bert_config = configs.BertConfig(
        **{
            'hidden_size': 128,
            'hidden_act': 'gelu',
            'initializer_range': 0.02,
            'vocab_size': 30522,
            'hidden_dropout_prob': 0.1,
            'num_attention_heads': 2,
            'type_vocab_size': 2,
            'max_position_embeddings': 512,
            'num_hidden_layers': 2,
            'intermediate_size': 512,
            'attention_probs_dropout_prob': 0.1
        })
    test_model = linear_vae_cell._BERT(
        max_seq_length, bert_config, trainable=False)

    preprocess_tfhub_url = 'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3'
    text_input = tf.keras.layers.Input(shape=(), dtype=tf.string)
    preprocessor = hub.KerasLayer(preprocess_tfhub_url)
    encoder_inputs = preprocessor(text_input)
    sequence_outputs = test_model(encoder_inputs)
    pooled_outputs = test_model(encoder_inputs, return_sequence=False)

    self.assertEqual([None, 128, 128], sequence_outputs.shape.as_list())
    self.assertEqual([None, 128], pooled_outputs.shape.as_list())


if __name__ == '__main__':
  absltest.main()

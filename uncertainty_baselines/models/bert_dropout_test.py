# coding=utf-8
# Copyright 2020 The Uncertainty Baselines Authors.
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

"""Tests for BERT ,podel with Monte Carlo Dropout."""
from absl.testing import parameterized

import tensorflow as tf
import uncertainty_baselines as ub

from uncertainty_baselines.models import bert_dropout
from official.nlp.bert import configs as bert_configs


class DropoutModelBertTest(tf.test.TestCase, parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.random_seed = 42

    self.batch_size = 8
    self.num_heads = 2
    self.key_dim = 8
    self.seq_length = 6
    self.hidden_dim = 16
    self.num_classes = 10
    self.dropout_rate = 0.1

    self.bert_test_config = bert_configs.BertConfig(
        attention_probs_dropout_prob=0.12,
        hidden_dropout_prob=0.34,
        hidden_act='gelu',
        hidden_size=self.hidden_dim,
        initializer_range=0.02,
        intermediate_size=self.hidden_dim,
        max_position_embeddings=self.seq_length,
        num_attention_heads=self.num_heads,
        num_hidden_layers=2,
        type_vocab_size=2,
        vocab_size=30522)

    # Typical shape of a layer output tensor.
    self.input_shape_3d = (self.seq_length, self.hidden_dim)
    # Typical shape of an attention score tensor.
    self.input_shape_4d = (self.num_heads, self.seq_length, self.seq_length)

    self.input_3d = tf.random.normal((self.batch_size,) + self.input_shape_3d)
    self.input_4d = tf.random.normal((self.batch_size,) + self.input_shape_4d)

  @parameterized.named_parameters(('no_mc_dropout_3d', 3, False, False),
                                  ('no_mc_dropout_4d', 4, False, False),
                                  ('classic_dropout_3d', 3, True, False),
                                  ('classic_dropout_4d', 4, True, False),
                                  ('channel_wise_dropout_3d', 3, True, True),
                                  ('channel_wise_dropout_4d', 4, True, True))
  def test_monte_carlo_dropout(self, input_dim, use_mc_dropout,
                               channel_wise_dropout):
    """Tests if MC dropout can be correctly enabled and disabled."""
    inputs_tensor = self.input_4d if input_dim == 4 else self.input_3d
    inputs_shape = inputs_tensor.shape[1:]

    # Compiles dropout model.
    inputs = tf.keras.Input(shape=inputs_shape, batch_size=self.batch_size)
    outputs_dropout = bert_dropout._monte_carlo_dropout(
        inputs,
        self.dropout_rate,
        use_mc_dropout=use_mc_dropout,
        channel_wise_dropout=channel_wise_dropout)
    model = tf.keras.Model(inputs=inputs, outputs=outputs_dropout)

    # Computes dropout output.
    tf.random.set_seed(self.random_seed)
    outputs_tensor = model(inputs_tensor, training=False)

    if use_mc_dropout:
      self.assertNotAllClose(inputs_tensor, outputs_tensor)
    else:
      self.assertAllClose(inputs_tensor, outputs_tensor)

  @parameterized.named_parameters(('inputs_shape_2', (12,)),
                                  ('inputs_shape_5', (4, 12, 12, 32)))
  def test_monte_carlo_dropout_input_shape(self, inputs_dim_shape):
    """Tests if monte_carlo_dropout captures wrong input shape."""
    with self.assertRaises(ValueError):
      # Total input shape is inputs_dim_shape + 1.
      inputs = tf.keras.Input(
          shape=inputs_dim_shape, batch_size=self.batch_size)
      outputs_dropout = bert_dropout._monte_carlo_dropout(
          inputs,
          self.dropout_rate,
          use_mc_dropout=True,
          channel_wise_dropout=True)
      _ = tf.keras.Model(inputs=inputs, outputs=outputs_dropout)

  @parameterized.named_parameters(('no_mc_dropout', False, False),
                                  ('classic_dropout', True, False),
                                  ('channel_wise_dropout', True, True))
  def test_multihead_attention(self, use_mc_dropout, channel_wise_dropout):
    """Tests if DropoutMultiHeadAttention can be compiled successfully."""
    inputs_tensor = self.input_3d
    inputs_shape = inputs_tensor.shape[1:]

    # Compiles model and get output.
    inputs = tf.keras.Input(shape=inputs_shape, batch_size=self.batch_size)
    outputs_multihead_attention = bert_dropout.DropoutMultiHeadAttention(
        num_heads=self.num_heads,
        key_dim=self.key_dim,
        dropout=self.dropout_rate,
        use_mc_dropout=use_mc_dropout,
        channel_wise_dropout=channel_wise_dropout)(inputs, inputs, inputs)
    model = tf.keras.Model(inputs=inputs, outputs=outputs_multihead_attention)
    outputs_tensor = model(inputs_tensor, training=False)

    # Compares shape.
    output_shape_expected = [self.batch_size, self.seq_length, self.hidden_dim]
    output_shape_observed = outputs_tensor.shape.as_list()

    self.assertListEqual(output_shape_observed, output_shape_expected)

  @parameterized.named_parameters(('no_mc_dropout', False, False),
                                  ('classic_dropout', True, False),
                                  ('channel_wise_dropout', True, True))
  def test_transformer(self, use_mc_dropout, channel_wise_dropout):
    """Tests if DropoutTransformer can be compiled successfully."""
    inputs_tensor = self.input_3d
    inputs_shape = inputs_tensor.shape[1:]

    # Compiles model and get output.
    inputs = tf.keras.Input(shape=inputs_shape, batch_size=self.batch_size)
    outputs_transformer = bert_dropout.DropoutTransformer(
        num_attention_heads=self.num_heads,
        intermediate_size=self.hidden_dim,
        intermediate_activation=tf.nn.relu,
        dropout_rate=self.dropout_rate,
        attention_dropout_rate=self.dropout_rate,
        use_mc_dropout_mha=use_mc_dropout,
        use_mc_dropout_att=use_mc_dropout,
        use_mc_dropout_ffn=use_mc_dropout,
        channel_wise_dropout_mha=channel_wise_dropout,
        channel_wise_dropout_att=channel_wise_dropout,
        channel_wise_dropout_ffn=channel_wise_dropout)(inputs)
    model = tf.keras.Model(inputs=inputs, outputs=outputs_transformer)
    outputs_tensor = model(inputs_tensor, training=False)

    # Compares shape.
    output_shape_expected = [self.batch_size, self.seq_length, self.hidden_dim]
    output_shape_observed = outputs_tensor.shape.as_list()

    self.assertListEqual(output_shape_observed, output_shape_expected)

  @parameterized.named_parameters(('no_mc_dropout', False),
                                  ('with_mc_dropout', True))
  def test_bert_classifier(self, use_mc_dropout):
    """Tests if DropoutBertClassifier can be compiled successfully."""
    # Compile a mock input model
    inputs = tf.keras.Input(shape=self.seq_length, batch_size=self.batch_size)
    outputs = tf.keras.layers.Lambda(lambda x: x)(inputs)
    network = tf.keras.Model(inputs=inputs, outputs=[outputs, outputs])

    # Compiles dropout model.
    model = bert_dropout.DropoutBertClassifier(
        network,
        num_classes=self.num_classes,
        dropout_rate=self.dropout_rate,
        use_mc_dropout=use_mc_dropout)

    # Computes dropout output.
    tf.random.set_seed(self.random_seed)
    inputs_tensor = tf.random.normal((self.batch_size, self.seq_length))
    outputs_tensor = model(inputs_tensor, training=False)

    output_shape_expected = [self.batch_size, self.num_classes]
    output_shape_observed = outputs_tensor.shape.as_list()
    self.assertEqual(output_shape_observed, output_shape_expected)

  def test_get_mc_dropout_transformer_encoder(self):
    """Tests if bert_config can be correctly passed into TransformerEncoder."""
    bert_encoder = bert_dropout.get_mc_dropout_transformer_encoder(
        self.bert_test_config,
        use_mc_dropout_mha=True,
        use_mc_dropout_att=True,
        use_mc_dropout_ffn=True,
        channel_wise_dropout_mha=True,
        channel_wise_dropout_att=True,
        channel_wise_dropout_ffn=True)

    hidden_layers = bert_encoder.hidden_layers
    example_hidden_layer = hidden_layers[0]
    example_attention_layer = example_hidden_layer._attention_layer

    self.assertLen(hidden_layers, self.bert_test_config.num_hidden_layers)
    self.assertIsInstance(example_hidden_layer,
                          bert_dropout.DropoutTransformer)
    self.assertIsInstance(example_attention_layer,
                          bert_dropout.DropoutMultiHeadAttention)
    self.assertEqual(example_hidden_layer._dropout_rate,
                     self.bert_test_config.hidden_dropout_prob)
    self.assertEqual(example_attention_layer._dropout,
                     self.bert_test_config.attention_probs_dropout_prob)

  def test_create_model(self):
    """Integration test for create_model."""
    bert_model, bert_encoder = ub.models.DropoutBertBuilder(
        num_classes=self.num_classes,
        bert_config=self.bert_test_config,
        use_mc_dropout_mha=True,
        use_mc_dropout_att=True,
        use_mc_dropout_ffn=True,
        channel_wise_dropout_mha=True,
        channel_wise_dropout_att=True,
        channel_wise_dropout_ffn=True)

    self.assertIsInstance(bert_model, bert_dropout.DropoutBertClassifier)
    self.assertIsInstance(bert_encoder,
                          bert_dropout.DropoutTransformerEncoder)


if __name__ == '__main__':
  tf.test.main()

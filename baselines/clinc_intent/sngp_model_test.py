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

# Lint as: python3
"""Tests for sngp_model."""
from absl.testing import parameterized

import numpy as np
import tensorflow as tf

import sngp_model  # local file import
from official.nlp.bert import configs as bert_configs

SNFeedforward = sngp_model.SpectralNormalizedFeedforwardLayer
SNAttention = sngp_model.SpectralNormalizedMultiHeadAttention
SNTransformer = sngp_model.SpectralNormalizedTransformer


def _compute_spectral_norm(weight):
  """Computes the spectral norm for a numpy weight matrix."""
  # TODO(b/165683434): Support different re-shaping options.
  if weight.ndim > 2:
    # Reshape weight to a 2D matrix.
    weight_shape = weight.shape
    weight = weight.reshape((-1, weight_shape[-1]))
  return np.max(np.linalg.svd(weight, compute_uv=False))


def _compute_layer_spectral_norms(layer):
  """Computes the spectral norm for all kernels in a layer."""
  return [
      _compute_spectral_norm(weight.numpy())
      for weight in layer.weights
      if 'kernel' in weight.name
  ]


class SngpModelTest(tf.test.TestCase, parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.random_seed = 42

    self.batch_size = 4
    self.seq_length = 12
    self.hidden_dim = 32
    self.num_heads = 8
    self.key_dim = self.hidden_dim // self.num_heads

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
        vocab_size=128)

    self.input_shape_3d = tf.TensorShape(
        (self.batch_size, self.seq_length, self.hidden_dim))
    self.input_shape_4d = tf.TensorShape(
        (self.batch_size, self.seq_length, self.num_heads, self.key_dim))

    # Layer arguments.
    self.sn_norm_multiplier = 0.15
    self.spec_norm_kwargs = dict(
        iteration=1000, norm_multiplier=self.sn_norm_multiplier)
    self.attention_kwargs = dict(num_heads=self.num_heads, key_dim=self.key_dim)
    self.feedforward_kwargs = dict(
        intermediate_size=1024,
        intermediate_activation='gelu',
        dropout=0.1,
        use_layer_norm=True)

  def test_make_spec_norm_dense_layer(self):
    """Tests if the weights of spec_norm_dense_layer is correctly normalized."""
    # For a input sequence tensor [batch_size, a, b], defines a matrix
    # multiplication op (along hidden dimension b) in eisum notation.
    einsum_equation = 'abc,cd->abd'

    eisum_layer_class = sngp_model.make_spec_norm_dense_layer(
        **self.spec_norm_kwargs)
    dense_layer = eisum_layer_class(
        output_shape=(self.seq_length, 10),
        equation=einsum_equation,
        activation='relu')

    # Perform normalization.
    dense_layer.build(self.input_shape_3d)
    dense_layer.update_weights()
    normalized_kernel = dense_layer.layer.kernel.numpy()

    spectral_norm_computed = _compute_spectral_norm(normalized_kernel)
    self.assertAllClose(
        spectral_norm_computed, self.sn_norm_multiplier, atol=1e-3)

  @parameterized.named_parameters(('feedforward', False), ('attention', True))
  def test_layer_spectral_normalization(self, test_attention):
    """Tests if the layer weights can be correctly normalized."""
    layer_class = SNAttention if test_attention else SNFeedforward
    input_shape = self.input_shape_4d if test_attention else self.input_shape_3d
    kwargs = self.attention_kwargs if test_attention else self.feedforward_kwargs

    # Create input data.
    tf.random.set_seed(self.random_seed)
    random_data = tf.random.normal(input_shape)
    input_tensors = (random_data,) * 2 if test_attention else (random_data,)

    layer_instance = layer_class(
        use_spec_norm=True, spec_norm_kwargs=self.spec_norm_kwargs, **kwargs)

    # Invoke spectral normalization via model call.
    _ = layer_instance(*input_tensors)

    spec_norm_list_observed = _compute_layer_spectral_norms(layer_instance)
    spec_norm_list_expected = [self.sn_norm_multiplier
                              ] * len(spec_norm_list_observed)

    self.assertAllClose(spec_norm_list_observed, spec_norm_list_expected,
                        atol=1e-3)

  @parameterized.named_parameters(('att_and_ffn', True, True),
                                  ('att_only', False, True),
                                  ('ffn_only', True, False))
  def test_transformer_spectral_normalization(self, use_spec_norm_att,
                                              use_spec_norm_ffn):
    """Tests if the transorfmer weights can be correctly normalized."""
    tf.random.set_seed(self.random_seed)
    input_tensor = tf.random.normal(self.input_shape_3d)

    transformer_model = SNTransformer(
        num_attention_heads=self.num_heads,
        intermediate_size=self.hidden_dim,
        intermediate_activation='gelu',
        use_layer_norm_att=False,
        use_layer_norm_ffn=False,
        use_spec_norm_att=use_spec_norm_att,
        use_spec_norm_ffn=use_spec_norm_ffn,
        spec_norm_kwargs=self.spec_norm_kwargs)
    _ = transformer_model(input_tensor)

    spec_norm_list_all = _compute_layer_spectral_norms(transformer_model)

    # Collect spectral norms of the normalized kernel matrices.
    spec_norm_list_observed = []
    if use_spec_norm_att:
      spec_norm_list_observed += spec_norm_list_all[:3]
    if use_spec_norm_ffn:
      spec_norm_list_observed += spec_norm_list_all[-2:]
    spec_norm_list_expected = [self.sn_norm_multiplier
                              ] * len(spec_norm_list_observed)

    self.assertAllClose(
        spec_norm_list_observed, spec_norm_list_expected, atol=1e-3)

  def test_transformer_encoder_spectral_normalization(self):
    """Tests if the transorfmer encoder weights are correctly normalized."""
    input_ids = tf.ones((self.batch_size, self.seq_length), dtype=tf.int32)
    input_tensors = [input_ids, input_ids, input_ids]

    transformer_encoder = (
        sngp_model.get_spectral_normalized_transformer_encoder(
            bert_config=self.bert_test_config,
            spec_norm_kwargs=self.spec_norm_kwargs,
            use_layer_norm_att=True,
            use_layer_norm_ffn=True,
            use_spec_norm_att=True,
            use_spec_norm_ffn=True))
    _ = transformer_encoder(input_tensors)

    # Currently the model does not apply spectral normalization to the CLS
    # pooler layer (i.e. the last trainable kernel matrix). Remove it from
    # evaluation for now.
    spec_norm_list_observed = _compute_layer_spectral_norms(transformer_encoder)
    del spec_norm_list_observed[-1]
    spec_norm_list_expected = [self.sn_norm_multiplier
                              ] * len(spec_norm_list_observed)

    self.assertAllClose(
        spec_norm_list_observed, spec_norm_list_expected, atol=1e-3)


if __name__ == '__main__':
  tf.test.main()

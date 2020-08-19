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

SNFeedforward = sngp_model.SpectralNormalizedFeedforwardLayer


def _compute_spectral_norm(weight):
  """Computes the spectral norm for a numpy weight matrix."""
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
    self.max_seq_length = 12
    self.hidden_dim = 32
    self.num_heads = 8
    self.key_dim = self.hidden_dim // self.num_heads

    self.input_shape_3d = tf.TensorShape(
        (self.batch_size, self.max_seq_length, self.hidden_dim))
    self.input_shape_4d = tf.TensorShape(
        (self.batch_size, self.max_seq_length, self.num_heads, self.key_dim))

    # Layer arguments.
    self.sn_norm_multiplier = 0.95
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
        output_shape=(self.max_seq_length, 10),
        equation=einsum_equation,
        activation='relu')

    # Perform normalization.
    dense_layer.build(self.input_shape_3d)
    dense_layer.update_weights()
    normalized_kernel = dense_layer.layer.kernel.numpy()

    spectral_norm_computed = _compute_spectral_norm(normalized_kernel)
    self.assertAllClose(
        spectral_norm_computed, self.sn_norm_multiplier, atol=1e-3)

  def test_layer_spectral_normalization(self):
    """Tests if the layer weights can be correctly normalized."""
    # Create input data.
    tf.random.set_seed(self.random_seed)
    input_tensors = tf.random.normal(self.input_shape_3d)

    layer_instance = SNFeedforward(
        use_spec_norm=True,
        spec_norm_kwargs=self.spec_norm_kwargs,
        **self.feedforward_kwargs)

    # Invoke spectral normalization via model call.
    _ = layer_instance(input_tensors)

    spec_norm_list_observed = _compute_layer_spectral_norms(layer_instance)
    spec_norm_list_expected = [self.sn_norm_multiplier] * 2

    self.assertAllClose(
        spec_norm_list_observed, spec_norm_list_expected, atol=1e-3)


if __name__ == '__main__':
  tf.test.main()

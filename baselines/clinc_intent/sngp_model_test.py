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
"""Tests for sngp_model.

## References:

[1] Hanie Sedghi, Vineet Gupta, Philip M. Long.
    The Singular Values of Convolutional Layers.
    In _International Conference on Learning Representations_, 2019.
"""
from absl.testing import parameterized

import numpy as np
import tensorflow as tf

import sngp_model  # local file import


def _compute_spectral_norm(weight):
  if weight.ndim > 2:
    # Computes Conv2D via FFT transform as in [1].
    weight = np.fft.fft2(weight, weight.shape[1:3], axes=[0, 1])
  return np.max(np.linalg.svd(weight, compute_uv=False))


class SngpModelTest(tf.test.TestCase, parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.batch_size = 4
    self.max_seq_length = 12
    self.hidden_dim = 32
    self.input_shape = (self.batch_size, self.max_seq_length, self.hidden_dim)
    self.input_tensor = tf.random.normal(self.input_shape)

    # For a input sequence tensor [batch_size, a, b], defines a matrix
    # multiplication op (along hidden dimension b) in eisum notation.
    self.einsum_equation = 'abc,cd->abd'

    self.sn_iterations = 1000
    self.sn_norm_multiplier = 0.95

  def test_make_spec_norm_dense_layer(self):
    """Tests if the weights of spec_norm_dense_layer is correctly normalized."""
    eisum_layer_class = sngp_model.make_spec_norm_dense_layer(
        iteration=self.sn_iterations,
        norm_multiplier=self.sn_norm_multiplier)
    dense_layer = eisum_layer_class(
        output_shape=(self.max_seq_length, 10),
        equation=self.einsum_equation,
        activation='relu')

    # Perform normalization.
    dense_layer.build(self.input_shape)
    dense_layer.update_weights()
    normalized_kernel = dense_layer.layer.kernel.numpy()

    spectral_norm_computed = _compute_spectral_norm(normalized_kernel)
    self.assertAllClose(spectral_norm_computed,
                        self.sn_norm_multiplier, atol=1e-3)

if __name__ == '__main__':
  tf.test.main()

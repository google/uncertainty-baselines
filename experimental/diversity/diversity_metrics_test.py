# coding=utf-8
# Copyright 2024 The Uncertainty Baselines Authors.
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

"""Tests for diversity_metrics."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
from absl.testing import absltest
from absl.testing import parameterized

import numpy as np
import sklearn.metrics
import tensorflow.compat.v2 as tf

import diversity_metrics  # local file import from experimental.diversity


class DiversityMetricsTest(parameterized.TestCase, tf.test.TestCase):

  def setUp(self):
    np.random.seed(0)
    tf.random.set_seed(0)
    super().setUp()

  def test_pairwise_cosine_similarity(self):
    x = np.random.rand(3, 2)
    computed_matrix = diversity_metrics.pairwise_cosine_similarity(
        x, normalize=True)
    x_normalized = x / np.linalg.norm(x, axis=-1)[:, np.newaxis]
    expected_matrix = sklearn.metrics.pairwise.cosine_similarity(x_normalized)
    self.assertAllClose(expected_matrix, computed_matrix)

  def test_pairwise_euclidean_distances(self):
    x = np.random.rand(3, 2)
    computed_matrix = diversity_metrics.pairwise_euclidean_distances(
        x, normalize=True)
    x_normalized = x / np.linalg.norm(x, axis=-1)[:, np.newaxis]
    expected_matrix = sklearn.metrics.pairwise.euclidean_distances(
        x_normalized, squared=True)
    self.assertAllClose(expected_matrix, computed_matrix)

  def test_pairwise_l1_distance(self):
    x = np.random.rand(3, 2)
    computed_matrix = diversity_metrics.pairwise_l1_distances(x, normalize=True)
    x_normalized = x / np.linalg.norm(x, axis=-1)[:, np.newaxis]
    expected_matrix = sklearn.metrics.pairwise.manhattan_distances(x_normalized)
    self.assertAllClose(expected_matrix, computed_matrix)

  def test_compute_laplacian_kernel(self):
    x = np.random.rand(3, 2)
    computed_matrix = diversity_metrics.compute_laplacian_kernel(
        x, normalize=True)
    x_normalized = x / np.linalg.norm(x, axis=-1)[:, np.newaxis]
    expected_matrix = sklearn.metrics.pairwise.laplacian_kernel(x_normalized)
    self.assertAllClose(expected_matrix, computed_matrix)

  @parameterized.parameters(
      {
          'kernel':
              'linear',
          'kernel_method':
              functools.partial(diversity_metrics.pairwise_cosine_similarity)
      },
      {
          'kernel':
              'rbf',
          'kernel_method':
              functools.partial(diversity_metrics.compute_rbf_kernel)
      },
      {
          'kernel':
              'l1',
          'kernel_method':
              functools.partial(diversity_metrics.compute_laplacian_kernel)
      },
  )
  def test_dpp_log_determinant(self, kernel, kernel_method):
    x = np.random.rand(3, 2)
    kernel_method_result = kernel_method(
        x) + tf.keras.backend.epsilon() * tf.linalg.eye(
            3, dtype=tf.double)
    expected_log_det = tf.linalg.logdet(kernel_method_result)
    computed_log_det = -diversity_metrics.dpp_negative_logdet(
        x, kernel=kernel, bandwidth=0., normalize=True)
    self.assertAlmostEqual(expected_log_det, computed_log_det)


if __name__ == '__main__':
  tf.enable_v2_behavior()
  absltest.main()

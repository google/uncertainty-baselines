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

"""Metrics for diversity."""

import tensorflow.compat.v2 as tf


def pairwise_euclidean_distances(x: tf.Tensor,
                                 normalize: bool = True) -> tf.Tensor:
  """Compute the matrix of pairwise euclidean distances."""
  # TODO(ghassen): this only works for 2D matrices (without a batch dimension).
  if len(x.shape) > 2:
    raise NotImplementedError('Cannot compute the pairwise L2 distance for'
                              'matrices of higher dimension than 2. '
                              'len(x.shape) = {} '.format(len(x.shape)))
  if normalize:
    x = tf.math.l2_normalize(x, axis=-1)
    # This is the most memory and compute-efficient approach to compute pairwise
    # distances but is sofar limited to 2D tensors (without a batch dimension)
    # as it cannot handle dynamic batch_sizes.
  squared_x = tf.reduce_sum(x * x, axis=1)
  squared_x = tf.reshape(squared_x, [-1, 1])
  squared_diff = squared_x - 2 * tf.matmul(
      x, x, transpose_b=True) + tf.transpose(squared_x)
  return squared_diff


def pairwise_cosine_similarity(x: tf.Tensor,
                               normalize: bool = True) -> tf.Tensor:
  """Compute the pairwise cosine similarity matrix."""
  if len(x.shape) > 2:
    raise NotImplementedError('Cannot compute the pairwise L2 distance for'
                              'matrices of higher dimension than 2. '
                              'len(x.shape) = {} '.format(len(x.shape)))
  if normalize:
    x = tf.math.l2_normalize(x, axis=-1)
  result = tf.matmul(x, x, transpose_b=True)
  return result


def cosine_similarity(x: tf.Tensor, normalize: bool = True) -> tf.Tensor:
  """Compute the sum of pairwise_cosine_similarity matrix."""
  similarity_matrix = pairwise_cosine_similarity(x, normalize=normalize)
  # Sum over the features then average over the ensemble members.
  return tf.reduce_mean(tf.reduce_sum(similarity_matrix, axis=-1))


def cosine_distance(x: tf.Tensor, normalize: bool = True) -> tf.Tensor:
  return 1.0 - pairwise_cosine_similarity(x, normalize=normalize)


def pairwise_l1_distances(x: tf.Tensor, normalize: bool = True) -> tf.Tensor:
  if normalize:
    x = tf.math.l2_normalize(x, axis=-1)
  expanded_a = tf.expand_dims(x, 1)
  expanded_b = tf.expand_dims(x, 0)
  abs_diff = tf.math.abs(expanded_a - expanded_b)
  result = tf.reduce_sum(abs_diff, 2)
  return result


def compute_rbf_kernel(x: tf.Tensor,
                       bandwidth: float = 0.,
                       normalize: bool = True) -> tf.Tensor:
  # TODO(ghassen): try the median/quantiles heuristic for setting rbf bandwidth.
  if not bandwidth:
    bandwidth = 1. / x.shape[-1]  # 1. / num_features
  rbf_matrix = pairwise_euclidean_distances(x, normalize=normalize)
  return tf.math.exp(-bandwidth * rbf_matrix)


def compute_laplacian_kernel(x: tf.Tensor,
                             bandwidth: float = 0.,
                             normalize: bool = True) -> tf.Tensor:
  if not bandwidth:
    bandwidth = 1. / x.shape[-1]  # 1. / num_features
  laplacian_matrix = pairwise_l1_distances(x, normalize=normalize)
  return tf.math.exp(-bandwidth * laplacian_matrix)


def dpp_negative_logdet(x: tf.Tensor,
                        kernel: str = 'rbf',
                        bandwidth: float = 0.,
                        normalize: bool = True) -> tf.Tensor:
  """Computes the log determinant of a DPP similarity."""
  # If feature vectors have different norms, the DPP will be biased towards
  # selecting large-norm examples. Therefore, the kernel methods normalize the
  # features which leads to equivalence between certain kernels.
  if kernel in ['l2', 'rbf', 'gaussian']:
    similarity_matrix = compute_rbf_kernel(x, bandwidth, normalize=normalize)
  # Note: On L2-normalized data, cos similarity is equivalent to linear_kernel.
  elif kernel in ['linear', 'dot-product'] or (kernel == 'cos-sim' and
                                               normalize):
    similarity_matrix = pairwise_cosine_similarity(x, normalize=normalize)
  elif kernel in ['l1', 'laplacian', 'manhattan']:
    similarity_matrix = compute_laplacian_kernel(
        x, bandwidth, normalize=normalize)
  else:
    raise ValueError('Could not recognize kernel = {}: not in [l2, rbf, '
                     'gaussian, linear, dot-product, cos-sim, l1, laplacian'
                     'manhattan'.format(kernel))
  # This computes a numerically stable log determinant.
  similarity_matrix = tf.cast(similarity_matrix, dtype=tf.float64)
  similarity_matrix += tf.keras.backend.epsilon() * tf.linalg.eye(
      similarity_matrix.shape[0], dtype=tf.double)
  return -2.0 * tf.reduce_sum(
      tf.math.log(tf.linalg.diag_part(tf.linalg.cholesky(similarity_matrix))),
      axis=-1)

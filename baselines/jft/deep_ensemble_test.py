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

"""Tests for deep_ensemble."""

from absl.testing import parameterized
import flax.linen as nn
import jax
import jax.numpy as jnp
import tensorflow as tf
import deep_ensemble  # local file import from baselines.jft


class DummyModel(nn.Module):
  output_dim: int

  @nn.compact
  def __call__(self, x, train=None):
    y = nn.Dense(self.output_dim)(x)
    # For simplicity, we set the pre-logits to be equal to the logits.
    return y, {'pre_logits': y}


class DeepEnsembleTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.parameters((1, 'softmax_xent'), (3, 'softmax_xent'),
                            (5, 'softmax_xent'), (1, 'sigmoid_xent'),
                            (3, 'sigmoid_xent'), (5, 'sigmoid_xent'))
  def test_ensemble_pred_fn(self, ensemble_size, loss):
    num_classes = 3
    image_dim = 5
    batch_size = 8
    model = DummyModel(output_dim=num_classes)

    images = jax.random.normal(
        jax.random.PRNGKey(0), shape=(batch_size, image_dim))
    kernels = jax.random.normal(
        jax.random.PRNGKey(1), shape=(ensemble_size, image_dim, num_classes))
    biases = jax.random.normal(
        jax.random.PRNGKey(2), shape=(ensemble_size, num_classes))

    params = {}
    for e in range(ensemble_size):
      params[e] = {'Dense_0': {'kernel': kernels[e], 'bias': biases[e]}}
    raw_logits = jnp.asarray(
        [jnp.dot(images, kernels[e]) + biases[e] for e in range(ensemble_size)])

    pred_fn = deep_ensemble.ensemble_prediction_fn
    actual_logits, actual_pre_logits = pred_fn(model.apply, params, images,
                                               loss)

    if loss == 'softmax_xent':
      link_fn = jax.nn.softmax
    else:
      link_fn = jax.nn.sigmoid

    expected_probs = jnp.mean(link_fn(raw_logits), axis=0)
    self.assertAllClose(link_fn(actual_logits), expected_probs)
    self.assertAllClose(link_fn(actual_pre_logits), expected_probs)

if __name__ == '__main__':
  tf.test.main()

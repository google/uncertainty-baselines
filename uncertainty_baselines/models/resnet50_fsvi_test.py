# coding=utf-8
# Copyright 2021 The Uncertainty Baselines Authors.
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

import haiku as hk
import jax
import tensorflow as tf
import numpy as np

import uncertainty_baselines as ub
from uncertainty_baselines.models.resnet50_fsvi import zero_padding_2d


class ResNet50FSVITest(tf.test.TestCase):

  def testCreateModel(self):
    batch_size = 31

    def forward(inputs, rng_key, stochastic, is_training):
      model = ub.models.ResNet50FSVI(
          output_dim=10,
          stochastic_parameters=True,
          dropout=False,
          dropout_rate=0.,
      )
      return model(inputs, rng_key, stochastic, is_training)

    init_fn, apply_fn = hk.transform_with_state(forward)
    key = jax.random.PRNGKey(42)
    x = jax.random.normal(key, shape=(batch_size, 32, 32, 3))
    params, state = init_fn(key, x, key, stochastic=True, is_training=True)
    output, _ = apply_fn(
        params, state, key, x, key, stochastic=True, is_training=True)
    self.assertEqual(output.shape, (31, 10))

  def test_zero_padding_2D(self):
    key = jax.random.PRNGKey(42)
    x = jax.random.normal(key, shape=(2, 32, 32, 3))
    padding = 3
    actual = zero_padding_2d(x, padding=padding)
    # TODO(nbdand): make this work, currently have the following error
    # tensorflow.python.framework.errors_impl.InternalError: Cannot dlopen all
    # CUDA libraries.
    expected = tf.keras.layers.ZeroPadding2D(padding=3)(x)
    assert actual.shape == expected.shape
    assert np.abs(expected.numpy() - actual).max() < 1e-10


if __name__ == '__main__':
  tf.test.main()

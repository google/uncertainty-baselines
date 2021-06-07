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

"""Tests for jax_utils."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import tensorflow as tf

from uncertainty_baselines.models import jax_utils


class BatchensembleTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.parameters(
      (4, [3, 20], 0.0, 10),     # 2-Dimensional input
      (2, [5, 7, 3], -0.5, 10),  # 3-Dimensional input
      (2, [5, 7, 2, 3], -0.5, 10),  # 4-Dimensional input
  )
  def test_params_shapes(self, ens_size, inputs_shape, random_sign_init,
                         output_dim):
    alpha_init = jax_utils.make_sign_initializer(random_sign_init)
    gamma_init = jax_utils.make_sign_initializer(random_sign_init)
    inputs = jax.random.normal(jax.random.PRNGKey(0), inputs_shape,
                               dtype=jnp.float32)
    tiled_inputs = jnp.tile(inputs, [ens_size] + [1] * (inputs.ndim - 1))

    layer = jax_utils.DenseBatchEnsemble(
        features=output_dim,
        ens_size=ens_size,
        alpha_init=alpha_init,
        gamma_init=gamma_init)

    tiled_outputs, params = layer.init_with_output(jax.random.PRNGKey(0),
                                                   tiled_inputs)
    params_shape = jax.tree_map(lambda x: x.shape, params)
    expected_kernel_shape = (inputs_shape[-1], output_dim)
    expected_alpha_shape = (ens_size, inputs_shape[-1])
    expected_gamma_shape = (ens_size, output_dim)
    self.assertEqual(expected_kernel_shape, params_shape["params"]["kernel"])
    self.assertEqual(expected_alpha_shape,
                     params_shape["params"]["fast_weight_alpha"])
    self.assertEqual(expected_gamma_shape,
                     params_shape["params"]["fast_weight_gamma"])

    loop_outputs = []
    for i in range(ens_size):
      alpha_shape = (1,) * (inputs.ndim - 1) + (-1,)
      alpha = params["params"]["fast_weight_alpha"][i].reshape(alpha_shape)
      perturb_inputs = inputs * alpha
      outputs = jnp.dot(perturb_inputs, params["params"]["kernel"])
      loop_outputs.append(outputs * params["params"]["fast_weight_gamma"][i] +
                          params["params"]["bias"][i])
    loop_outputs_list = jnp.concatenate(loop_outputs, axis=0)
    expected_outputs_shape = tiled_inputs.shape[:-1] + (output_dim,)
    self.assertEqual(tiled_outputs.shape, expected_outputs_shape)
    self.assertAllClose(tiled_outputs, loop_outputs_list)
    pass


if __name__ == "__main__":
  absltest.main()

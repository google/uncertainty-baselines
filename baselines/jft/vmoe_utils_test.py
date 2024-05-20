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

"""Tests of vmoe_utils."""
from unittest import mock
from absl.testing import absltest
from absl.testing import parameterized
import flax
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
import vmoe_utils  # local file import from baselines.jft

ConfigDict = ml_collections.ConfigDict


class VmoeUtilsTest(parameterized.TestCase):

  def test_variables_partition_spec(self):
    params = {'my_model': {'Moe': {'Mlp': 0, 'mlp': 1}, 'self-attention': 3}}
    partition_spec = vmoe_utils.get_variables_partition_spec(params)
    partition_spec = flax.core.unfreeze(partition_spec)
    expected_partition_spec = {
        'my_model': {
            'Moe': {
                'Mlp': jax.sharding.PartitionSpec(('expert',)),
                'mlp': jax.sharding.PartitionSpec(),
            },
            'self-attention': jax.sharding.PartitionSpec(),
        }
    }
    jax.tree.map(np.testing.assert_equal, expected_partition_spec,
                      partition_spec)

  def test_deep_ensemble_reshape_outputs(self):
    logits_and_prelogits = [
        (np.ones((1,)), 2*np.ones((2,))), (np.zeros((1,)), -np.ones((2,)))
    ]

    reshape = vmoe_utils.deep_ensemble_reshape_outputs_fn
    logits, prelogits = reshape(logits_and_prelogits)
    np.testing.assert_array_equal(logits, jnp.asarray([[1], [0]]))
    np.testing.assert_array_equal(prelogits, jnp.asarray([[2, 2], [-1, -1]]))

  def test_efficient_ensemble_reshape_outputs(self):
    ensemble_size = 2
    x = np.arange(6).reshape((6, 1))
    logits_and_prelogits = [(x, -x)]
    reshape = vmoe_utils.efficient_ensemble_reshape_outputs_fn
    logits, prelogits = reshape(logits_and_prelogits, ensemble_size)
    # Both logits and prelogits have shapes (ensemble size, batch size, dim).
    # In that case, (2, 6//2, 1).
    reshaped_outputs = jnp.asarray([[[0], [2], [4]], [[1], [3], [5]]])
    np.testing.assert_array_equal(logits, reshaped_outputs)
    np.testing.assert_array_equal(prelogits, -reshaped_outputs)


if __name__ == '__main__':
  absltest.main()

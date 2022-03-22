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

"""Tests of vmoe_utils."""

from absl.testing import absltest
from absl.testing import parameterized
import flax
import jax
from jax.experimental import pjit
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
                'Mlp': pjit.PartitionSpec(('expert',)),
                'mlp': pjit.PartitionSpec(),
            },
            'self-attention': pjit.PartitionSpec(),
        }
    }
    jax.tree_multimap(np.testing.assert_equal, expected_partition_spec,
                      partition_spec)


if __name__ == '__main__':
  absltest.main()

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

"""Set of functions related to the manipulations of V-MoE's."""
from typing import Any, Mapping

import flax
import jax
from jax.experimental import pjit
import ml_collections


def get_pjit_eval_fn_with_mesh(eval_fn, mesh, in_axis_resources, num_outputs):
  """Makes eval_fn a pjit function distributed according to the mesh."""
  out_axis_resources = tuple([pjit.PartitionSpec() for _ in range(num_outputs)])

  pjit_eval_fn = pjit.pjit(
      fun=eval_fn,
      in_axis_resources=in_axis_resources,
      out_axis_resources=out_axis_resources)

  def eval_fn_with_mesh(params, images, labels, mask):
    with mesh:
      outputs = pjit_eval_fn(params, images, labels, mask)
    return outputs

  return eval_fn_with_mesh


def get_variables_partition_spec(oss_params):
  """Specifies how the params are partitioned for pjit."""
  is_frozen_dict = isinstance(oss_params, flax.core.FrozenDict)
  if is_frozen_dict:
    oss_params = oss_params.unfreeze()

  variables_partition_spec = {}
  for name in flax.traverse_util.flatten_dict(oss_params):
    if 'Moe/Mlp' in '/'.join(name):
      variables_partition_spec[name] = pjit.PartitionSpec(('expert',))
    else:
      variables_partition_spec[name] = pjit.PartitionSpec()
  variables_partition_spec = flax.core.freeze(
      flax.traverse_util.unflatten_dict(variables_partition_spec))
  return variables_partition_spec



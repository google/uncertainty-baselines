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

"""EncoderDecoder model for Heteroscedastic Transformer."""

from typing import Mapping, Optional
from flax.core import scope as flax_scope
import jax.numpy as jnp
from t5x import utils
import t5x.models as t5x_models
from models import models as ub_models  # local file import from baselines.t5

Array = t5x_models.Array


class EncoderDecoderHeteroscedasticClassifierModel(
    ub_models.EncoderDecoderClassifierModel):
  """A wrapper of EncoderDecoderClassifierModel to support Heteroscedastic head."""

  def get_initial_variables(
      self,
      rng: jnp.ndarray,
      input_shapes: Mapping[str, Array],
      input_types: Optional[Mapping[str, jnp.dtype]] = None
  ) -> flax_scope.FrozenVariableDict:
    initial_variables = super().get_initial_variables(
        rng=rng, input_shapes=input_shapes, input_types=input_types)
    if 'params_axes' not in initial_variables:
      return initial_variables
    # For Flax parameters defined using `self.param`, we need to declare
    # their axes' names here. See
    # go/t5x/usage/partitioning.md#overriding-axis-names-from-external-codebase
    # for more information.
    # Because we use the default shard `1` for activations and parameters,
    # the names are used here are not important. But they need to satisfy
    #   + each dimension needs to have corresponding axis name,
    #   + per parameter, the axis names need to be different.
    return utils.override_params_axes_names(
        initial_variables,
        params_axes_names_override=[
            ('decoder/heteroscedastic_head/.*bias', ('vocab',)),
            ('decoder/heteroscedastic_head/.*kernel', ('mlp', 'vocab')),
        ])


class EncoderDecoderHeteroscedasticBeamScoreModel(
    ub_models.EncoderDecoderBeamScoreModel,
    EncoderDecoderHeteroscedasticClassifierModel):
  """A wrapper of EncoderDecoderBeamScoreModel to support Heteroscedastic head."""

  def get_initial_variables(
      self,
      rng: jnp.ndarray,
      input_shapes: Mapping[str, Array],
      input_types: Optional[Mapping[str, jnp.dtype]] = None
  ) -> flax_scope.FrozenVariableDict:
    return EncoderDecoderHeteroscedasticClassifierModel.get_initial_variables(
        self, rng=rng, input_shapes=input_shapes, input_types=input_types)

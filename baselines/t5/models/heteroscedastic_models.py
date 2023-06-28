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

"""EncoderDecoder model for Heteroscedastic Transformer."""

from typing import List, Mapping, Optional, Tuple
from flax.core import scope as flax_scope
import jax.numpy as jnp
from t5x import utils
import t5x.models as t5x_models
from models import models as ub_models  # local file import from baselines.t5

Array = t5x_models.Array
AssignmentMap = List[Tuple[str, Optional[str]]]


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
            # Additional parameter for the automatic tuning of the temperature.
            ('decoder/heteroscedastic_head/pre_sigmoid_temperature', ('mlp',))
        ])


class EncoderDecoderHeteroscedasticBeamScoreModel(  # pytype: disable=signature-mismatch  # overriding-parameter-count-checks
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


def get_assignment_map(
    use_pretrained_head: bool, num_factors: int, location_parameter_spec: str
) -> AssignmentMap:
  """Defines the assignment map that controls the checkpoint restoring logic.

  Args:
    use_pretrained_head: Whether to reuse the head of the pretrained model for
      the location parameter (=mean) of the heteroscedastic layer. Note that
      only the kernel is transferred since the T5 head comes with no bias term.
    num_factors: Number of factors used in the low-rank parametrisation of the
      covariance in the heteroscedastic layer. When it is <= 0, only a diagonal
      parametrisation is used.
    location_parameter_spec: String that specifies how the location parameters
      are encoded in the heteroscedastic layer. They can be represented as a
      nn.Dense ('layer') or as explicit (kernel, bias) parameters defined via
      self.param ('param'). For context, location_parameter_spec makes it
      possible to reuse embeddings in place of the weight matrix of the nn.Dense
      layer producing the location parameters.

  Returns:
    The `assignment_map` to be passed to utils.RestoreCheckpointConfig.
  """
  assignment_map = []

  if not use_pretrained_head:
    return assignment_map

  # No map for the bias term since not present in the T5 head: See
  #  https://github.com/google-research/t5x/blob/main/t5x/examples/scalable_t5/network.py#L356
  #  https://github.com/google-research/t5x/blob/main/t5x/examples/scalable_t5/layers.py#L405
  assignment_map_head = []

  if location_parameter_spec == 'layer':
    assignment_map_head = [(
        r'(.*)decoder/heteroscedastic_head/loc_layer/kernel(.*)',
        r'\1decoder/logits_dense/kernel\2',
    )]
  elif location_parameter_spec == 'param':
    assignment_map_head = [(
        r'(.*)decoder/heteroscedastic_head/loc_layer_kernel(.*)',
        r'\1decoder/logits_dense/kernel\2',
    )]
  # Else, we do not have explicit kernel parameter and we instead rely on some
  # shared parameters such as the embeddings of the model.

  assignment_map = assignment_map_head + [
      (r'(.*)decoder/heteroscedastic_head/diag_layer(.*)', None),
  ]
  if num_factors > 0:
    assignment_map += [
        # This covers `scale_layer` as well as `scale_layer_homoscedastic` and
        # `scale_layer_heteroscedastic` in the parameter-efficient case.
        (r'(.*)decoder/heteroscedastic_head/scale_layer(.*)', None),
    ]
  return assignment_map

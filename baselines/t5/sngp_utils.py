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

"""Some helpers for using sngp binary."""

from typing import Mapping, Optional

import flax
from flax.core import scope as flax_scope
import jax
import jax.numpy as jnp
from t5x import adafactor
from t5x import losses
from t5x import models
from t5x import utils
from models import models as ub_models  # local file import from baselines.t5

unfreeze = flax.core.unfreeze
Array = models.Array


# TODO(phandu): Move this class to `ub.baselines.t5.models`.
class EncoderDecoderGPModel(models.EncoderDecoderModel):
  """A wrapper of t5x.models.EncoderDecoderModel to support mutable updates."""

  def loss_fn(
      self,
      params,
      batch,
      dropout_rng,
  ):
    # For evaluation, we just simply use t5x implementation.
    if dropout_rng is None:
      return super().loss_fn(params, batch, dropout_rng)

    logits, state = self._compute_logits(
        params, batch, dropout_rng, mutable=['intermediates'])
    targets = batch['decoder_target_tokens']
    weights = batch.get('decoder_loss_weights', None)
    loss, total_z_loss, _ = losses.compute_weighted_cross_entropy(
        logits,
        targets=targets,
        weights=weights,
        label_smoothing=self._label_smoothing,
        z_loss=self._z_loss,
        loss_normalizing_factor=self._loss_normalizing_factor)
    metrics = models.compute_base_metrics(logits, targets, weights, loss,
                                          total_z_loss)

    # Get the head states and their updated values.
    # See ub.models.t5_gp.GaussianProcessDecoder._apply_gp_layer for details.
    head_state_flat = [
        v for path, v in sorted(
            flax.traverse_util.flatten_dict(flax.core.unfreeze(params)).items())
        if 'gp_head_state' in path
    ]
    state_flat = flax.traverse_util.flatten_dict(
        flax.core.unfreeze(state['intermediates']))
    head_state_new = [
        v[0] for path, v in state_flat.items() if 'gp_head_state_new' in path
    ][0]
    head_state_new_flat = [
        v for _, v in sorted(
            flax.traverse_util.flatten_dict(flax.core.unfreeze(
                head_state_new)).items())
    ]

    # We will add to the loss a zero factor with non-zero gradients.
    # The factor for `x` will be `x * x_new` i.e. the gradient of the loss
    # with respect to `x` will be `x_new`. In the optimizer, we will update
    # `x` to this gradient value `x_new`.
    head_state_new_flat = jax.lax.stop_gradient(head_state_new_flat)
    factor = sum((x * x_new).sum()
                 for x, x_new in zip(head_state_flat, head_state_new_flat))
    loss = loss + factor - jax.lax.stop_gradient(factor)
    return loss, metrics

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
            ('decoder/gp_head/output_layer/bias', ('vocab',)),
            ('decoder/gp_head/output_layer/kernel', ('mlp', 'vocab')),
            ('decoder/gp_head_state/random_features/.*bias', ('mlp',)),
            ('decoder/gp_head_state/random_features/.*kernel', ('embed',
                                                                'mlp')),
            ('decoder/gp_head_state/.*precision_matrix', ('embed', 'vocab')),
            ('decoder/gp_head_state/step', ()),
        ])


class AdafactorGP(adafactor.Adafactor):
  """A wrapper of t5x.adafactor.Adafactor to support mutable updates."""

  def apply_param_gradient(self, step, hyper_params, param, state, grad, path):
    if 'gp_head_state' in path:
      # For head_state parameters, we will use grad as the new value.
      return grad.astype(param.dtype), state

    return super().apply_param_gradient(step, hyper_params, param, state, grad,
                                        path)


class EncoderDecoderGPClassifierModel(ub_models.EncoderDecoderClassifierModel,
                                      EncoderDecoderGPModel):
  """A wrapper of EncoderDecoderClassifierModel to support mutable updates."""

  def loss_fn(self, *args, **kwargs):
    return EncoderDecoderGPModel.loss_fn(self, *args, **kwargs)

  def get_initial_variables(
      self,
      rng: jnp.ndarray,
      input_shapes: Mapping[str, Array],
      input_types: Optional[Mapping[str, jnp.dtype]] = None
  ) -> flax_scope.FrozenVariableDict:
    initial_variables = EncoderDecoderGPModel.get_initial_variables(
        self, rng=rng, input_shapes=input_shapes, input_types=input_types)
    return initial_variables


class EncoderDecoderGPBeamScoreModel(ub_models.EncoderDecoderBeamScoreModel,
                                     EncoderDecoderGPModel):
  """A wrapper of EncoderDecoderBeamScoreModel to support mutable updates."""

  def loss_fn(self, *args, **kwargs):
    return EncoderDecoderGPModel.loss_fn(self, *args, **kwargs)

  def get_initial_variables(
      self,
      rng: jnp.ndarray,
      input_shapes: Mapping[str, Array],
      input_types: Optional[Mapping[str, jnp.dtype]] = None
  ) -> flax_scope.FrozenVariableDict:
    initial_variables = EncoderDecoderGPModel.get_initial_variables(
        self, rng=rng, input_shapes=input_shapes, input_types=input_types)
    return initial_variables

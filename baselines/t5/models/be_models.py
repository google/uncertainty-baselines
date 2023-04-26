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

"""EncoderDecoder model for BatchEnsemble Transformer."""

from typing import Any, Mapping, Optional, Union, Tuple
from flax.core import scope as flax_scope
import jax
import jax.numpy as jnp
import t5x.models as t5x_models
from models import gp_models  # local file import from baselines.t5
from models import models as ub_models  # local file import from baselines.t5

Array = t5x_models.Array


class EncoderDecoderBEClassifierModel(ub_models.EncoderDecoderClassifierModel):
  """A wrapper of EncoderDecoderClassifierModel to support BatchEnsemble loss."""

  def loss_fn(
      self,
      params,
      batch,
      dropout_rng,
  ):
    target_tokens = batch['decoder_target_tokens']
    # Tile the labels for batch ensembles.
    ens_size = self.module.ens_size
    batch['decoder_target_tokens'] = jnp.tile(target_tokens, [ens_size] + [1] *
                                              (target_tokens.ndim - 1))
    loss_weights = batch['decoder_loss_weights']
    if loss_weights is not None:
      batch['decoder_loss_weights'] = jnp.tile(loss_weights, [ens_size] + [1] *
                                               (loss_weights.ndim - 1))
    return super().loss_fn(params, batch, dropout_rng)

  def _compute_argmax_score(
      self,
      params: t5x_models.PyTreeDef,
      batch: Mapping[str, jnp.ndarray],
      return_intermediates: bool = False,
      dropout_rng: Optional[jnp.ndarray] = None,
      ensemble_probs: bool = True,
  ) -> Union[jnp.ndarray, Tuple[jnp.ndarray, Mapping[str, Any]]]:
    """Compute class logits on a batch."""
    ens_size = self.module.ens_size
    sequence_scores = super()._compute_argmax_score(params, batch,
                                                    return_intermediates,
                                                    dropout_rng, ensemble_probs)
    if return_intermediates:
      sequence_scores, intermediates = sequence_scores
    if ens_size > 1:
      sequence_scores = jnp.reshape(sequence_scores,
                                    (ens_size, -1) + sequence_scores.shape[1:])  # pytype: disable=attribute-error  # jax-ndarray
      if ensemble_probs:
        # Computes log(mean(exp(logits))) along the first dimension.
        sequence_scores = (
            jax.nn.logsumexp(sequence_scores, axis=0) - jnp.log(ens_size))
      else:
        sequence_scores = jnp.mean(sequence_scores, axis=0)
    if return_intermediates:
      return sequence_scores, intermediates
    return sequence_scores

  def _compute_logits_from_slice(self,
                                 decoding_state,
                                 params,
                                 encoded_inputs,
                                 raw_inputs,
                                 max_decode_length,
                                 rngs=None,
                                 ensemble_probs=True):
    """Token slice to logits from decoder model."""
    ens_size = self.module.ens_size
    k = jax.tree_util.tree_flatten(params)[0][0].shape[0]
    flat_logits, new_cache = super()._compute_logits_from_slice(
        decoding_state, params, encoded_inputs, raw_inputs, max_decode_length,
        rngs, ensemble_probs)
    if ens_size > 1:
      flat_logits = jnp.reshape(flat_logits,
                                (k, ens_size, -1) + flat_logits.shape[1:])
      if ensemble_probs:
        flat_logits = (
            jax.nn.logsumexp(flat_logits, axis=1) - jnp.log(ens_size))
      else:
        flat_logits = jnp.mean(flat_logits, axis=1)
      flat_logits = jnp.reshape(flat_logits, (-1,) + flat_logits.shape[2:])
    return flat_logits, new_cache


class EncoderDecoderBEGpClassifierModel(EncoderDecoderBEClassifierModel,
                                        gp_models.EncoderDecoderGPModel):
  """A wrapper of EncoderDecoderClassifierModel for BatchEnsemble and GP."""

  def loss_fn(
      self,
      params,
      batch,
      dropout_rng,
  ):
    target_tokens = batch['decoder_target_tokens']
    # Tile the labels for batch ensembles.
    ens_size = self.module.ens_size
    batch['decoder_target_tokens'] = jnp.tile(target_tokens, [ens_size] + [1] *
                                              (target_tokens.ndim - 1))
    loss_weights = batch['decoder_loss_weights']
    if loss_weights is not None:
      batch['decoder_loss_weights'] = jnp.tile(loss_weights, [ens_size] + [1] *
                                               (loss_weights.ndim - 1))
    return gp_models.EncoderDecoderGPModel.loss_fn(self, params, batch,
                                                   dropout_rng)

  def get_initial_variables(
      self,
      rng: jnp.ndarray,
      input_shapes: Mapping[str, Array],
      input_types: Optional[Mapping[str, jnp.dtype]] = None
  ) -> flax_scope.FrozenVariableDict:
    initial_variables = gp_models.EncoderDecoderGPModel.get_initial_variables(
        self, rng=rng, input_shapes=input_shapes, input_types=input_types)
    return initial_variables


class EncoderDecoderBEBeamScoreModel(ub_models.EncoderDecoderBeamScoreModel,  # pytype: disable=signature-mismatch  # overriding-parameter-count-checks
                                     EncoderDecoderBEClassifierModel):
  """A wrapper of EncoderDecoderClassifierModel to support BatchEnsemble loss."""

  def loss_fn(self, params, batch, dropout_rng):
    return EncoderDecoderBEClassifierModel.loss_fn(self, params, batch,
                                                   dropout_rng)

  def _compute_argmax_score(
      self,
      params: t5x_models.PyTreeDef,
      batch: Mapping[str, jnp.ndarray],
      return_intermediates: bool = False,
      dropout_rng: Optional[jnp.ndarray] = None,
      ensemble_probs: bool = True,
  ) -> Union[jnp.ndarray, Tuple[jnp.ndarray, Mapping[str, Any]]]:
    """Compute class logits on a batch."""
    return EncoderDecoderBEClassifierModel._compute_argmax_score(
        self,
        params,
        batch,
        return_intermediates=return_intermediates,
        dropout_rng=dropout_rng,
        ensemble_probs=ensemble_probs)

  def _compute_logits_from_slice(self,
                                 decoding_state,
                                 params,
                                 encoded_inputs,
                                 raw_inputs,
                                 max_decode_length,
                                 rngs=None,
                                 ensemble_probs=True):
    """Token slice to logits from decoder model."""
    return EncoderDecoderBEClassifierModel._compute_logits_from_slice(
        self,
        decoding_state,
        params,
        encoded_inputs,
        raw_inputs,
        max_decode_length,
        rngs=rngs,
        ensemble_probs=ensemble_probs)


class EncoderDecoderBEGpBeamScoreModel(EncoderDecoderBEBeamScoreModel,  # pytype: disable=signature-mismatch  # overriding-parameter-count-checks
                                       EncoderDecoderBEGpClassifierModel):
  """A wrapper of EncoderDecoderBeamScoreModel for BatchEnsemble and GP."""

  def loss_fn(self, params, batch, dropout_rng):
    return EncoderDecoderBEGpClassifierModel.loss_fn(self, params, batch,
                                                     dropout_rng)

  def get_initial_variables(
      self,
      rng: jnp.ndarray,
      input_shapes: Mapping[str, Array],
      input_types: Optional[Mapping[str, jnp.dtype]] = None
  ) -> flax_scope.FrozenVariableDict:
    initial_variables = EncoderDecoderBEGpClassifierModel.get_initial_variables(
        self, rng=rng, input_shapes=input_shapes, input_types=input_types)
    return initial_variables

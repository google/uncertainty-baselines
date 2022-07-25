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

"""Customized T5X Models."""
import functools
from typing import Any, Mapping, MutableMapping, Optional, Tuple, Union, Sequence

from flax import core as flax_core
from flax import traverse_util
from flax.training import common_utils
import jax
import jax.numpy as jnp
import numpy as np
from t5x import losses
import t5x.models as t5x_models
import decoding  # local file import from baselines.t5

INT_CLASSES = (int, jnp.integer)


def _cache_concatenate(
    caches: Mapping[str, jnp.ndarray]) -> Mapping[str, jnp.ndarray]:
  """Using t5x.decoding.cache_map to concatenate caches together.

  This function concatenates the values of the caches except for the scalar
  field `cache_index`, which we still want to keep it scalar.

  Args:
    caches: A batch of caches to concatenate. Each value of a cache has shape
      [B, ...] except for the scalar `cache_index`.

  Returns:
    The concatenated version of `caches`. Each value has shape [K * B, ...]
    except for the scalar `cache_index`. Here `K` is the length of the input
    `caches`.
  """
  caches = jax.tree_util.tree_map(lambda x: x[0] if x.ndim == 1 else x, caches)
  caches = decoding.cache_map(lambda x: x.reshape((-1,) + x.shape[2:]), caches)
  return caches


def _partial_map(f, args):
  """Like jax.lax.map but allow the first argument to have batch size 1.

  This is helpful for MC Dropout where we want to perform lax.map over a
  function of network's parameters and some other arguments but do not want
  to broadcast the parameters.

  Args:
    f: A function to map over.
    args: A batch of arguments of f.

  Returns:
    The concatenated output of f over a batch of args.
  """
  k = jax.tree_util.tree_flatten(args[0])[0][0].shape[0]
  if k > 1:
    return jax.lax.map(f, args)

  p = jax.tree_util.tree_map(lambda x: x[0], args[0])
  vals = args[1:]

  def _f_partial(vals):
    return f((p,) + vals)

  # Treat special case where the remaining arguments do not contain JAX
  # arrays. Some examples for such scenario:
  #   + when f takes only 1 argument p.
  #   + when f takes two arguments (p, x) but x is None.
  flatten_vals = jax.tree_util.tree_flatten(vals)[0]
  if not flatten_vals:
    y = _f_partial(vals)
    # Add a singleton dimension to y to match lax.map behavior.
    return jax.tree_util.tree_map(lambda x: jnp.expand_dims(x, 0), y)
  else:
    return jax.lax.map(_f_partial, vals)


def _compute_token_entropy(logits: jnp.ndarray) -> jnp.ndarray:
  """Computes predictive entropy for each output token."""
  model_class_log_probs = jax.nn.log_softmax(logits, axis=-1)
  return jnp.sum(
      jnp.exp(model_class_log_probs) * model_class_log_probs, axis=-1)


class EncoderDecoderClassifierModel(t5x_models.EncoderDecoderModel):
  """An EncoderDecoderModel that outputs class logits for classification.

  This model is equivalent to t5x's EncoderDecoderModel, but its score_batch()
  function is modified to output class logits with shape
  (batch_size, output_len, num_classes) rather than the log likelihood scores of
  the top sequences (shape: batch_size, output_len, num_decode).

  Attributes:
    label_tokens: Ordered sequence of tokens representing the predicted classes.
    label_token_ids: Ordered sequence of integers representing the id of the
      class tokens in the model vocabulary.
  """

  def __init__(self,
               label_tokens: Optional[Sequence[Union[str, int]]] = None,
               temperature: float = 1.,
               **kwargs):
    """Model initializer.

    Args:
      label_tokens: Ordered sequence of tokens for the output classes. If None
        then the whole vocabulary is used like labels.
      temperature: Temperature parameter to be used to scale the predicted
        logits. So that the output probability is `softmax( logits /
        temperature)`. This is applied only to the inference time (i.e., to
        `_compute_argmax_score` and `predict_batch_with_aux`) and does not
        impact training.
      **kwargs: Keyword arguements to be passed to parent class (
        t5x_models.EncoderDecoderModel).
    """
    # TODO(jereliu): Make the input arguments explicit after t5x library
    # becomes stable.
    super().__init__(**kwargs)
    self.temperature = temperature
    self.label_tokens = label_tokens
    self.label_token_ids = self._get_label_token_ids()

  def _get_label_token_ids(self):
    """Makes a list of vocab ids for label tokens."""

    if self.label_tokens is None:
      return jnp.arange(0, self.output_vocabulary.vocab_size)

    label_token_ids = []

    for token in self.label_tokens:
      token_id = self._output_vocabulary.encode(token)

      # Convert token_id to integer.
      if not isinstance(token_id, INT_CLASSES):
        if np.size(token_id) != 1:
          raise ValueError('Class names must correspond to single tokens. '
                           f'Got class name "{token}" with token {token_id}.')
        token_id = int(np.asarray(token_id).item())

      label_token_ids.append(token_id)

    return jnp.array(label_token_ids)

  def _compute_argmax_score(
      self,
      params: t5x_models.PyTreeDef,
      batch: Mapping[str, jnp.ndarray],
      return_intermediates: bool = False,
      dropout_rng: Optional[jnp.ndarray] = None,
      ensemble_probs: bool = True,
  ) -> Union[jnp.ndarray, Tuple[jnp.ndarray, Mapping[str, Any]]]:
    """Compute class logits on a batch.

    If `params` is a list, we will perform ensemble over the output
    scores for each set of parameters but only return intermediate
    variables of the first set of parameters.

    Args:
      params: A set of parameters or a list of sets of parameters.
      batch: The input data.
      return_intermediates: Whether to return the intermediate variables.
      dropout_rng: optional dropout rng. If a batch of rngs is provided, we will
        perform MC Dropout evaluation.
      ensemble_probs: Whether to perform ensemble in probs or logits spaces.

    Returns:
      The class logits or a tuple of class logits and intermediate variables.
    """
    weights = batch['decoder_loss_weights']
    target_tokens = batch['decoder_target_tokens']
    # Setup a list of params (list of shape [K]) and dropout_rngs
    # (jnp.ndarray of shape [K, 2] or None) for performning ensemble.
    run_ensemble = isinstance(params, list)
    run_mc_dropout = (dropout_rng is not None) and (dropout_rng.ndim == 2)
    if run_ensemble:
      if dropout_rng is not None:
        raise ValueError('Dropout is not allowed for batch ensemble.')
      params_list = params
      dropout_rngs = None
    elif run_mc_dropout:
      params_list = [params]
      dropout_rngs = dropout_rng
    else:
      params_list = [params]
      if dropout_rng is None:
        dropout_rngs = None
      else:
        # Add a singleton dimension to dropout so its shape is [1, 2].
        dropout_rngs = jnp.expand_dims(dropout_rng, 0)

    batch_params = jax.tree_util.tree_map(lambda *vals: jnp.stack(vals),
                                               *params_list)

    def get_sequence_scores_and_intermediates(vals):
      params, dropout_rng = vals
      if return_intermediates:
        logits, modified_variables = self._compute_logits(
            params=params,
            batch=batch,
            mutable=['intermediates'],
            dropout_rng=dropout_rng)

        # Inside self.module, we called nn.Module.sow to track various
        # intermediate values. We extract them here.
        intermediates = flax_core.unfreeze(
            modified_variables.get('intermediates', {}))

        # Track per-token labels and loss weights as well. These are not
        # intermediate values of logit computation, so we manually add them
        # here.
        intermediates.setdefault('decoder', {})
        intermediates['decoder']['target_tokens'] = (target_tokens,)
        intermediates['decoder']['loss_weights'] = (weights,)
        # Note that the values are singleton tuples. This is because values
        # inside `intermediates` should be tuples tracking all instantiations
        # of a value. These values each have just one instantiation, hence
        # singletons.
      else:
        # Type of logits: jnp.ndarray.
        logits = self._compute_logits(params, batch, dropout_rng=dropout_rng)
        intermediates = None

      # Returns class logits, shape (batch_size, output_len, num_class).
      sequence_scores = logits[:, :, self.label_token_ids]
      return sequence_scores, intermediates

    batch_sequence_scores, intermediates = _partial_map(
        get_sequence_scores_and_intermediates, (batch_params, dropout_rngs))

    if batch_sequence_scores.shape[0] == 1:
      sequence_scores = batch_sequence_scores[0]
    else:
      # Normalize the logits, assuming that last dimension is num_classes.
      batch_log_probs = jax.nn.log_softmax(batch_sequence_scores, axis=-1)
      if ensemble_probs:
        # Computes log(mean(exp(logits))) along the first dimension.
        sequence_scores = (
            jax.nn.logsumexp(batch_log_probs, axis=0) -
            jnp.log(batch_log_probs.shape[0]))
      else:
        sequence_scores = jnp.mean(batch_log_probs, axis=0)

    # Apply temperature scaling.
    sequence_scores = sequence_scores / self.temperature

    if return_intermediates:
      # We only return the intermediates of the first set of `params_list`.
      intermediates = jax.tree_util.tree_map(lambda x: x[0], intermediates)
      return sequence_scores, intermediates

    return sequence_scores

  def score_batch(
      self,
      params: t5x_models.PyTreeDef,
      batch: Mapping[str, jnp.ndarray],
      return_intermediates: bool = False,
      return_beam_predictions: bool = False,
      dropout_seed: Optional[int] = None,
      num_mcdropout_samples: Optional[int] = None,
      ensemble_probs: bool = True,
      intermediates_to_track: Optional[Sequence[str]] = None,
      num_decodes: int = 1,
  ) -> Union[jnp.ndarray, Tuple[jnp.ndarray, Mapping[str, Any]]]:
    """Compute logit scores on a batch.

    Args:
      params: model parameters.
      batch: a batch of inputs.
      return_intermediates: Whether to return the intermediate variables.
      return_beam_predictions: Whether to return addition beam predictions along
        with the logit scores.
      dropout_seed: The seed to generate random keys for dropout samples.
      num_mcdropout_samples: The number of dropout samples for evaluation. If
        provided, we will perform MC Dropout evaluation.
      ensemble_probs: Whether to perform ensemble in probs or logits spaces.
      intermediates_to_track: A list/tuple of intermediate fields to return.
        We will flatten `intermediates` and only return values for flatten keys
        that are specified in this list/tuple.
      num_decodes: the number of beams to use in beam search.

    Returns:
      The logit scores on a batch and optional intermediate variables.
    """
    dropout_rng = None
    if dropout_seed is not None:
      dropout_rng = jax.random.PRNGKey(dropout_seed)
    if num_mcdropout_samples is not None:
      dropout_rng = jax.random.split(dropout_rng, num_mcdropout_samples)
    # Computes argmax predictive logits over the whole class,
    # shape (batch_size, output_len, num_class).
    sequence_scores = self._compute_argmax_score(params, batch,
                                                 return_intermediates,
                                                 dropout_rng, ensemble_probs)

    if return_intermediates:
      sequence_scores, intermediates = sequence_scores

    if return_beam_predictions:
      # Also returns beam prediction and corresponding token-level probability
      # from top-k beam search, shape (batch_size, beam_size, output_len).
      beam_predictions, beam_scores_dict = self.predict_batch_with_aux(
          params,
          batch,
          return_all_decodes=True,
          dropout_rng=dropout_rng,
          ensemble_probs=ensemble_probs,
          num_decodes=num_decodes)
      beam_scores = beam_scores_dict['scores']
      sequence_scores = (sequence_scores, beam_predictions, beam_scores)

    # Prepares output.
    if return_intermediates:
      if intermediates_to_track:
        if 'encoded_inputs' in intermediates_to_track:
          inputs = batch['encoder_input_tokens']
          encoded_inputs = self.module.apply({'params': params},
                                             inputs,
                                             enable_dropout=dropout_rng
                                             is not None,
                                             rngs=dropout_rng,
                                             method=self.module.encode)
          intermediates['encoded_inputs'] = encoded_inputs

        intermediates_flat = traverse_util.flatten_dict(intermediates)
        intermediates = {}
        for key_flat, value in intermediates_flat.items():
          key = '/'.join(key_flat)
          if key in intermediates_to_track:
            intermediates[key] = value
      return sequence_scores, intermediates

    return sequence_scores

  def _compute_logits_from_slice(self,
                                 flat_ids,
                                 flat_cache,
                                 params,
                                 encoded_inputs,
                                 raw_inputs,
                                 max_decode_length,
                                 rngs=None,
                                 ensemble_probs=True):
    """Token slice to logits from decoder model.

    Different from the upstream implementation, here we assume that `params`,
    `encoded_inputs`, `rngs` are batched with ensemble size be their first
    dimension.

    Assume that the ensemble size is K. Shapes of the following variables are
      flat_ids: [K * batch * beam, seq_len=1]
      cache is expanded inside beam_search to become flat_cache
      flat_cache: [K * batch * beam, num_heads, depth_per_head, max_decode_len]
      flat_logits: [K * batch * beam, seq_len=1, vocab]

    For ensembling, we will apply beam search with the new batch size
    (K * batch). Here `flat_ids` and `flat_logits` should have the same values
    across K replicas. The `flat_cache` might have different values across K
    replicas because we will not perform ensemble over the cache.

    Args:
      flat_ids: The input token slices.
      flat_cache: The last cache value.
      params: A batch of sets of parameters.
      encoded_inputs: A batch of encoded inputs that correspond to the above
        batch of params.
      raw_inputs: The raw inputs used for encoder padding mask.
      max_decode_length: Maximum length of decoded sequence.
      rngs: A batch of optional dropout rngs to be used by inference-time
        dropout.
      ensemble_probs: Whether to perform ensemble in probs or logits spaces.

    Returns:
      A tuple of logits and the new cache.
    """
    # `params`, `encoded_inputs`, `rngs` have additional ensemble batch
    # dimensions [K, ...] so we rename them here to for readability.
    batch_params = params
    batch_encoded_inputs = encoded_inputs
    batch_rngs = rngs
    k = jax.tree_util.tree_flatten(encoded_inputs)[0][0].shape[0]
    # Turns flatten values into a batch of K replicas.
    batch_flat_ids = flat_ids.reshape((k, -1) + flat_ids.shape[1:])

    def _cache_reshape(x):
      if jnp.ndim(x) > 0:
        return x.reshape((k, -1) + x.shape[1:])
      else:
        return jnp.broadcast_to(x, (k,))

    batch_flat_cache = jax.tree_util.tree_map(_cache_reshape, flat_cache)

    def get_flat_logits_and_new_cache(vals):
      params, flat_cache, encoder_inputs, rngs, flat_ids = vals
      flat_logits, new_vars = self.module.apply(
          {
              'params': params,
              'cache': flat_cache
          },
          encoder_inputs,
          raw_inputs,  # only needed for encoder padding mask
          flat_ids,
          flat_ids,
          enable_dropout=rngs is not None,
          rngs=rngs,
          decode=True,
          max_decode_length=max_decode_length,
          mutable=['cache'],
          method=self.module.decode)
      # Remove sequence length dimension since it's always 1 during decoding.
      flat_logits = jnp.squeeze(flat_logits, axis=1)
      return flat_logits, new_vars['cache']

    batch_logits, batch_new_cache = _partial_map(
        get_flat_logits_and_new_cache,
        (batch_params, batch_flat_cache, batch_encoded_inputs, batch_rngs,
         batch_flat_ids))

    # Normalize the logits, assuming that last dimension is num_classes.
    batch_log_probs = jax.nn.log_softmax(batch_logits, axis=-1)
    if ensemble_probs:
      # Computes log(mean(exp(logits))) along the first dimension.
      flat_logits = (
          jax.nn.logsumexp(batch_log_probs, axis=0) -
          jnp.log(batch_log_probs.shape[0]))
    else:
      flat_logits = jnp.mean(batch_log_probs, axis=0)

    # Scatter the ensemble logits to K replicas.
    flat_logits = jnp.broadcast_to(flat_logits, (k,) + flat_logits.shape)
    flat_logits = jnp.reshape(flat_logits, (-1,) + flat_logits.shape[2:])

    # Make sure to reshape the new flat cache back to its original format.
    new_cache = _cache_concatenate(batch_new_cache)
    return flat_logits, new_cache

  def predict_batch_with_aux(
      self,
      params: t5x_models.PyTreeDef,
      batch: Mapping[str, jnp.ndarray],
      rng: Optional[jax.random.KeyArray] = None,
      decoder_params: Optional[MutableMapping[str, Any]] = None,
      return_all_decodes: bool = False,
      num_decodes: int = 1,
      dropout_rng: Optional[jnp.ndarray] = None,
      ensemble_probs: bool = True,
  ) -> Tuple[jnp.ndarray, Mapping[str, jnp.ndarray]]:
    """Predict with fast decoding beam search on a batch.

    Args:
      params: model parameters.
      batch: a batch of inputs.
      rng: an optional RNG key to use during prediction (e.g., for decoding).
      decoder_params: additional (model-independent) parameters for the decoder.
      return_all_decodes: whether to return the entire beam or just the top-1.
      num_decodes: the number of beams to use in beam search.
      dropout_rng: optional dropout rng. If a batch of rngs is provided, we will
        perform MC Dropout evaluation.
      ensemble_probs: Whether to perform ensemble in probs or logits spaces.

    Returns:
      A tuple containing:
        the batch of predictions, with the entire beam if requested
        an auxiliary dictionary of decoder scores
    """
    # Prepare zeroed-out autoregressive cache.
    # [batch, input_len]
    inputs = batch['encoder_input_tokens']
    # [batch, target_len]
    target_shape = batch['decoder_input_tokens'].shape
    target_type = batch['decoder_input_tokens'].dtype

    run_ensemble = isinstance(params, list)
    run_mc_dropout = (dropout_rng is not None) and (dropout_rng.ndim == 2)
    if run_ensemble:
      if dropout_rng is not None:
        raise ValueError('Dropout must be None for ensemble.')
      params_list = params
      batch_rngs = None
      k = len(params_list)
    elif run_mc_dropout:
      params_list = [params]
      batch_rngs = {'dropout': dropout_rng}
      k = dropout_rng.shape[0]
    else:
      params_list = [params]
      k = 1
      if dropout_rng is None:
        batch_rngs = None
      else:
        # Add a singleton dimension to dropout so its shape is [1, 2].
        batch_rngs = {'dropout': jnp.expand_dims(dropout_rng, 0)}

    batch_params = jax.tree_util.tree_map(lambda *vals: jnp.stack(vals),
                                               *params_list)

    def get_cache(args):
      params, rngs = args
      cache = self.module.apply({'params': params},
                                jnp.ones(inputs.shape, inputs.dtype),
                                jnp.ones(target_shape, target_type),
                                jnp.ones(target_shape, target_type),
                                decode=True,
                                enable_dropout=rngs is not None,
                                rngs=rngs,
                                mutable=['cache'])[1]['cache']
      return cache

    batch_cache = _partial_map(get_cache, (batch_params, batch_rngs))
    # Flatten (k, batch, ...) to (k * batch, ...) for beam search.
    cache = _cache_concatenate(batch_cache)

    # Prepare transformer fast-decoder call for beam search: for beam search, we
    # need to set up our decoder model to handle a batch size equal to
    # batch_size * num_decodes, where each batch item's data is expanded
    # in-place rather than tiled.
    # i.e. if we denote each batch element subtensor as el[n]:
    # [el0, el1, el2] --> beamsize=2 --> [el0,el0,el1,el1,el2,el2]
    # [batch * num_decodes, input_len, emb_dim]
    def get_encoded_inputs(args):
      params, rngs = args
      encoded_inputs = decoding.flat_batch_beam_expand(
          self.module.apply({'params': params},
                            inputs,
                            enable_dropout=rngs is not None,
                            rngs=rngs,
                            method=self.module.encode), num_decodes)
      return encoded_inputs

    batch_encoded_inputs = _partial_map(get_encoded_inputs,
                                        (batch_params, batch_rngs))

    # [batch * num_decodes, input_len]
    raw_inputs = decoding.flat_batch_beam_expand(inputs, num_decodes)

    tokens_ids_to_logits = functools.partial(
        self._compute_logits_from_slice,
        params=batch_params,
        encoded_inputs=batch_encoded_inputs,
        raw_inputs=raw_inputs,
        max_decode_length=target_shape[1],
        rngs=batch_rngs,
        ensemble_probs=ensemble_probs)

    if decoder_params is None:
      decoder_params = {}
    if rng is not None:
      if decoder_params.get('decode_rng') is not None:
        raise ValueError(
            f'Got RNG both from the `rng` argument ({rng}) and '
            f"`decoder_params['decode_rng']` ({decoder_params['decode_rng']}). "
            'Please specify one or the other.')
      decoder_params['decode_rng'] = rng

    # For beam search, `decoder_prompt_inputs` is only used to obtain batch size
    # and max decode length information. For temperature sampling,
    # `decode_prompt_inputs` will be filled with the sampled ids.
    batch_shape = batch['decoder_input_tokens'].shape
    batch_dtype = batch['decoder_input_tokens'].dtype
    # In ensemble beam search, the new batch size is (ensemble_size * batch).
    prompt_shape = (k * batch_shape[0],) + batch_shape[1:]
    decoder_prompt_inputs = jnp.zeros(prompt_shape, dtype=batch_dtype)

    # Using the above-defined single-step decoder function, run a
    # beam search over possible sequences given input encoding.
    # decodes: [k * batch, num_decodes, max_decode_len + 1]
    # scores: [k * batch, num_decodes]
    scanned = hasattr(self.module, 'scan_layers') and self.module.scan_layers
    decodes, scores = self._decode_fn(
        inputs=decoder_prompt_inputs,
        cache=cache,
        tokens_to_logits=tokens_ids_to_logits,
        eos_id=self.output_vocabulary.eos_id,
        num_decodes=num_decodes,
        cache_offset=1 if scanned else 0,
        **decoder_params)
    # The decodes and scores should be constant accross k replicas.
    # Here we only return 1 replicas.
    decodes = decodes[:batch_shape[0]]
    scores = scores[:batch_shape[0]]

    # Applies temperature scaling.
    scores = scores / self.temperature

    # Beam search returns [batch, n_beam, n_length] with beam dimension sorted
    # in increasing order of log-probability.
    # Return the highest scoring beam sequence.
    if return_all_decodes:
      return decodes, {'scores': scores}
    else:
      return decodes[:, -1, :], {'scores': scores[:, -1]}


# TODO(phandu): Rename this class to EncoderDecoderSequentialModel.
class EncoderDecoderBeamScoreModel(EncoderDecoderClassifierModel):
  """An EncoderDecoderModel that outputs beam scores for prediction.

  This model is equivalent to t5x's EncoderDecoderModel, but with its
  predict_batch() function is modified to output both the predicted sequence and
  the corresponding log-likelihood scores. The latter is needed for quantifying
  model uncertainty for the structured output.

  It is intended to be used for model inference with
  EncoderDecoderModel.predict_batch_with_aux()'s `num_decodes` set to > 1 and
  `return_all_decodes` set to `True`.
  """

  def predict_batch(
      self,
      params: t5x_models.PyTreeDef,
      batch: Mapping[str, jnp.ndarray],
      rng: Optional[jnp.ndarray] = None,
      return_scores: bool = False,
      num_mcdropout_samples: Optional[int] = None,
      dropout_seed: Optional[int] = None,
      ensemble_probs: bool = True,
  ) -> Union[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]]:
    """Thin wrapper around `self.predict_batch_with_aux`.

    Args:
      params: model parameters.
      batch: a batch of inputs.
      rng: an optional RNG key to use during prediction (e.g., for decoding).
      return_scores: whether to return log-likelihood scores along with the
        predicted sequence.
      num_mcdropout_samples: The number of dropout samples for evaluation. If
        provided, we will perform MC Dropout evaluation.
      dropout_seed: The seed to generate random keys for dropout samples.
      ensemble_probs: Whether to perform ensemble in probs or logits spaces.

    Returns:
      The model predictions with optional scores.
    """
    dropout_rng = None
    if dropout_seed is not None:
      dropout_rng = jax.random.PRNGKey(dropout_seed)
    if num_mcdropout_samples is not None:
      dropout_rng = jax.random.split(dropout_rng, num_mcdropout_samples)
    # The return value is a 2-tuple of the predicted sequences and the
    # scores for each predicted sequence.
    predictions, scores_dict = self.predict_batch_with_aux(
        params=params,
        batch=batch,
        rng=rng,
        dropout_rng=dropout_rng,
        ensemble_probs=ensemble_probs)
    scores = scores_dict['scores']

    if return_scores:
      return predictions, scores
    return predictions

  def score_batch(
      self,
      params: t5x_models.PyTreeDef,
      batch: Mapping[str, jnp.ndarray],
      return_intermediates: bool = False,
      *,
      dropout_seed: Optional[int] = None,
      num_mcdropout_samples: Optional[int] = None,
      ensemble_probs: bool = True,
      intermediates_to_track: Optional[Sequence[str]] = None,
  ) -> Union[jnp.ndarray, Tuple[jnp.ndarray, Mapping[str, Any]]]:
    """Compute logit scores on a batch.

    Args:
      params: model parameters.
      batch: a batch of inputs.
      return_intermediates: Whether to return the intermediate variables.
      dropout_seed: The seed to generate random keys for dropout samples.
      num_mcdropout_samples: The number of dropout samples for evaluation. If
        provided, we will perform MC Dropout evaluation.
      ensemble_probs: Whether to perform ensemble in probs or logits spaces.
      intermediates_to_track: A list/tuple of intermediate fields to return. We
        will flatten `intermediates` and only return values for flatten keys
        that are specified in this list/tuple.

    Returns:
      The logit scores on a batch and optional intermediate variables.
    """
    dropout_rng = None
    if dropout_seed is not None:
      dropout_rng = jax.random.PRNGKey(dropout_seed)
    if num_mcdropout_samples is not None:
      dropout_rng = jax.random.split(dropout_rng, num_mcdropout_samples)
    # Computes argmax predictive logits over the whole class,
    # shape (batch_size, output_len, num_class).
    sequence_scores = self._compute_argmax_score(params, batch,
                                                 return_intermediates,
                                                 dropout_rng, ensemble_probs)

    if return_intermediates:
      logits, intermediates = sequence_scores
    else:
      logits = sequence_scores

    weights = batch['decoder_loss_weights']
    target_tokens = batch['decoder_target_tokens']

    # Purposefully don't use config.z_loss because that term is for training
    # stability and shouldn't affect our reported scores.
    token_scores = -losses.cross_entropy_with_logits(
        logits,
        common_utils.onehot(
            target_tokens, jnp.shape(logits)[-1], on_value=1, off_value=0),
        z_loss=0.0)[0] * weights

    sequence_scores = token_scores.sum(-1)

    if return_intermediates:
      intermediates.setdefault('entropy', {})
      intermediates['entropy']['logits'] = logits
      intermediates['entropy']['token_entropy'] = _compute_token_entropy(logits)
      if intermediates_to_track:
        # infer binary's write_fn requires intermediates to be a flatten dict.
        # So we will return flatten dictionary here. For example, given
        # intermediates = {'entropy': {'token_entropy': 1.}}
        # the returned dict will be {'entropy/token_entropy': 1.}
        intermediates_flat = traverse_util.flatten_dict(intermediates)
        intermediates = {}
        for key_flat, value in intermediates_flat.items():
          key = '/'.join(key_flat)
          if key in intermediates_to_track:
            intermediates[key] = value
      return sequence_scores, intermediates

    return sequence_scores

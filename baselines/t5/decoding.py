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

"""Fast decoding routines for inference from a trained model.

This is a fork of `t5x.decoding` with modifications to also output token-level
probabilities. It is useful for simutaneous generation of predicted sequences
and their token-level probabilities.
"""
import functools

from typing import Any, Callable, Mapping, Optional, Tuple, Union
import flax
import jax
from jax import lax
from jax import random
import jax.numpy as jnp

from t5x import binary_search
import t5x.decoding as t5x_decoding

# Constants
# "Effective negative infinity" constant for masking in beam search.
NEG_INF = t5x_decoding.NEG_INF
MIN_TEMPERATURE = t5x_decoding.MIN_TEMPERATURE

# Imports symbols for data classes.
DecodingState = t5x_decoding.DecodingState

# Imports symbols for helper functions.
brevity_penalty = t5x_decoding.brevity_penalty
cache_map = t5x_decoding.cache_map
cache_gather_beams = t5x_decoding.cache_gather_beams
flatten_beam_dim = t5x_decoding.flatten_beam_dim
flat_batch_beam_expand = t5x_decoding.flat_batch_beam_expand
unflatten_beam_dim = t5x_decoding.unflatten_beam_dim
gather_beams = t5x_decoding.gather_beams
gather_topk_beams = t5x_decoding.gather_topk_beams
top_k_two_stage = t5x_decoding.top_k_two_stage

_is_tracer = t5x_decoding._is_tracer  # pylint:disable=protected-access
_dynamic_update_vector_slice_in_dim = t5x_decoding._dynamic_update_vector_slice_in_dim  # pylint:disable=protected-access

#------------------------------------------------------------------------------
# Temperature Sampling
#------------------------------------------------------------------------------


@flax.struct.dataclass
class SamplingLoopState:
  """Holds sampling state data.

  This data class is identical to `t5x_decoding.SamplingLoopState`, except for
  also including a `token_log_prob_no_temp` field that contains no-temperature
  token-level probabilities.

  Attributes:
    cur_index: [batch_size] array position of the sampling loop in the length
      dimension.
    sequences: [batch_size * num_decodes, max_decode_len] array of current
      sampled sequence prefixes.
    cache: any mapping of arrays, e.g. flax attention cache.
    cur_token: [batch_size, num_decodes] single timestep slice containing
      current tokens.
    ended: [batch_size, num_decodes] binary array marking completed sequences.
    rng: Jax PRNGKey
    log_prob: [batch_size, num_decodes] array of log probs for each sequence.
    token_log_prob_no_temp: [batch_size * num_decodes, max_decode_len] array of
      no-temperature token-level log probs for each sequence.
  """
  cur_index: jnp.ndarray
  sequences: jnp.ndarray
  cache: Mapping[str, jnp.ndarray]
  cur_token: jnp.ndarray
  ended: jnp.ndarray
  rng: jnp.ndarray
  log_prob: jnp.ndarray
  # Allows the loop state to carry token-level probabilities.
  token_log_prob_no_temp: jnp.ndarray


def temperature_sample(
    inputs: jnp.ndarray,
    cache: Mapping[str, jnp.ndarray],
    tokens_to_logits: Callable[[DecodingState],
                               Tuple[jnp.ndarray, Mapping[str, jnp.ndarray]]],
    eos_id: int,
    decode_rng: Optional[jnp.ndarray] = None,
    num_decodes: int = 1,
    temperature: Union[float, jnp.ndarray] = 1.0,
    topk: int = 1,
    topp: float = 0.0,
    cache_offset: int = 0,
    initial_index: Optional[jnp.ndarray] = None,
    max_decode_steps: Optional[Union[int, jnp.ndarray]] = None,
    max_decode_steps_hard_limit: Optional[int] = None,
    rescale_log_probs: bool = True,
    state_callback_fn: Optional[Callable[[SamplingLoopState],
                                         SamplingLoopState]] = None,
    logit_callback_fn: Optional[Callable[[jnp.ndarray, SamplingLoopState],
                                         jnp.ndarray]] = None,
    return_token_scores: bool = False
) -> Tuple[jnp.ndarray, jnp.ndarray]:
  """Temperature sampling with token-level probability outputs.

  This function is identical to `t5x_decoding.temperature_sample`, except that
  it supports returning token-level, no-temperature log probability in its
  decodes (controlled by the argument `return_token_scores`).

  Args:
    inputs: array: [batch_size, max_decode_len] int32 sequence of tokens.
    cache: flax attention cache.
    tokens_to_logits: fast autoregressive decoder function taking single token
      slices and cache and returning next-token logits and updated cache.
    eos_id: int: end-of-sentence token for target vocabulary.
    decode_rng: JAX PRNGKey.
    num_decodes: number of decoded sequences to be returned.
    temperature: float: sampling temperature factor. As it approaches zero this
      becomes equivalent to greedy sampling.
    topk: integer: if nonzero only use the top-k logits to sample next token, if
      zero don't use any cutoff and sample from full logits over vocabulary.
    topp: float: if nonzero only use the smallest number of logits whose
      cumulative sum of probs adds up to (at least) topp. Will raise ValueError
      if it's nonzero when topk is nonzero.
    cache_offset: axis offset for cache, arising from scanned layers.
    initial_index: Optional[array]: [batch_size] int32 a vector of loop indexes
      to start decoding at.
    max_decode_steps: int: an optional maximum number of decoding steps. If
      None, it will decode until the full input shape `inputs.shape[1]` is
      filled. max_decode_steps begins counting after the prompt, so it will
      decode at most len(prompt) + max_decode_steps tokens.
    max_decode_steps_hard_limit: int: an optional fixed hard limit on
      max_decode_steps. If this is set (not None and > 0), and max_decode_steps
      is also set, then max_decode_steps will be clipped to this limit. The
      value max_decode_steps can be an ndarray, but max_decode_steps_hard_limit
      must be a Python integer or None.
    rescale_log_probs: bool: whether to apply temperature, topp, and topk
      rescaling to the log probs which are returned. If True, the log_probs will
      include these transformations (for example, with topk=1, all log_probs
      will be identically 0.0). If False, the log_probs will not be affected,
      and topk/topp/temperature will not affect sequence probabilities.
    state_callback_fn: Function that modifies the sampling loop state before
      each step. This can be used to manipulate any part of the state either on
      the accelerator or on the host using host callback. The function should
      take a SamplingLoopState as argument, and it returns the updated state.
      See `decoding_test.py` for an example usage.
    logit_callback_fn: Function that modifies the logits before each temperature
      sampling step. The function should take arguments (logits, state) and it
      should return the modified logits. See `decoding_test.py` for an example
      usage.
    return_token_scores: Whether to return no-temperature token-level
      log probability scores instead of the sequence-level scores.

  Returns:
    A tuple (decodes, log_prob) where `decodes` is sampled sequences with shape
    [batch_size, num_decodes, max_decode_len] sorted by sequence-level log
    probability. `log_prob` is [batch_size, num_decodes] log probability of each
    of the sampled sequences (if `return_token_scores=False`), or the
    [batch_size, num_decodes, max_decode_len] log probability of each of the
    token in the sampled sequences (if `return_token_scores=True`).
  """
  if decode_rng is None:
    decode_rng = jax.random.PRNGKey(0)

  if (max_decode_steps_hard_limit is not None and
      max_decode_steps_hard_limit > 0 and max_decode_steps is not None):
    max_decode_steps = jnp.minimum(max_decode_steps,
                                   max_decode_steps_hard_limit)

  if num_decodes > 1:
    # [batch, len] -> [batch * num_decodes, len]
    expanded_inputs = flat_batch_beam_expand(inputs, num_decodes)
    expanded_cache = cache_map(
        functools.partial(
            flat_batch_beam_expand, beam_size=num_decodes, offset=cache_offset),
        cache,
        # When we start with a prefilled cache, the cache index is no longer a
        # scalar that will broadcast across multiple decodes, it is a vector and
        # needs to be updated to handle the multiple decodes.
        apply_to_index=initial_index is not None)
    if initial_index is not None:
      initial_index = flat_batch_beam_expand(initial_index, num_decodes)
  else:
    expanded_inputs = inputs
    expanded_cache = cache

  # NB: This is the main place we differ from the original function:
  # the `_temperature_sample_single_trial` returns token-level log_prob
  # (shape [batch * num_decodes, len]) rather than sequence-level log_prob
  # (shape [batch * num_decodes]).
  # expanded_decodes: [batch * num_decodes, len]
  # expanded_log_prob: [batch * num_decodes]
  # expanded_token_log_prob_no_temp: [batch * num_decodes, len]
  (expanded_decodes, expanded_log_prob,
   expanded_token_log_prob_no_temp) = _temperature_sample_single_trial(
       expanded_inputs,
       expanded_cache,
       tokens_to_logits,
       eos_id,
       decode_rng,
       temperature,
       topk,
       topp,
       initial_index=initial_index,
       max_decode_steps=max_decode_steps,
       rescale_log_probs=rescale_log_probs,
       state_callback_fn=state_callback_fn,
       logit_callback_fn=logit_callback_fn)

  batch_size = inputs.shape[0]
  # [batch * num_decodes, len] -> [batch, num_decodes, len]
  decodes = unflatten_beam_dim(expanded_decodes, batch_size, num_decodes)
  # [batch * num_decodes] -> [batch, num_decodes]
  log_prob = unflatten_beam_dim(expanded_log_prob, batch_size, num_decodes)
  # [batch * num_decodes, len] -> [batch, num_decodes, len]
  token_log_prob_no_temp = unflatten_beam_dim(
      expanded_token_log_prob_no_temp, batch_size, num_decodes)

  # Sort `decodes` and `log_prob` by increasing log probabilities of the sampled
  # sequence.
  # [batch, num_decodes, 1]
  idxs = jnp.expand_dims(jnp.argsort(log_prob, axis=-1), axis=-1)

  # returns sorted result.
  # [batch, num_decodes, len]
  decodes_sorted = jnp.take_along_axis(decodes, idxs, axis=1)

  # NB: This is the second place we differ from original function:
  # We allow model to return token-level score if `return_token_scores=True`.
  if return_token_scores:
    # [batch, num_decodes, len]
    log_prob_sorted = jnp.take_along_axis(
        token_log_prob_no_temp, idxs, axis=1)
  else:
    # [batch, num_decodes]
    log_prob_sorted = jnp.take_along_axis(
        log_prob, jnp.squeeze(idxs, axis=-1), axis=-1)

  return decodes_sorted, log_prob_sorted


def _temperature_sample_single_trial(
    inputs: jnp.ndarray,
    cache: Mapping[str, jnp.ndarray],
    tokens_to_logits: Callable[[DecodingState],
                               Tuple[jnp.ndarray, Mapping[str, jnp.ndarray]]],
    eos_id: int,
    prng_key: jnp.ndarray,
    temperature: Union[float, jnp.ndarray] = 1.0,
    topk: int = 20,
    topp: Union[float, jnp.ndarray] = 0.0,
    initial_index: Optional[jnp.ndarray] = None,
    max_decode_steps: Optional[Union[int, jnp.ndarray]] = None,
    rescale_log_probs: bool = True,
    state_callback_fn: Optional[Callable[[SamplingLoopState],
                                         SamplingLoopState]] = None,
    logit_callback_fn: Optional[Callable[[jnp.ndarray, SamplingLoopState],
                                         jnp.ndarray]] = None
) -> jnp.ndarray:
  """A helper function for `temperature_sample` with token-level probs."""
  # We can check the values of topp and topk only if they are not dynamic.
  if not _is_tracer(topp) and topp and topk:
    raise ValueError('At most one of `topp` or `topk` may be non-zero.')

  batch_size, max_decode_len = inputs.shape

  if max_decode_steps is not None:
    # We can check the max_decode_steps bounds only if it is not dynamic.
    if not _is_tracer(max_decode_steps) and max_decode_steps > inputs.shape[1]:
      raise ValueError('Cannot decode more steps than the sequence length.')

    # The number of decode steps required to process the prefix is the number
    #   of non-zero tokens, since inputs[0] == 0 is the BOS token.
    # `max_decode_len[j]` is the number of non-padding tokens in the jth element
    #   of the returned sequences capped at `len(inputs)`, assuming that the
    #   early stop doesn't occur. This is true with or without
    #   `max_decode_steps`.
    # When the while loop index `i` for the `j`th element `i[j] =
    #   max_decode_len[j] - 1`, the generated token populate sequences[i[j]+1]].
    #   Since sequences[:, 0] is BOS token, the generated token is
    #   `max_decode_len[j]`th non-padding tokens and hence `j`th element is
    #   ended.
    max_decode_len = jnp.sum(inputs != 0, axis=1) + max_decode_steps
    max_decode_len = jnp.minimum(inputs.shape[1], max_decode_len)

  # In the case of starting generation from a non-zero index, it is possible for
  # one batch element to reach `max_decode_len` number of decoding steps before
  # another. In order to let the last element decoder all the way to
  # `max_decode_len` number of steps, we add a final garbage token to the end of
  # the sequences. Any element that has reached `max_decode_len` before the rest
  # of the elements will continually overwrite this token until all elements
  # finish.
  # [batch, length+1] -> [batch, length+2]
  extra_input_tokens = 2
  expanded_prompt_inputs = jnp.append(
      inputs,
      jnp.zeros((batch_size, extra_input_tokens), dtype=inputs.dtype),
      axis=1)
  end_marker = jnp.array(eos_id)

  temperature = jnp.asarray(temperature)

  # Initialize sampling loop state.
  # initial loop PRNGKey
  rng0 = prng_key
  # the per batch-item holding current token in loop.
  if initial_index is None:
    # the per batch-item loop position counter.
    i0 = jnp.zeros((batch_size), dtype=jnp.int32)
    # the per batch-item holding current token in loop.
    token0 = jnp.zeros((batch_size, 1), dtype=jnp.int32)
  else:
    # the per batch-item loop position counter.
    i0 = initial_index
    # the per batch-item holding current token in loop.
    # Select the token that the initial index is pointing to.
    token0 = jnp.take_along_axis(
        expanded_prompt_inputs, jnp.expand_dims(i0, axis=1), axis=1)
  # per batch-item state bit indicating if sentence has finished.
  ended0 = jnp.zeros((batch_size, 1), dtype=jnp.bool_)
  # (batch, length+2) array containing prefix prompt tokens for sampling loop
  # as well as the generated output of newly sampled tokens.
  sequences0 = expanded_prompt_inputs
  log_prob0 = jnp.zeros((batch_size,), dtype=jnp.float32)
  # NB: This is the first place we differ from the original:
  # by adding an all-zero token_log_prob0 to the initialization.
  token_log_prob0 = jnp.zeros((batch_size, max_decode_len), dtype=jnp.float32)
  sampling_loop_init_state = SamplingLoopState(i0, sequences0, cache, token0,
                                               ended0, rng0, log_prob0,
                                               token_log_prob0)
  # Initial eos count to be used to determine whether eos is "generated". Many
  # inputs follow the format bos, inputs..., eos, targets..., eos. By counting
  # the number of eos tokens we can detect when a new one is added, instead of
  # just finding the one that probably ends the inputs.
  # [batch, 1]
  initial_eos_count = jnp.sum(sequences0 == end_marker, axis=-1, keepdims=True)

  def sampling_loop_cond_fn(state: SamplingLoopState) -> bool:
    """Sampling loop termination condition."""
    # Have all sampled sequences reached an end marker?
    # Different elements in the batch can be at different loop indices, if any
    # of our examples are not at the end, keep going.
    all_sequences_ended = jnp.all(state.ended)
    return ~all_sequences_ended

  def sampling_loop_body_fn(state: SamplingLoopState) -> SamplingLoopState:
    """Sampling loop state update."""

    if state_callback_fn is not None:
      state = state_callback_fn(state)

    # Split RNG for sampling.
    rng1, rng2 = random.split(state.rng)
    # Call fast-decoder model on current tokens to get next-position logits.
    decoding_state = DecodingState(
        cur_index=state.cur_index,
        sequences=state.sequences[:, :-extra_input_tokens],
        cur_token=state.cur_token,
        cache=state.cache)
    logits, new_cache = tokens_to_logits(decoding_state)
    # Sample next token from logits.

    if logit_callback_fn is not None:
      logits = logit_callback_fn(logits, state)

    def sample_logits_with_nonzero_temperature(logits):
      scaled_logits = logits / jnp.maximum(temperature, MIN_TEMPERATURE)
      if topk:
        scaled_logits = binary_search.topk_mask(scaled_logits, topk, NEG_INF)  # pytype: disable=wrong-arg-types  # jax-ndarray

      # When topp is dynamic, we always use it since we cannot check
      # non-zeroness (but it will have no effect if topp is 0.0).
      if _is_tracer(topp) or topp:
        scaled_logits = binary_search.topp_mask(scaled_logits, topp, NEG_INF)  # pytype: disable=wrong-arg-types  # jax-ndarray

      # [batch]
      next_token = random.categorical(rng1, scaled_logits).astype(jnp.int32)

      # log probability of the current token conditioned on the previously
      # sampled and prefix tokens.
      # [batch, vocab] -> [batch, vocab]
      if rescale_log_probs:
        log_probs = jax.nn.log_softmax(scaled_logits)
      else:
        log_probs = jax.nn.log_softmax(logits)

      # [batch, vocab] -> [batch]
      next_log_prob = jnp.squeeze(
          jnp.take_along_axis(
              log_probs, jnp.expand_dims(next_token, axis=1), axis=-1),
          axis=-1)

      # NB: This is the second place we differ from the original:
      # Appends no-temperature version of `next_log_prob`.
      # The logic is similar to above block.
      # [batch, vocab] -> [batch, vocab].
      log_probs_no_temp = jax.nn.log_softmax(logits)
      # [batch, vocab] -> [batch].
      next_log_prob_no_temp = jnp.squeeze(
          jnp.take_along_axis(
              log_probs_no_temp,
              jnp.expand_dims(next_token, axis=1), axis=-1), axis=-1)

      return (next_token, next_log_prob, next_log_prob_no_temp)

    def sample_logits_with_no_temperature(logits):
      # For zero temperature, we always want the greedy output, regardless
      # of the values of topk and topp.

      next_token = jnp.argmax(logits, -1).astype(jnp.int32)

      if rescale_log_probs:
        next_log_prob = jnp.zeros_like(next_token, dtype=jnp.float32)
      else:
        log_probs = jax.nn.log_softmax(logits)
        next_log_prob = jnp.squeeze(
            jnp.take_along_axis(
                log_probs, jnp.expand_dims(next_token, axis=1), axis=-1),
            axis=-1)

      # NB: This is the 3rd place we differ from the original:
      # Appends no-temperature version of `next_log_prob`, so the signature is
      # consistent with `sample_logits_with_nonzero_temperature`.
      next_log_prob_no_temp = next_log_prob.copy()
      return (next_token, next_log_prob, next_log_prob_no_temp)

    # Perform sampling with temperature
    (next_token, next_log_prob,
     next_log_prob_no_temp) = lax.cond(temperature > MIN_TEMPERATURE,
                                       sample_logits_with_nonzero_temperature,
                                       sample_logits_with_no_temperature,
                                       logits)

    # When different batch elements are at different points in the loop counter,
    # it is possible that an element that started at a higher index will reach
    # `max_decode_len` before other elements. When this happens we need to make
    # sure this element continuous overwrites our new garbage collection index.
    # Here we clamp `i` to `max_decode_len`. This will cause the a write to
    # `max_decode_len + 1` which is the final index in `sequences`. Subsequent
    # loop body executions will also get their value clamped causing continual
    # overwriting of the final garbage position until all examples are finished.
    i = jnp.minimum(state.cur_index, max_decode_len)

    # Only use sampled tokens if we're past provided prefix tokens.
    # Select the next token from sequences.
    # [batch]
    next_input_token = jnp.squeeze(
        jnp.take_along_axis(
            state.sequences, jnp.expand_dims(i + 1, axis=1), axis=1),
        axis=1)
    # Check if the next token is padding (a target) or non-padding (an input).
    # Mask will have `1` for targets and `0` for inputs.
    out_of_prompt = (next_input_token == 0)
    # Select the sampled next token for targets and the actual next token for
    # inputs (teacher forcing).
    # [batch]
    next_token = (
        next_token * out_of_prompt + next_input_token * ~out_of_prompt)

    # NB: This is the 4th place we differ from the original:
    # by appending new no-temperature `next_log_prob` to the
    # `next_token_log_prob`. Only add probability if outside prefix region.
    # [batch, len] -> [batch, len]
    next_log_prob_to_append = (
        next_log_prob_no_temp * out_of_prompt) * jnp.squeeze(
            ~state.ended, axis=-1).astype(jnp.int32)
    next_token_log_prob_no_temp = (
        state.token_log_prob_no_temp.at[:, i].set(next_log_prob_to_append))

    # only add probability if outside prefix region
    # [batch] -> [batch]
    next_log_prob = state.log_prob + (
        next_log_prob * out_of_prompt) * jnp.squeeze(
            ~state.ended, axis=-1).astype(jnp.int32)

    # [batch] -> [batch, 1]
    next_token = jnp.expand_dims(next_token, axis=-1)

    # If end-marker reached for batch item, only emit padding tokens.
    # [batch, 1] * [batch, 1] -> [batch, 1]
    next_token_or_endpad = next_token * ~state.ended
    # Add current sampled tokens to recorded sequences.
    one_hot = jax.nn.one_hot(
        i + 1, state.sequences.shape[1], dtype=state.sequences.dtype)
    new_sequences = state.sequences * (1 -
                                       one_hot) + next_token_or_endpad * one_hot
    # new_sequences = dynamic_update_vector_slice_in_dim(sequences,
    #                                                    next_token_or_endpad,
    #                                                    i + 1,
    #                                                    0)
    # Count eos tokens in the sequences and compare to the initial count
    # [batch, 1]
    cur_eos_count = jnp.sum(new_sequences == end_marker, axis=-1, keepdims=True)
    # [batch, 1]

    # Have we reached max decoding length?
    # We generally index into sequences[:, i + 1], and sequences.shape[1] =
    # max_decode_len + 2, therefore i == max_decode_len - 1 will write to
    # sequences[-2] which is our last valid location. i == max_decode_len will
    # write to sequences[-1] which is our garbage collection token. Thus `i`
    # should be strictly less than max_decode_len.
    has_additional_eos = cur_eos_count > initial_eos_count
    ended = state.ended | has_additional_eos | jnp.expand_dims(
        i >= max_decode_len - 1, axis=1)

    # NB: This is the 5th place we differ from the original:
    # Added `next_token_log_prob` to the `SamplingLoopState`.
    return SamplingLoopState(i + 1, new_sequences, new_cache,
                             next_token_or_endpad, ended, rng2,
                             next_log_prob, next_token_log_prob_no_temp)

  # Run sampling loop and collect final state.
  final_state = lax.while_loop(sampling_loop_cond_fn, sampling_loop_body_fn,
                               sampling_loop_init_state)

  # Pick part of the state corresponding to the sampled sequences.
  final_sequences = final_state.sequences
  log_prob = final_state.log_prob
  # NB: This is the last place we differ from the original:
  # we also return the `token_log_prob`.
  token_log_prob = final_state.token_log_prob_no_temp
  # Drop the first position because they are dummy bos tokens. Drop the new
  # garbage collection token at the end too.
  return final_sequences[:, 1:-1], log_prob, token_log_prob  # pytype: disable=bad-return-type  # jax-ndarray


#------------------------------------------------------------------------------
# BEAM Sampling
#------------------------------------------------------------------------------


@flax.struct.dataclass
class BeamState:
  """Holds beam search state data."""
  # The position of the decoding loop in the length dimension.
  cur_index: jax.Array  # scalar int32: current decoded length index
  # The active sequence log probabilities and finished sequence scores.
  live_logprobs: jax.Array  # float32: [batch_size, beam_size]
  finished_scores: jax.Array  # float32: [batch_size, beam_size]
  # The current active-beam-searching and finished sequences.
  live_seqs: jax.Array  # int32: [batch_size, beam_size, max_decode_len]
  live_prob_seqs: jax.Array  # int32: [batch_size, beam_size,
  #                                          max_decode_len]
  finished_seqs: jax.Array  # int32: [batch_size, beam_size,
  #                                         max_decode_len]
  finished_prob_seqs: jax.Array  # int32: [batch_size, beam_size,
  #                                              max_decode_len]
  # Records which of the 'finished_seqs' is occupied and not a filler slot.
  finished_flags: jax.Array  # bool: [batch_size, beam_size]
  # The current state of the autoregressive decoding caches.
  # Any pytree of arrays, e.g. flax attention Cache object
  cache: Any


def beam_init(batch_size: int,
              beam_size: int,
              max_decode_len: int,
              cache: Mapping[str, jnp.ndarray],
              offset: int = 0) -> BeamState:
  """Initializes the beam search state data structure."""
  cur_index0 = jnp.array(0)
  live_logprobs0 = jnp.tile(
      jnp.array([0.0] + [NEG_INF] * (beam_size - 1)), [batch_size, 1])
  finished_scores0 = jnp.ones((batch_size, beam_size)) * NEG_INF
  live_seqs0 = jnp.zeros((batch_size, beam_size, max_decode_len), jnp.int32)
  live_prob_seqs0 = jnp.zeros((batch_size, beam_size, max_decode_len),
                              jnp.float32)
  finished_seqs0 = jnp.zeros((batch_size, beam_size, max_decode_len), jnp.int32)
  finished_prob_seqs0 = jnp.zeros((batch_size, beam_size, max_decode_len),
                                  jnp.float32)
  finished_flags0 = jnp.zeros((batch_size, beam_size), jnp.bool_)
  # add beam dimension to attention cache pytree elements
  beam_cache0 = cache_map(
      lambda x: t5x_decoding.add_beam_dim(x, beam_size, offset), cache)
  return BeamState(
      cur_index=cur_index0,
      live_logprobs=live_logprobs0,
      finished_scores=finished_scores0,
      live_seqs=live_seqs0,
      live_prob_seqs=live_prob_seqs0,
      finished_seqs=finished_seqs0,
      finished_prob_seqs=finished_prob_seqs0,
      finished_flags=finished_flags0,
      cache=beam_cache0)


# Beam search routine:


def beam_search(
    inputs: jnp.ndarray,
    cache: Mapping[str, jnp.ndarray],
    tokens_to_logits: Callable[[DecodingState],
                               Tuple[jnp.ndarray, Mapping[str, jnp.ndarray]]],
    eos_id: int,
    num_decodes: int = 4,
    alpha: float = 0.6,
    max_decode_len: Optional[int] = None,
    decode_rng: Optional[jnp.ndarray] = None,
    cache_offset: int = 0,
    return_token_scores: bool = False,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
  """Beam search for transformer machine translation.

  Args:
    inputs: array: [batch_size, length] int32 sequence of tokens.
    cache: flax attention cache.
    tokens_to_logits: fast autoregressive decoder function taking single token
      slices and cache and returning next-token logits and updated cache.
    eos_id: int: id of end-of-sentence token for target vocabulary.
    num_decodes: number of decoded sequences to be returned. This is equivalent
      to the number of beams used in the beam search.
    alpha: float: scaling factor for brevity penalty.
    max_decode_len: int: an optional maximum length of decoded sequence. If
      None, it uses `inputs.shape[1]` as `max_decode_len`.
    decode_rng: Unused decoder RNG seed.
    cache_offset: axis offset for cache, arising from scanned layers.
    return_token_scores: Whether to return token-level probability of the
      decoded sequences.

  Returns:
     Tuple of:
       [batch_size, beam_size, max_decode_len] top-scoring sequences
       [batch_size, beam_size] beam-search scores. or shape
       [batch_size, beam_size, max_decode_len] if `return_token_scores=True`.
  """
  del decode_rng
  # We liberally annotate shape information for clarity below.

  beam_size = num_decodes

  batch_size = inputs.shape[0]
  end_marker = jnp.array(eos_id)
  if max_decode_len is None:
    max_decode_len = inputs.shape[1]
  # We start with a dummy token in the beginning so extend the maximum length.
  max_decode_len += 1

  # initialize beam search state
  beam_search_init_state = beam_init(batch_size, beam_size, max_decode_len,
                                     cache, cache_offset)

  def beam_search_loop_cond_fn(state: BeamState) -> bool:
    """Beam search loop termination condition."""
    # Have we reached max decoding length?
    # Because we mutate the "i+1" position, we stop one token before the end.
    not_at_end = (state.cur_index < max_decode_len - 1)

    # Is no further progress in the beam search possible?
    # Get the best possible scores from alive sequences.
    min_brevity_penalty = brevity_penalty(alpha, max_decode_len)
    best_live_scores = state.live_logprobs[:, -1:] / min_brevity_penalty
    # Get the worst scores from finished sequences.
    worst_finished_scores = jnp.min(
        state.finished_scores, axis=1, keepdims=True)
    # Mask out scores from slots without any actual finished sequences.
    worst_finished_scores = jnp.where(state.finished_flags,
                                      worst_finished_scores, NEG_INF)
    # If no best possible live score is better than current worst finished
    # scores, the search cannot improve the finished set further.
    search_terminated = jnp.all(worst_finished_scores > best_live_scores)

    # If we're not at the max decode length, and the search hasn't terminated,
    # continue looping.
    return not_at_end & (~search_terminated)  # pytype: disable=bad-return-type  # jax-devicearray

  def beam_search_loop_body_fn(state: BeamState) -> BeamState:
    """Beam search loop state update function."""
    # Collect the current position slice along length to feed the fast
    # autoregressive decoder model.  Flatten the beam dimension into batch
    # dimension for feeding into the model.
    # --> [batch * beam, 1]
    flat_ids = flatten_beam_dim(
        lax.dynamic_slice(state.live_seqs, (0, 0, state.cur_index),
                          (batch_size, beam_size, 1)))
    # Flatten beam dimension into batch to be compatible with model.
    # {[batch, beam, ...], ...} --> {[batch * beam, ...], ...}
    flat_cache = cache_map(
        functools.partial(flatten_beam_dim, offset=cache_offset), state.cache)

    # Call fast-decoder model on current tokens to get next-position logits.
    # --> [batch * beam, vocab]
    decoding_state = DecodingState(
        cur_index=state.cur_index,
        sequences=flatten_beam_dim(state.live_seqs),
        cur_token=flat_ids,
        cache=flat_cache)
    flat_logits, new_flat_cache = tokens_to_logits(decoding_state)

    # unflatten beam dimension
    # [batch * beam, vocab] --> [batch, beam, vocab]
    logits = unflatten_beam_dim(flat_logits, batch_size, beam_size)
    # Unflatten beam dimension in attention cache arrays
    # {[batch * beam, ...], ...} --> {[batch, beam, ...], ...}
    new_cache = cache_map(
        lambda x: unflatten_beam_dim(x, batch_size, beam_size, cache_offset),
        new_flat_cache)

    # Gather log probabilities from logits.
    # --> [batch, beam, vocab]
    candidate_log_probs = jax.nn.log_softmax(logits)
    # Add new logprobs to existing prefix logprobs.
    # --> [batch, beam, vocab]
    log_probs = (
        candidate_log_probs + jnp.expand_dims(state.live_logprobs, axis=2))

    # We'll need the vocab size, gather it from the log probability dimension.
    vocab_size = log_probs.shape[-1]

    # Each item in batch has beam_size * vocab_size candidate sequences.
    # For each item, get the top 2*k candidates with the highest log-
    # probabilities. We gather the top 2*K beams here so that even if the best
    # K sequences reach EOS simultaneously, we have another K sequences
    # remaining to continue the live beam search.
    beams_to_keep = 2 * beam_size
    # Flatten beam and vocab dimensions.
    flat_log_probs = log_probs.reshape((batch_size, beam_size * vocab_size))
    # Gather the top 2*K scores from _all_ beams.
    # --> [batch, 2*beams], [batch, 2*beams]
    topk_log_probs, topk_indices = top_k_two_stage(
        flat_log_probs, k=beams_to_keep)
    # Recover the beam index by floor division.
    topk_beam_indices = topk_indices // vocab_size

    # Gather 2*k top beams and their token-level scores.
    # --> [batch, 2*beams, length]
    topk_seq = gather_beams(state.live_seqs, topk_beam_indices, batch_size,
                            beam_size, beams_to_keep)
    topk_prob_seq = gather_beams(state.live_prob_seqs, topk_beam_indices,
                                 batch_size, beam_size, beams_to_keep)

    # Append the most probable 2*K token IDs to the top 2*K sequences
    # Recover token id by modulo division and expand Id array for broadcasting.
    # --> [batch, 2*beams, 1]
    topk_ids = jnp.expand_dims(topk_indices % vocab_size, axis=2)
    topk_probs = jnp.expand_dims(topk_log_probs, axis=2)

    # Update sequences for the 2*K top-k new sequences.
    # --> [batch, 2*beams, length]
    topk_seq = lax.dynamic_update_slice(topk_seq, topk_ids,
                                        (0, 0, state.cur_index + 1))
    topk_prob_seq = lax.dynamic_update_slice(topk_prob_seq, topk_probs,
                                             (0, 0, state.cur_index + 1))

    # Update LIVE (in-progress) sequences:
    # Did any of these sequences reach an end marker?
    # --> [batch, 2*beams]
    newly_finished = (topk_seq[:, :, state.cur_index + 1] == end_marker)
    # To prevent these newly finished sequences from being added to the LIVE
    # set of active beam search sequences, set their log probs to a very large
    # negative value.
    new_log_probs = topk_log_probs + newly_finished * NEG_INF
    # Determine the top k beam indices (from top 2*k beams) from log probs.
    # --> [batch, beams]
    _, new_topk_indices = lax.top_k(new_log_probs, k=beam_size)
    new_topk_indices = jnp.flip(new_topk_indices, axis=1)
    # Gather the top k beams (from top 2*k beams).
    # --> [batch, beams, length], [batch, beams]
    top_alive_seq, top_alive_prob_seq, top_alive_log_probs = gather_beams(
        [topk_seq, topk_prob_seq, new_log_probs], new_topk_indices, batch_size,
        2 * beam_size, beam_size)

    # Determine the top k beam indices from the original set of all beams.
    # --> [batch, beams]
    top_alive_indices = gather_beams(topk_beam_indices, new_topk_indices,
                                     batch_size, 2 * beam_size, beam_size)
    # With these, gather the top k beam-associated caches.
    # --> {[batch, beams, ...], ...}
    top_alive_cache = cache_gather_beams(new_cache, top_alive_indices,
                                         batch_size, beam_size, beam_size, True,
                                         cache_offset)

    # Update FINISHED (reached end of sentence) sequences:
    # Calculate new seq scores from log probabilities.
    new_scores = topk_log_probs / brevity_penalty(alpha, state.cur_index + 1)  # pytype: disable=wrong-arg-types  # jax-devicearray
    # Mask out the still unfinished sequences by adding large negative value.
    # --> [batch, 2*beams]
    new_scores += (~newly_finished) * NEG_INF

    # Combine sequences, scores, and flags along the beam dimension and compare
    # new finished sequence scores to existing finished scores and select the
    # best from the new set of beams.
    finished_seqs = jnp.concatenate(  # --> [batch, 3*beams, length]
        [state.finished_seqs, topk_seq],
        axis=1)
    finished_prob_seqs = jnp.concatenate(  # --> [batch, 3*beams, length]
        [state.finished_prob_seqs, topk_prob_seq],
        axis=1)
    finished_scores = jnp.concatenate(  # --> [batch, 3*beams]
        [state.finished_scores, new_scores], axis=1)
    finished_flags = jnp.concatenate(  # --> [batch, 3*beams]
        [state.finished_flags, newly_finished], axis=1)
    # --> [batch, beams, length], [batch, beams, length],
    # --> [batch, beams], [batch, beams]
    (top_finished_seq, top_finished_prob_seq, top_finished_scores,
     top_finished_flags) = gather_topk_beams(
         [finished_seqs, finished_prob_seqs, finished_scores, finished_flags],
         finished_scores, batch_size, beam_size)

    return BeamState(
        cur_index=state.cur_index + 1,
        live_logprobs=top_alive_log_probs,
        finished_scores=top_finished_scores,
        live_seqs=top_alive_seq,
        live_prob_seqs=top_alive_prob_seq,
        finished_seqs=top_finished_seq,
        finished_prob_seqs=top_finished_prob_seq,
        finished_flags=top_finished_flags,
        cache=top_alive_cache)

  # Run while loop and get final beam search state.
  final_state = lax.while_loop(beam_search_loop_cond_fn,
                               beam_search_loop_body_fn, beam_search_init_state)

  # Account for the edge-case where there are no finished sequences for a
  # particular batch item. If so, return live sequences for that batch item.
  # --> [batch]
  none_finished = jnp.any(final_state.finished_flags, axis=1)
  # --> [batch, beams, length]
  finished_seqs = jnp.where(none_finished[:, None, None],
                            final_state.finished_seqs, final_state.live_seqs)
  # --> [batch, beams]
  finished_scores = jnp.where(none_finished[:,
                                            None], final_state.finished_scores,
                              final_state.live_logprobs)
  # --> [batch, beams, length]
  finished_prob_seqs = jnp.where(none_finished[:, None, None],
                                 final_state.finished_prob_seqs,
                                 final_state.live_prob_seqs)

  # Converts the cumulative sum of log probs to token-level log probs.
  # --> [batch, beams, length]
  finished_token_scores = jnp.diff(finished_prob_seqs, axis=2)
  # Clip the last value caused by the `diff` operator of the padding
  # value 0.0 and the last cumulative sum. That value is positive because
  # the sum of all log probabilities is negative.
  finished_token_scores = jnp.clip(finished_token_scores, a_max=0.)

  if return_token_scores:
    return finished_seqs[:, :, 1:], finished_token_scores
  return finished_seqs[:, :, 1:], finished_scores

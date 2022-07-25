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

"""Fast decoding routines for inference from a trained model."""
import functools

from typing import Callable, Mapping, Optional, Tuple
import flax
import jax
from jax import lax
import jax.numpy as jnp

import t5x.decoding as t5x_decoding

# Constants
# "Effective negative infinity" constant for masking in beam search.
NEG_INF = t5x_decoding.NEG_INF

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

#------------------------------------------------------------------------------
# BEAM Sampling
#------------------------------------------------------------------------------


@flax.struct.dataclass
class BeamState:
  """Holds beam search state data."""
  # The position of the decoding loop in the length dimension.
  cur_index: jnp.DeviceArray  # scalar int32: current decoded length index
  # The active sequence log probabilities and finished sequence scores.
  live_logprobs: jnp.DeviceArray  # float32: [batch_size, beam_size]
  finished_scores: jnp.DeviceArray  # float32: [batch_size, beam_size]
  # The current active-beam-searching and finished sequences.
  live_seqs: jnp.DeviceArray  # int32: [batch_size, beam_size, max_decode_len]
  live_prob_seqs: jnp.DeviceArray  # int32: [batch_size, beam_size,
  #                                          max_decode_len]
  finished_seqs: jnp.DeviceArray  # int32: [batch_size, beam_size,
  #                                         max_decode_len]
  finished_prob_seqs: jnp.DeviceArray  # int32: [batch_size, beam_size,
  #                                              max_decode_len]
  # Records which of the 'finished_seqs' is occupied and not a filler slot.
  finished_flags: jnp.DeviceArray  # bool: [batch_size, beam_size]
  # The current state of the autoregressive decoding caches.
  # Any pytree of arrays, e.g. flax attention Cache object
  cache: t5x_decoding.PyTreeDef


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
    tokens_to_logits: Callable[[jnp.ndarray, Mapping[str, jnp.ndarray]],
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
    return not_at_end & (~search_terminated)

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
    flat_logits, new_flat_cache = tokens_to_logits(flat_ids, flat_cache)

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
    new_scores = topk_log_probs / brevity_penalty(alpha, state.cur_index + 1)
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

  if return_token_scores:
    return finished_seqs[:, :, 1:], finished_token_scores
  return finished_seqs[:, :, 1:], finished_scores

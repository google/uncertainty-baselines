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

"""Heteroscedastic model with T5 backbone."""
from typing import Any, Optional

import edward2.jax as ed
import flax.linen as nn
import jax
import jax.numpy as jnp
import t5x.examples.t5.layers as t5_layers
import t5x.examples.t5.network as t5_network

# Jax data types.
Array = Any


# This class follows `network.Decoder` implementation.
class HeteroscedasticDecoder(nn.Module):
  """T5 Decoder with Heteroscedastic output head."""
  config: t5_network.T5Config
  shared_embedding: nn.Module
  # Heteroscedastic parameters.
  temperature: float = 1.0
  mc_samples: int = 1000
  num_factors: int = 0
  param_efficient: bool = False
  return_locs: bool = False
  share_samples_across_batch: bool = False
  tune_temperature: bool = False
  temperature_lower_bound: Optional[float] = None
  temperature_upper_bound: Optional[float] = None
  latent_dim: Optional[int] = None
  cov_layer_kernel_init_scale: Optional[float] = None

  def setup(self):
    if self.config.logits_via_embedding:
      raise ValueError('Sharing the embedding weights in the decoder output '
                       'layer is not supported in the heteroscedastic decoder.')
    softmax_het_layer = ed.nn.MCSoftmaxDenseFA
    self.heteroscedastic_layer = softmax_het_layer(
        self.config.vocab_size,
        self.num_factors,
        self.temperature,
        self.param_efficient,
        self.mc_samples,
        self.mc_samples,
        logits_only=True,
        return_locs=self.return_locs,
        share_samples_across_batch=self.share_samples_across_batch,
        tune_temperature=self.tune_temperature,
        temperature_lower_bound=self.temperature_lower_bound,
        temperature_upper_bound=self.temperature_upper_bound,
        latent_dim=self.latent_dim,
        cov_layer_kernel_init_scale=self.cov_layer_kernel_init_scale,
        name='heteroscedastic_head')

  @nn.compact
  def __call__(self,
               encoded,
               decoder_input_tokens,
               decoder_positions=None,
               decoder_mask=None,
               encoder_decoder_mask=None,
               deterministic=False,
               decode=False,
               max_decode_length=None) -> Array:
    cfg = self.config
    assert decoder_input_tokens.ndim == 2  # [batch, len]
    rel_emb = t5_layers.RelativePositionBiases(
        num_buckets=32,
        max_distance=128,
        num_heads=cfg.num_heads,
        dtype=cfg.dtype,
        embedding_init=nn.initializers.variance_scaling(1.0, 'fan_avg',
                                                        'uniform'),
        name='relpos_bias')

    # [batch, length] -> [batch, length, emb_dim]
    y = self.shared_embedding(decoder_input_tokens.astype('int32'))
    y = nn.Dropout(
        rate=cfg.dropout_rate, broadcast_dims=(-2,))(
            y, deterministic=deterministic)
    y = y.astype(cfg.dtype)

    for lyr in range(cfg.num_decoder_layers):
      # [batch, length, emb_dim] -> [batch, length, emb_dim]
      y = t5_network.DecoderLayer(
          config=cfg,
          relative_embedding=rel_emb,
          name=f'layers_{lyr}')(
              y,
              encoded,
              decoder_mask=decoder_mask,
              encoder_decoder_mask=encoder_decoder_mask,
              deterministic=deterministic,
              decode=decode,
              max_decode_length=max_decode_length)

    y = t5_layers.LayerNorm(dtype=cfg.dtype, name='decoder_norm')(y)
    y = nn.Dropout(
        rate=cfg.dropout_rate, broadcast_dims=(-2,))(
            y, deterministic=deterministic)

    # NOTE: This is the place that will be different from the T5 transformer.
    # [batch, length, emb_dim] -> [batch, length, vocab_size]
    assert y.ndim == 3
    batch, length = y.shape[:2]  # pylint: disable=unreachable
    logits = self.heteroscedastic_layer(y.reshape((batch * length, -1)))
    logits = logits.reshape((batch, length, -1))

    return logits


class TransformerHeteroscedastic(t5_network.Transformer):
  """T5 Transformer with Heteroscedastic output head."""
  config: t5_network.T5Config
  # Heteroscedastic parameters.
  temperature: float = 1.0
  mc_samples: int = 1000
  num_factors: int = 0
  param_efficient: bool = False
  return_locs: bool = False
  eval_rng_seed: int = 0
  share_samples_across_batch: bool = False
  tune_temperature: bool = False
  temperature_lower_bound: Optional[float] = None
  temperature_upper_bound: Optional[float] = None
  latent_dim: Optional[int] = None
  cov_layer_kernel_init_scale: Optional[float] = None

  def setup(self):
    cfg = self.config
    self.shared_embedding = t5_layers.Embed(
        num_embeddings=cfg.vocab_size,
        features=cfg.emb_dim,
        dtype=cfg.dtype,
        attend_dtype=jnp.float32,  # for logit training stability
        embedding_init=nn.initializers.normal(stddev=1.0),
        one_hot=True,
        name='token_embedder')

    self.encoder = t5_network.Encoder(
        config=cfg, shared_embedding=self.shared_embedding)
    self.decoder = HeteroscedasticDecoder(
        config=cfg,
        shared_embedding=self.shared_embedding,
        temperature=self.temperature,
        mc_samples=self.mc_samples,
        num_factors=self.num_factors,
        param_efficient=self.param_efficient,
        return_locs=self.return_locs,
        share_samples_across_batch=self.share_samples_across_batch,
        tune_temperature=self.tune_temperature,
        temperature_lower_bound=self.temperature_lower_bound,
        temperature_upper_bound=self.temperature_upper_bound,
        latent_dim=self.latent_dim,
        cov_layer_kernel_init_scale=self.cov_layer_kernel_init_scale,
    )

  @staticmethod
  def modify_rngs(eval_rng_seed, rngs):
    # For evaluation, we use a constant seed specified in eval_rng_seed.
    if rngs is None:
      rng = jax.random.PRNGKey(eval_rng_seed)
      diag_noise_rng, standard_noise_rng = jax.random.split(rng)
      rngs = {
          'diag_noise_samples': diag_noise_rng,
          'standard_norm_noise_samples': standard_noise_rng
      }
    else:
      if 'dropout' in rngs:
        split_rng = 'dropout'
      elif 'params' in rngs:
        split_rng = 'params'
      else:
        raise ValueError('Missing `dropout` or `params` rng for the network.')
      rng_updates = dict(
          zip([split_rng, 'diag_noise_samples', 'standard_norm_noise_samples'],
              jax.random.split(rngs[split_rng], 3)))
      rngs.update(rng_updates)

    assert 'diag_noise_samples' in rngs
    assert 'standard_norm_noise_samples' in rngs
    return rngs

  def apply(self, *args, **kwargs):
    rngs = kwargs.get('rngs', None)
    rngs = self.modify_rngs(self.eval_rng_seed, rngs)
    kwargs['rngs'] = rngs
    return super().apply(*args, **kwargs)

  def init(self, rngs, *args, **kwargs):
    if not isinstance(rngs, dict):
      rngs = {'params': rngs}
    rngs = self.modify_rngs(self.eval_rng_seed, rngs)
    return super().init(rngs, *args, **kwargs)

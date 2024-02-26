# coding=utf-8
# Copyright 2024 The Uncertainty Baselines Authors.
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

"""BatcheEnsemble Decoder with GP last layer."""

from typing import Callable, Iterable, Optional, Tuple

import edward2.jax as ed
import flax.linen as nn
import jax.numpy as jnp
import t5x.examples.t5.layers as t5_layers
import t5x.examples.t5.network as t5_network

from uncertainty_baselines.models import t5_batchensemble
from uncertainty_baselines.models import t5_gp

# Type annotations
Array = jnp.ndarray
DType = jnp.dtype
PRNGKey = jnp.ndarray
Shape = Iterable[int]
Activation = Callable[..., Array]
# Parameter initializers.
Initializer = Callable[[PRNGKey, Shape, DType], Array]


class BEGpDecoder(t5_gp.GaussianProcessDecoder):
  """T5 Decoder with BatchEnsemble and GP layers."""
  config: t5_network.T5Config
  shared_embedding: nn.Module
  ens_size: int = 1
  random_sign_init: float = 0.5
  be_decoder_layers: Optional[Tuple[int, ...]] = None
  # Gaussian process parameters.
  use_gp_layer: bool = True
  covmat_momentum: float = -1.
  normalize_input: bool = True
  # NOTE: We let mean_field_factor as in the constructor to
  # preserve the signature of `__call__` method.
  mean_field_factor: float = -1.
  ridge_penalty: float = 1.
  steps_per_epoch: Optional[int] = None

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
    if not self.use_gp_layer:
      raise ValueError(
          'If use_gp_layer is False, use BatchEnsembleDecoder instead.'
      )
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

    be_decoder_layers = [
        x if x >= 0 else (x + cfg.num_decoder_layers)
        for x in list(self.be_decoder_layers)
        if x is not None
    ]
    for lyr in range(cfg.num_decoder_layers):
      # [batch, length, emb_dim] -> [batch, length, emb_dim]
      if lyr in be_decoder_layers:
        if lyr != cfg.num_decoder_layers - 1:
          raise ValueError('Currently only support lastdecoder layer to be BE.')
        y = t5_batchensemble.BEDecoderLayer(
            config=cfg,
            ens_size=self.ens_size,
            random_sign_init=self.random_sign_init,
            relative_embedding=rel_emb,
            name=f'layers_{lyr}')(
                y,
                encoded,
                decoder_mask=decoder_mask,
                encoder_decoder_mask=encoder_decoder_mask,
                deterministic=deterministic,
                decode=decode,
                max_decode_length=max_decode_length)
      else:
        y = t5_network.DecoderLayer(
            config=cfg, relative_embedding=rel_emb, name=f'layers_{lyr}')(
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
    self.sow('intermediates', 'pre_logits', y)
    # [batch, length, emb_dim] -> [batch, length, vocab_size]
    # NOTE: In t5x.trainer, dropout_rng is not None when training and is
    # None when evaluating. In t5x.models.EncoderDecoderModel, when using
    # the transformer's apply method to get the logits, a flag
    # `enable_dropout` is provided. It is None if dropout_rng is None.
    # Finally, in t5_network.Transformer.decode method, `deterministic` is
    # set to `not enable_dropout`. So we can use this flag to decide if we
    # are in the train mode.
    # TODO(phandu): Consider to drop this treat and expose the `train`
    # argument to this call method, when the pipeline is in better shape.
    train = not deterministic

    # Using Gaussian process output layer.
    # This is the only place that T5-GP differs from deterministic T5.
    # TODO(phandu): Consider adding a new class field like
    # `store_random_features` for `return_random_features` argument
    # of RandomFeatureGaussianProcess.
    x_gp = self._apply_gp_layer(y, train=train)

    # Gaussian process layer output: a tuple of logits, covmat, and
    # optionally random features.
    self.sow('intermediates', 'logits', x_gp[0])
    covmat = x_gp[1]
    if covmat is not None:
      # The gp_layer considers all dimensions except the last one as
      # batch dimensions and returns a diagonal covariance matrix whose
      # size is the size (i.e. product) of those batch dimensions.
      # So we need to reshape here.
      covmat = covmat.reshape(x_gp[0].shape[:-1])
    self.sow('intermediates', 'covmat', covmat)

    if not train:
      # During inference, compute posterior mean by adjusting the original
      # logits with predictive uncertainty.
      logits = ed.nn.utils.mean_field_logits(
          logits=x_gp[0],
          covmat=covmat,
          mean_field_factor=self.mean_field_factor)
    else:
      logits = x_gp[0]

    return logits


class TransformerBEGp(t5_network.Transformer):
  """T5 Transformer with Gaussian Process output head."""
  config: t5_network.T5Config
  ens_size: int
  random_sign_init: float
  be_decoder_layers: Tuple[int, ...] = ()
  # Gaussian process parameters.
  use_gp_layer: bool = True
  covmat_momentum: float = -1.
  normalize_input: bool = True
  mean_field_factor: float = -1.
  ridge_penalty: float = 1.
  steps_per_epoch: Optional[int] = None

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
    self.decoder = BEGpDecoder(
        config=cfg,
        shared_embedding=self.shared_embedding,
        use_gp_layer=self.use_gp_layer,
        covmat_momentum=self.covmat_momentum,
        normalize_input=self.normalize_input,
        mean_field_factor=self.mean_field_factor,
        ridge_penalty=self.ridge_penalty,
        steps_per_epoch=self.steps_per_epoch,
        ens_size=self.ens_size,
        random_sign_init=self.random_sign_init,
        be_decoder_layers=self.be_decoder_layers)

  def decode(
      self,
      encoded,
      encoder_input_tokens,  # only needed for masks
      decoder_input_tokens,
      decoder_target_tokens,
      encoder_segment_ids=None,
      decoder_segment_ids=None,
      decoder_positions=None,
      enable_dropout=True,
      decode=False,
      max_decode_length=None):
    """Applies Transformer decoder-branch on encoded-input and target."""
    cfg = self.config
    if decoder_target_tokens.shape[0] > encoded.shape[0]:
      untile = int(decoder_target_tokens.shape[0] / self.ens_size)
      decoder_target_tokens = decoder_target_tokens[0:untile, ...]

    # Make padding attention masks.
    if decode:
      # Do not mask decoder attention based on targets padding at
      # decoding/inference time.
      decoder_mask = None
      encoder_decoder_mask = t5_layers.make_attention_mask(
          jnp.ones_like(decoder_target_tokens),
          encoder_input_tokens > 0,
          dtype=cfg.dtype)
    else:
      decoder_mask = t5_layers.make_decoder_mask(
          decoder_target_tokens=decoder_target_tokens,
          dtype=cfg.dtype,
          decoder_segment_ids=decoder_segment_ids)
      encoder_decoder_mask = t5_layers.make_attention_mask(
          decoder_target_tokens > 0, encoder_input_tokens > 0, dtype=cfg.dtype)

    # Add segmentation block-diagonal attention masks if using segmented data.
    if encoder_segment_ids is not None:
      if decode:
        raise ValueError(
            'During decoding, packing should not be used but '
            '`encoder_segment_ids` was passed to `Transformer.decode`.')

      encoder_decoder_mask = t5_layers.combine_masks(
          encoder_decoder_mask,
          t5_layers.make_attention_mask(
              decoder_segment_ids,
              encoder_segment_ids,
              jnp.equal,
              dtype=cfg.dtype))

    logits = self.decoder(
        encoded,
        decoder_input_tokens=decoder_input_tokens,
        decoder_positions=decoder_positions,
        decoder_mask=decoder_mask,
        encoder_decoder_mask=encoder_decoder_mask,
        deterministic=not enable_dropout,
        decode=decode,
        max_decode_length=max_decode_length)
    return logits

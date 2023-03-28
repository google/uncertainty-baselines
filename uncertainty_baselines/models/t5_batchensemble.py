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

"""T5 Transformer network using BatchEnsemble."""

import functools
import operator
from typing import Any, Callable, Optional, Sequence, Tuple, Union

import edward2.jax as ed
import flax.linen as nn
from flax.linen import partitioning as nn_partitioning
import jax.numpy as jnp
import t5x.examples.t5.layers as t5_layers
import t5x.examples.t5.network as t5_network

param_with_axes = nn_partitioning.param_with_axes
with_sharding_constraint = nn_partitioning.with_sharding_constraint

# Type annotations
Array = jnp.ndarray
DType = jnp.dtype
PRNGKey = jnp.ndarray
Shape = Sequence[int]
Activation = Callable[..., Array]
# Parameter initializers.
Initializer = Callable[[PRNGKey, Shape, DType], Array]


# This class follows edward2.jax.nn.DenseBatchEnsemble, except using names axes.
class DenseBatchEnsemble(nn.Module):
  """A BatchEnsemble Dense layer (no bias) with axes names.

    Attributes:
      features: tuple with numbers of output features.
      dtype: the dtype of the computation (default: float32).
      kernel_init: initializer function for the weight matrix.
  """
  ens_size: int
  features: int
  kernel_axes: Tuple[str, str]
  dtype: DType = jnp.float32
  kernel_init: Initializer = nn.initializers.lecun_normal()
  alpha_init: Initializer = nn.initializers.ones
  gamma_init: Initializer = nn.initializers.ones

  @nn.compact
  def __call__(self, inputs: Array) -> Array:
    """Applies a linear transformation to the inputs along multiple dimensions.

    Args:
      inputs: The nd-array to be transformed.

    Returns:
      The transformed input.
    """
    if len(self.kernel_axes) != 2:
      raise ValueError(f'Currently only support kernel axes with rank 2, '
                       f'instead got {len(self.kernel_axes)}.')
    dtype = self.dtype or inputs.dtype
    inputs = jnp.asarray(inputs, dtype)
    input_dim = inputs.shape[-1]

    kernel_shape = (input_dim, self.features)
    kernel = param_with_axes(
        'kernel', self.kernel_init, kernel_shape, dtype, axes=self.kernel_axes)
    # We use relpos_buckets for ensemble axes name here because currently rules
    # for `ensemble` are not registered.
    alpha = param_with_axes(
        'fast_weight_alpha',
        self.alpha_init, (self.ens_size, input_dim),
        dtype,
        axes=('relpos_buckets', self.kernel_axes[0]))
    gamma = param_with_axes(
        'fast_weight_gamma',
        self.gamma_init, (self.ens_size, self.features),
        dtype,
        axes=('relpos_buckets', self.kernel_axes[1]))

    kernel = jnp.asarray(kernel, self.dtype)
    inputs_shape = inputs.shape
    inputs = jnp.reshape(inputs, (self.ens_size, -1) + inputs_shape[1:])
    outputs = jnp.einsum('E...C,EC,CD,ED->E...D', inputs, alpha, kernel, gamma)
    return jnp.reshape(outputs, (outputs.shape[0] * outputs.shape[1],) +
                       outputs.shape[2:])


# This class follows t5_layers.MlpBlock, except using DenseBatchEnsemble layer.
class BEMlpBlock(nn.Module):
  """Transformer MLP / feed-forward block.

  Attributes:
    intermediate_dim: Shared dimension of hidden layers.
    activations: Type of activations for each layer.  Each element is either
      'linear', a string function name in flax.linen, or a function.
    kernel_init: Kernel function, passed to the dense layers.
    deterministic: Whether the dropout layers should be deterministic.
    intermediate_dropout_rate: Dropout rate used after the intermediate layers.
    dtype: Type for the dense layer.
  """
  ens_size: int
  random_sign_init: float
  intermediate_dim: int = 2048
  activations: Sequence[Union[str, Activation]] = ('relu',)
  kernel_init: Initializer = nn.initializers.xavier_uniform()
  intermediate_dropout_rate: float = 0.1
  dtype: Any = jnp.float32

  @nn.compact
  def __call__(self, inputs, decode: bool = False, deterministic: bool = False):
    """Applies Transformer MlpBlock module."""
    # Iterate over specified MLP input activation functions.
    # e.g. ('relu',) or ('gelu', 'linear') for gated-gelu.
    activations = []
    for idx, act_fn in enumerate(self.activations):
      dense_name = 'wi' if len(self.activations) == 1 else f'wi_{idx}'
      x = DenseBatchEnsemble(
          self.ens_size,
          self.intermediate_dim,
          kernel_axes=('embed', 'mlp'),
          dtype=self.dtype,
          kernel_init=self.kernel_init,
          alpha_init=ed.nn.utils.make_sign_initializer(self.random_sign_init),
          gamma_init=ed.nn.utils.make_sign_initializer(self.random_sign_init),
          name=dense_name)(
              inputs)
      x = t5_layers._convert_to_activation_function(act_fn)(x)
      activations.append(x)

    # Take elementwise product of above intermediate activations.
    x = functools.reduce(operator.mul, activations)
    # Apply dropout and final dense output projection.
    x = nn.Dropout(
        rate=self.intermediate_dropout_rate, broadcast_dims=(-2,))(
            x, deterministic=deterministic)  # Broadcast along length.
    x = with_sharding_constraint(x, ('batch', 'length', 'mlp'))
    output = DenseBatchEnsemble(
        self.ens_size,
        inputs.shape[-1],
        kernel_axes=('mlp', 'embed'),
        dtype=self.dtype,
        kernel_init=self.kernel_init,
        alpha_init=ed.nn.utils.make_sign_initializer(self.random_sign_init),
        gamma_init=ed.nn.utils.make_sign_initializer(self.random_sign_init),
        name='wo')(
            x)
    return output


# This class follows `t5_network.DecoderLayer` implementation, except using
# BEMlpBlock. In the future we could consider merge this into DecoderLayer.
class BEDecoderLayer(nn.Module):
  """Transformer decoder layer that attends to the encoder."""
  config: t5_network.T5Config
  ens_size: int
  random_sign_init: float
  relative_embedding: nn.Module

  @nn.compact
  def __call__(self,
               inputs,
               encoded,
               decoder_mask=None,
               encoder_decoder_mask=None,
               deterministic=False,
               decode=False,
               max_decode_length=None):
    cfg = self.config

    # Relative position embedding as attention biases.
    l = max_decode_length if decode and max_decode_length else inputs.shape[-2]
    decoder_bias = self.relative_embedding(l, l, False)

    # inputs: embedded inputs to the decoder with shape [batch, length, emb_dim]
    x = t5_layers.LayerNorm(
        dtype=cfg.dtype, name='pre_self_attention_layer_norm')(
            inputs)

    # Self-attention block
    x = t5_layers.MultiHeadDotProductAttention(
        num_heads=cfg.num_heads,
        dtype=cfg.dtype,
        head_dim=cfg.head_dim,
        dropout_rate=cfg.dropout_rate,
        float32_logits=cfg.float32_attention_logits,
        name='self_attention')(
            x,
            x,
            decoder_mask,
            decoder_bias,
            deterministic=deterministic,
            decode=decode)
    x = nn.Dropout(
        rate=cfg.dropout_rate, broadcast_dims=(-2,))(
            x, deterministic=deterministic)
    x = x + inputs

    # Encoder-Decoder block.
    y = t5_layers.LayerNorm(
        dtype=cfg.dtype, name='pre_cross_attention_layer_norm')(
            x)
    y = t5_layers.MultiHeadDotProductAttention(
        num_heads=cfg.num_heads,
        dtype=cfg.dtype,
        head_dim=cfg.head_dim,
        dropout_rate=cfg.dropout_rate,
        float32_logits=cfg.float32_attention_logits,
        name='encoder_decoder_attention')(
            y, encoded, encoder_decoder_mask, deterministic=deterministic)
    y = nn.Dropout(
        rate=cfg.dropout_rate, broadcast_dims=(-2,))(
            y, deterministic=deterministic)
    y = y + x

    # MLP block.
    z = t5_layers.LayerNorm(dtype=cfg.dtype, name='pre_mlp_layer_norm')(y)

    # Tile inputs for BatchEnsemble layers.
    y = jnp.tile(y, [self.ens_size] + [1] * (y.ndim - 1))
    z = jnp.tile(z, [self.ens_size] + [1] * (z.ndim - 1))

    z = BEMlpBlock(
        ens_size=self.ens_size,
        random_sign_init=self.random_sign_init,
        intermediate_dim=cfg.mlp_dim,
        activations=cfg.mlp_activations,
        intermediate_dropout_rate=cfg.dropout_rate,
        dtype=cfg.dtype,
        name='mlp',
    )(z, deterministic=deterministic)
    z = nn.Dropout(
        rate=cfg.dropout_rate, broadcast_dims=(-2,))(
            z, deterministic=deterministic)
    z = z + y

    return z


# This class follows `t5_network.Decoder` implementation.
# Consider merging this into t5_network.
class BatchEnsembleDecoder(nn.Module):
  """T5 Decoder with BatchEnsemble layers."""
  config: t5_network.T5Config
  ens_size: int
  random_sign_init: float
  shared_embedding: nn.Module
  be_decoder_layers: Optional[Tuple[int, ...]] = None
  head_kernel_init: Initializer = nn.initializers.zeros

  @nn.compact
  def __call__(self,
               encoded,
               decoder_input_tokens,
               decoder_positions=None,
               decoder_mask=None,
               encoder_decoder_mask=None,
               deterministic=False,
               decode=False,
               max_decode_length=None):
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
    be_layer_started = False
    be_decoder_layers = [
        x if x >= 0 else (x + cfg.num_decoder_layers)
        for x in list(self.be_decoder_layers)
        if x is not None
    ]
    for lyr in range(cfg.num_decoder_layers):
      # [batch, length, emb_dim] -> [batch, length, emb_dim]
      if lyr in list(be_decoder_layers):
        if lyr != cfg.num_decoder_layers - 1:
          raise ValueError('Currently only support lastdecoder layer to be BE.')

        be_layer_started = True
        y = BEDecoderLayer(
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

    # [batch, length, emb_dim] -> [batch, length, vocab_size]
    if cfg.logits_via_embedding:
      # Use the transpose of embedding matrix for logit transform.
      logits = self.shared_embedding.attend(y)
      # Correctly normalize pre-softmax logits for this shared case.
      logits = logits / jnp.sqrt(y.shape[-1])
    else:
      if not be_layer_started:
        # Tile inputs for BatchEnsemble layers.
        y = jnp.tile(y, [self.ens_size] + [1] * (y.ndim - 1))
      logits = DenseBatchEnsemble(
          self.ens_size,
          cfg.vocab_size,
          kernel_axes=('embed', 'vocab'),
          kernel_init=self.head_kernel_init,
          alpha_init=ed.nn.utils.make_sign_initializer(self.random_sign_init),
          gamma_init=ed.nn.utils.make_sign_initializer(self.random_sign_init),
          dtype=jnp.float32,  # Use float32 for stabiliity.
          name='logits_dense')(
              y)
    return logits


class TransformerBE(t5_network.Transformer):
  """T5 Transformer with BatchEnsemble Decoder."""
  config: t5_network.T5Config
  ens_size: int
  random_sign_init: float
  be_decoder_layers: Tuple[int, ...] = ()

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
    self.decoder = BatchEnsembleDecoder(
        config=cfg,
        ens_size=self.ens_size,
        random_sign_init=self.random_sign_init,
        shared_embedding=self.shared_embedding,
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

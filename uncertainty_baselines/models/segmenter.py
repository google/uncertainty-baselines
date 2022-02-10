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

"""Segmenter Vision Transformer (ViT) model.

Based on scenic library implementation.
"""
from typing import Any, Callable, Optional, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
import ml_collections

Array = Any
PRNGKey = Any
Shape = Tuple[int]
Dtype = Any


class IdentityLayer(nn.Module):
  """Identity layer, convenient for giving a name to an array."""

  @nn.compact
  def __call__(self, x):
    return x


class AddPositionEmbs(nn.Module):
  """Adds (optionally learned) positional embeddings to the inputs.

  Attributes:
    posemb_init: positional embedding initializer.
  """
  # TODO(kellybuchanan): check initialization,
  # nn.initializers.normal(stddev=0.02) from BERT.
  posemb_init: Callable[[PRNGKey, Shape, Dtype], Array]

  @nn.compact
  def __call__(self, inputs):
    """Applies AddPositionEmbs module.

    By default this layer uses a fixed sinusoidal embedding table. If a
    learned position embedding is desired, pass an initializer to
    posemb_init.

    Args:
      inputs: Inputs to the layer.

    Returns:
      Output tensor with shape `(bs, timesteps, in_dim)`.
    """
    # inputs.shape is (batch_size, seq_len, emb_dim).
    assert inputs.ndim == 3, ('Number of dimensions should be 3,'
                              ' but it is: %d' % inputs.ndim)
    pos_emb_shape = (1, inputs.shape[1], inputs.shape[2])
    pe = self.param('pos_embedding', self.posemb_init, pos_emb_shape)
    return inputs + pe


class MlpBlock(nn.Module):
  """Transformer MLP / feed-forward block."""

  mlp_dim: int
  dtype: Dtype = jnp.float32
  out_dim: Optional[int] = None
  dropout_rate: float = 0.1
  kernel_init: Callable[[PRNGKey, Shape, Dtype],
                        Array] = nn.initializers.xavier_uniform()
  bias_init: Callable[[PRNGKey, Shape, Dtype],
                      Array] = nn.initializers.normal(stddev=1e-6)
  use_bias: bool = True
  precision: Optional[jax.lax.Precision] = None
  activation_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.gelu

  @nn.compact
  def __call__(self, inputs, *, deterministic):
    """Applies Transformer MlpBlock module."""
    actual_out_dim = inputs.shape[-1] if self.out_dim is None else self.out_dim
    x = nn.Dense(
        features=self.mlp_dim,
        dtype=self.dtype,
        use_bias=self.use_bias,
        kernel_init=self.kernel_init,
        bias_init=self.bias_init,
        precision=self.precision)(  # pytype: disable=wrong-arg-types
            inputs)
    x = self.activation_fn(x)
    x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=deterministic)
    output = nn.Dense(
        features=actual_out_dim,
        dtype=self.dtype,
        kernel_init=self.kernel_init,
        bias_init=self.bias_init,
        use_bias=self.use_bias,
        precision=self.precision)(  # pytype: disable=wrong-arg-types
            x)
    output = nn.Dropout(
        rate=self.dropout_rate)(
            output, deterministic=deterministic)
    return output


class Encoder1DBlock(nn.Module):
  """Transformer encoder layer.

  Attributes:
    inputs: input data.
    mlp_dim: dimension of the mlp on top of attention block.
    dtype: the dtype of the computation (default: float32).
    dropout_rate: dropout rate.
    attention_dropout_rate: dropout for attention heads.
    deterministic: bool, deterministic or not (to apply dropout).
    num_heads: Number of heads in nn.MultiHeadDotProductAttention
    stochastic_depth: probability of dropping a layer linearly grows from 0 to
      the provided value.

  """

  mlp_dim: int
  num_heads: int
  dtype: Dtype = jnp.float32
  dropout_rate: float = 0.1
  attention_dropout_rate: float = 0.1
  stochastic_depth: float = 0.0

  def get_stochastic_depth_mask(self, x: jnp.ndarray,
                                deterministic: bool) -> jnp.ndarray:
    """Generate the stochastic depth mask in order to apply layer-drop.

    Args:
      x: Input tensor.
      deterministic: Weather we are in the deterministic mode (e.g inference
        time) or not.

    Returns:
      Stochastic depth mask.
    """
    if not deterministic and self.stochastic_depth:
      shape = (x.shape[0],) + (1,) * (x.ndim - 1)
      return jax.random.bernoulli(
          self.make_rng('dropout'), self.stochastic_depth, shape)
    else:
      return 0.0

  @nn.compact
  def __call__(self, inputs, *, deterministic):
    """Applies Encoder1DBlock module.

    Args:
      inputs: Inputs to the layer.
      deterministic: Dropout will not be applied when set to true.

    Returns:
      output after transformer encoder block.
    """

    # Attention block.
    assert inputs.ndim == 3, f'Expected (batch, seq, hidden) got {inputs.shape}'
    x = nn.LayerNorm(dtype=self.dtype, name='LayerNorm_0')(inputs)
    x = nn.MultiHeadDotProductAttention(
        dtype=self.dtype,
        kernel_init=nn.initializers.xavier_uniform(),
        broadcast_dropout=False,
        deterministic=deterministic,
        dropout_rate=self.attention_dropout_rate,
        num_heads=self.num_heads,
        name='MultiHeadDotProductAttention_1')(x, x)
    x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=deterministic)
    x = x * (1.0 - self.get_stochastic_depth_mask(x, deterministic)) + inputs

    # MLP block.
    y = nn.LayerNorm(dtype=self.dtype, name='LayerNorm_2')(x)
    y = MlpBlock(
        mlp_dim=self.mlp_dim,
        dtype=self.dtype,
        name='MlpBlock_3',
        dropout_rate=self.dropout_rate,
        activation_fn=nn.gelu,
        kernel_init=nn.initializers.xavier_uniform(),
        bias_init=nn.initializers.normal(stddev=1e-6))(
            y, deterministic=deterministic)

    return y * (1.0 - self.get_stochastic_depth_mask(x, deterministic)) + x


class Encoder(nn.Module):
  """Transformer Model Encoder for sequence to sequence translation.

  Attributes:
    num_layers: number of layers
    mlp_dim: dimension of the mlp on top of attention block
    num_heads: Number of heads in nn.MultiHeadDotProductAttention
    dropout_rate: dropout rate.
    attention_dropout_rate: dropout rate in self attention.
    stochastic_depth: probability of dropping a layer linearly grows from 0 to
    the provided value. Our implementation of stochastic depth follows timm
    library, which does per-example layer dropping and uses independent
    dropping patterns for each skip-connection.
    dtype: Dtype of activations.
  """

  num_layers: int
  mlp_dim: int
  num_heads: int
  dropout_rate: float = 0.1
  attention_dropout_rate: float = 0.1
  stochastic_depth: float = 0.0
  dtype: Any = jnp.float32

  @nn.compact
  def __call__(self, inputs: jnp.ndarray, *, train: bool = False):
    """Applies Transformer model on the inputs.

    Args:
      inputs: Inputs to the layer.
      train: Set to `True` when training.

    Returns:
      output of a transformer encoder.
    """
    assert inputs.ndim == 3  # (batch, len, emb)
    x = AddPositionEmbs(
        posemb_init=nn.initializers.normal(stddev=0.02),  # from BERT.
        name='posembed_input')(
            inputs)
    x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not train)

    # Input Encoder
    for lyr in range(self.num_layers):
      x = Encoder1DBlock(
          mlp_dim=self.mlp_dim,
          dropout_rate=self.dropout_rate,
          attention_dropout_rate=self.attention_dropout_rate,
          stochastic_depth=(lyr / max(self.num_layers - 1, 1)) *
          self.stochastic_depth,
          name=f'encoderblock_{lyr}',
          num_heads=self.num_heads)(
              x, deterministic=not train)
    encoded = nn.LayerNorm(name='encoder_norm')(x)

    return encoded


class ViTBackbone(nn.Module):
  """Vision Transformer model backbone (everything except the head).

  Edited from VisionTransformer.
  """

  mlp_dim: int
  num_layers: int
  num_heads: int
  patches: ml_collections.ConfigDict
  hidden_size: int
  dropout_rate: float = 0.1
  attention_dropout_rate: float = 0.1
  classifier: str = 'gap'

  @nn.compact
  def __call__(self, inputs, *, train: bool):
    out = {}

    x = inputs
    n, h, w, c = x.shape

    # We can merge s2d+emb into a single conv; it's the same.
    x = nn.Conv(
        features=self.hidden_size,
        kernel_size=self.patches.size,
        strides=self.patches.size,
        padding='VALID',
        name='embedding')(
            x)

    # Here, x is a grid of embeddings.
    # TODO(dusenberrymw): Switch to self.sow(.).
    out['stem'] = x

    # Transformer.
    n, h, w, c = x.shape
    x = jnp.reshape(x, [n, h * w, c])

    # If we want to add a class token, add it here.
    if self.classifier == 'token':
      cls = self.param('cls', nn.initializers.zeros, (1, 1, c))
      cls = jnp.tile(cls, [n, 1, 1])
      x = jnp.concatenate([cls, x], axis=1)

    x = Encoder(name='Transformer',
                mlp_dim=self.mlp_dim,
                num_layers=self.num_layers,
                num_heads=self.num_heads,
                dropout_rate=self.dropout_rate,
                attention_dropout_rate=self.attention_dropout_rate,
                )(x, train=train)

    out['transformed'] = x

    return x, out


class SegVit(nn.Module):
  """Segmentation model with ViT backbone and decoder."""

  num_classes: int
  patches: ml_collections.ConfigDict
  backbone_configs: ml_collections.ConfigDict
  decoder_configs: ml_collections.ConfigDict

  @nn.compact
  def __call__(self, x: jnp.ndarray, *, train: bool, debug: bool = False):
    input_shape = x.shape
    b, h, w, _ = input_shape

    fh, fw = self.patches.size
    gh, gw = h // fh, w // fw

    if self.backbone_configs.type == 'vit':
      x, out = ViTBackbone(
          mlp_dim=self.backbone_configs.mlp_dim,
          num_layers=self.backbone_configs.num_layers,
          num_heads=self.backbone_configs.num_heads,
          patches=self.patches,
          hidden_size=self.backbone_configs.hidden_size,
          dropout_rate=self.backbone_configs.dropout_rate,
          attention_dropout_rate=self.backbone_configs.attention_dropout_rate,
          classifier=self.backbone_configs.classifier,
          name='backbone')(
              x, train=train)
    else:
      raise ValueError(f'Unknown backbone: {self.backbone_configs.type}.')

    output_projection = nn.Dense(
        self.num_classes,
        kernel_init=nn.initializers.zeros,
        name='output_projection')

    if self.decoder_configs.type == 'linear':
      # Linear head only, like Segmenter baseline:
      # https://arxiv.org/abs/2105.05633
      x = jnp.reshape(x, [b, gh, gw, -1])
      x = output_projection(x)
      # Resize bilinearly:
      x = jax.image.resize(x, [b, h, w, x.shape[-1]], 'linear')
      out['logits'] = x
    else:
      raise ValueError(
          f'Decoder type {self.decoder_configs.type} is not defined.')

    assert input_shape[:-1] == x.shape[:-1], (
        'Input and output shapes do not match: %d vs. %d.', input_shape[:-1],
        x.shape[:-1])

    return x, out


def segmenter_transformer(num_classes: int,
                          patches: Any,
                          backbone_configs: Any,
                          decoder_configs: Any
                          ):
  """Builds a Vision Transformer (ViT) model."""
  # TODO(dusenberrymw): Add API docs once config dict in VisionTransformer is
  # cleaned up.
  return SegVit(
      num_classes=num_classes,
      patches=patches,
      backbone_configs=backbone_configs,
      decoder_configs=decoder_configs,
    )

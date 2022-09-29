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
from typing import Any, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
import ml_collections
from uncertainty_baselines.models import vit

Array = Any
PRNGKey = Any
Shape = Tuple[int]
Dtype = Any


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
  def __call__(self, inputs: Array, *, train: bool):
    """Applies the module."""
    out = {}

    x = inputs
    n, h, w, c = x.shape

    # PatchEmbedding
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

    # Transformer blocks
    x = vit.Encoder(
        name='Transformer',
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
  def __call__(self, x: Array, *, train: bool, debug: bool = False):
    """Applies the module."""
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

    # remove CLS tokens for decoding
    if self.backbone_configs.classifier == 'token':
      x = x[..., 1:, :]

    output_projection = nn.Dense(
        self.num_classes,
        kernel_init=nn.initializers.variance_scaling(0.02, 'fan_in',
                                                     'truncated_normal'),
        name='output_projection')

    if self.decoder_configs.type == 'linear':
      # Linear head only, like Segmenter baseline:
      # https://arxiv.org/abs/2105.05633
      x = jnp.reshape(x, [b, gh, gw, -1])
      x = output_projection(x)
      # Resize bilinearly:
      x = jax.image.resize(x, [b, h, w, x.shape[-1]], 'bilinear')

      out['logits'] = x
    else:
      raise ValueError(
          f'Decoder type {self.decoder_configs.type} is not defined.')

    assert input_shape[:-1] == x.shape[:-1], (
        'Input and output shapes do not match: %d vs. %d.', input_shape[:-1],
        x.shape[:-1])

    return x, out

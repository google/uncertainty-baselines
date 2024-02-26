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

"""Segmenter Vision Transformer (ViT) model.

Based on scenic library implementation.
"""
from typing import Any, Callable, Optional, Tuple, Sequence, Iterable

import edward2.jax as ed
import flax.linen as nn
import jax
import jax.numpy as jnp
import ml_collections
from uncertainty_baselines.models import segmenter
from uncertainty_baselines.models import vit_batchensemble

Array = Any
PRNGKey = Any
Shape = Tuple[int]
DType = type(jnp.float32)

InitializeFn = Callable[[jnp.ndarray, Iterable[int], DType], jnp.ndarray]


class ViTBackboneBE(nn.Module):
  """Vision Transformer model backbone (everything except the head).

  Edited from VisionTransformer.
  """

  mlp_dim: int
  num_layers: int
  num_heads: int
  patches: ml_collections.ConfigDict
  hidden_size: int
  ens_size: int
  random_sign_init: float
  be_layers: Optional[Sequence[int]] = None
  dropout_rate: float = 0.1
  attention_dropout_rate: float = 0.1
  classifier: str = 'gap'

  @nn.compact
  def __call__(self, inputs: Array, *, train: bool):
    """Applies the module."""
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

    x, extra_info = vit_batchensemble.BatchEnsembleEncoder(
        name='Transformer',
        mlp_dim=self.mlp_dim,
        num_layers=self.num_layers,
        num_heads=self.num_heads,
        dropout_rate=self.dropout_rate,
        attention_dropout_rate=self.attention_dropout_rate,
        ens_size=self.ens_size,
        random_sign_init=self.random_sign_init,
        be_layers=self.be_layers,
        train=train,
    )(
        x)

    out.update(extra_info)
    out['transformed'] = x

    return x, out


class SegVitBE(nn.Module):
  """Segmentation model with ViT backbone and decoder."""

  num_classes: int
  patches: ml_collections.ConfigDict
  backbone_configs: ml_collections.ConfigDict
  decoder_configs: ml_collections.ConfigDict
  head_kernel_init: InitializeFn = nn.initializers.variance_scaling(  # pytype: disable=annotation-type-mismatch  # jax-types
      0.02, 'fan_in', 'truncated_normal')

  @nn.compact
  def __call__(self, x: Array, *, train: bool, debug: bool = False):
    """Applies the module."""
    input_shape = x.shape
    b, h, w, _ = input_shape

    fh, fw = self.patches.size
    gh, gw = h // fh, w // fw

    if self.backbone_configs.type == 'vit' and self.decoder_configs.type == 'linear':
      assert self.backbone_configs.ens_size == 1

    if self.backbone_configs.type == 'vit' and self.decoder_configs.type == 'linear_be':
      raise NotImplementedError(
          'Configuration with encoder {} and decoder {} is not implemented'
          .format(
              self.backbone_configs.type,
              self.decoder_configs.type,
          ))

    if self.backbone_configs.type == 'vit':
      x, out = segmenter.ViTBackbone(
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
    elif self.backbone_configs.type == 'vit_be':
      x, out = ViTBackboneBE(
          mlp_dim=self.backbone_configs.mlp_dim,
          num_layers=self.backbone_configs.num_layers,
          num_heads=self.backbone_configs.num_heads,
          patches=self.patches,
          hidden_size=self.backbone_configs.hidden_size,
          dropout_rate=self.backbone_configs.dropout_rate,
          attention_dropout_rate=self.backbone_configs.attention_dropout_rate,
          classifier=self.backbone_configs.classifier,
          ens_size=self.backbone_configs.ens_size,
          random_sign_init=self.backbone_configs.random_sign_init,
          be_layers=self.backbone_configs.be_layers,
          name='backbone')(
              x, train=train)
    else:
      raise ValueError(f'Unknown backbone: {self.backbone_configs.type}.')

    # remove CLS tokens for decoding
    if self.backbone_configs.classifier == 'token':
      x = x[..., 1:, :]

    if self.decoder_configs.type == 'linear':
      output_projection = nn.Dense(
          self.num_classes,
          kernel_init=self.head_kernel_init,
          name='output_projection')
    elif self.decoder_configs.type == 'linear_be':
      output_projection = ed.nn.DenseBatchEnsemble(
          self.num_classes,
          self.backbone_configs.ens_size,
          activation=None,
          alpha_init=ed.nn.utils.make_sign_initializer(
              self.backbone_configs.get('random_sign_init')),
          gamma_init=ed.nn.utils.make_sign_initializer(
              self.backbone_configs.get('random_sign_init')),
          kernel_init=self.head_kernel_init,
          name='output_projection_be')
    else:
      raise ValueError(
          f'Decoder type {self.decoder_configs.type} is not defined.')

    ens_size = self.backbone_configs.get('ens_size')

    # Linear head only, like Segmenter baseline:
    # https://arxiv.org/abs/2105.05633
    x = jnp.reshape(x, [b * ens_size, gh, gw, -1])
    x = output_projection(x)

    # Resize bilinearly:
    x = jax.image.resize(x, [b * ens_size, h, w, x.shape[-1]], 'bilinear')
    out['logits'] = x

    new_input_shape = tuple([
        input_shape[0] * ens_size,
    ] + list(input_shape[1:-1]))
    assert new_input_shape == x.shape[:-1], (
        'BE Input and output shapes do not match: %d vs. %d.', new_input_shape,
        x.shape[:-1])

    return x, out

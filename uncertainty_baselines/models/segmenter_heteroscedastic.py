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

"""Segmenter Heteroscedastic Vision Transformer (ViT) model.

Based on scenic library implementation.
"""
from typing import Any, Callable, Tuple, Iterable

import edward2.jax as ed
import flax.linen as nn
import jax
import jax.numpy as jnp
import ml_collections
from uncertainty_baselines.models import segmenter
from uncertainty_baselines.models import segmenter_be

Array = Any
PRNGKey = Any
Shape = Tuple[int]
DType = type(jnp.float32)

InitializeFn = Callable[[jnp.ndarray, Iterable[int], DType], jnp.ndarray]


class SegVitHet(nn.Module):
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
      x, out = segmenter_be.ViTBackboneBE(
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

    ens_size = self.backbone_configs.get('ens_size', 1)

    if self.decoder_configs.type == 'linear':
     # Linear head only, like Segmenter baseline:
     # https://arxiv.org/abs/2105.05633
      output_projection = nn.Dense(
          self.num_classes,
          kernel_init=self.head_kernel_init,
          name='output_projection')

      x = jnp.reshape(x, [b * ens_size, gh, gw, -1])
      x = output_projection(x)

    elif self.decoder_configs.type == 'gp':
      # Gaussian process layer output: (logits, covmat, and *random features)
      # *random features are optional
      output_projection = ed.nn.RandomFeatureGaussianProcess(
          features=self.num_classes,
          name='output_projection',
          **self.decoder_configs.gp_layer)

      x = jnp.reshape(x, [b*ens_size*gh*gw, -1])

      x_gp = output_projection(x)
      out['logits_gp'] = x_gp[0]
      out['covmat_gp'] = x_gp[1]

      if len(x_gp) > 2:
        out['random_features_gp'] = x_gp[2]

      if not train:
        # During inference, compute posterior mean by adjusting the original
        # logits with predictive uncertainty.
        x = ed.nn.utils.mean_field_logits(
            logits=x_gp[0],
            covmat=x_gp[1],
            mean_field_factor=self.decoder_configs.mean_field_factor)
      else:
        x = x_gp[0]

      x = jnp.reshape(x, [b*ens_size, gh, gw, -1])

    elif self.decoder_configs.type == 'het':
      output_projection = ed.nn.MCSoftmaxDenseFA(
          self.num_classes,
          self.decoder_configs.num_factors,
          self.decoder_configs.temperature,
          self.decoder_configs.param_efficient,
          self.decoder_configs.mc_samples,
          self.decoder_configs.mc_samples,
          logits_only=True,
          return_locs=self.decoder_configs.return_locs,
          name='output_projection')

      x = jnp.reshape(x, [b*ens_size*gh*gw, -1])
      x_heter = output_projection(x)

      out['logits_het'] = x_heter

      x = jnp.reshape(x_heter, [b*ens_size, gh, gw, -1])

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

      x = output_projection(x)

    else:
      raise ValueError(
          f'Decoder type {self.decoder_configs.type} is not defined.')

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

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

"""Random-feature Gaussian process model with vision transformer (ViT) backbone."""
import dataclasses

from typing import Any, Mapping, Tuple, Optional

import edward2.jax as ed
import flax.linen as nn
import jax.numpy as jnp

import ml_collections
import uncertainty_baselines.models.vit as vit

# Jax data types.
Array = Any
ConfigDict = ml_collections.ConfigDict
Dtype = Any
PRNGKey = Any
Shape = Tuple[int]

# Default field value for kwargs, to be used for data class declaration.
default_kwarg_dict = lambda: dataclasses.field(default_factory=dict)


class VisionTransformerGaussianProcess(nn.Module):
  """Vision Transformer model with Gaussian process last layer."""
  num_classes: int
  patches: Any
  transformer: Any
  hidden_size: int
  representation_size: Optional[int] = None
  classifier: str = 'token'
  # Gaussian process parameters.
  use_gp_layer: bool = True
  gp_layer_kwargs: Mapping[str, Any] = default_kwarg_dict()

  def setup(self):
    # pylint:disable=not-a-mapping
    if self.use_gp_layer:
      self.gp_layer = ed.nn.RandomFeatureGaussianProcess(
          features=self.num_classes, name='head', **self.gp_layer_kwargs)
    # pylint:enable=not-a-mapping

  @nn.compact
  def __call__(self,
               inputs: Array,
               train: bool,
               mean_field_factor: float = -1.,
               **gp_kwargs) -> Tuple[Array, Mapping[str, Any]]:
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

    x = vit.Encoder(name='Transformer', **self.transformer)(x, train=train)
    out['transformed'] = x

    if self.classifier == 'token':
      x = x[:, 0]
    elif self.classifier == 'gap':
      x = jnp.mean(x, axis=list(range(1, x.ndim - 1)))  # (1,) or (1,2)
    else:
      raise ValueError(f'Invalid classifier={self.classifier}')

    out['head_input'] = x

    if self.representation_size is not None:
      x = nn.Dense(features=self.representation_size, name='pre_logits')(x)
      out['pre_logits'] = x
      x = nn.tanh(x)
    else:
      x = vit.IdentityLayer(name='pre_logits')(x)
      out['pre_logits'] = x

    if not self.use_gp_layer:
      logits = nn.Dense(
          features=self.num_classes,
          name='head',
          kernel_init=nn.initializers.zeros)(
              x)
      out['logits'] = logits
    else:
      # Using Gaussian process output layer.
      # This is the only place that ViT-GP differs from determinisitc ViT.
      x_gp = self.gp_layer(x, **gp_kwargs)

      # Gaussian process layer output: a tuple of logits, covmat, and optionally
      # random features.
      out['logits'] = x_gp[0]
      out['covmat'] = x_gp[1]
      if len(x_gp) > 2:
        out['random_features'] = x_gp[2]

      if not train:
        # During inference, compute posterior mean by adjusting the original
        # logits with predictive uncertainty.
        logits = ed.nn.utils.mean_field_logits(
            logits=x_gp[0], covmat=x_gp[1], mean_field_factor=mean_field_factor)
      else:
        logits = x_gp[0]

    return logits, out


def vision_transformer_gp(num_classes: int, use_gp_layer: bool,
                          vit_kwargs: ConfigDict,
                          gp_layer_kwargs: Mapping[str, Any]):
  """Builds a Vision Transformer Gaussian process (ViT-GP) model."""
  return VisionTransformerGaussianProcess(
      num_classes=num_classes,
      use_gp_layer=use_gp_layer,
      gp_layer_kwargs=gp_layer_kwargs,
      **vit_kwargs)

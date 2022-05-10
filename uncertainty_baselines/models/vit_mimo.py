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

"""MIMO Vision Transformer model."""
from typing import Any, Optional, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
from uncertainty_baselines.models import vit

Array = Any
PRNGKey = Any
Shape = Tuple[int]
Dtype = Any


class VisionTransformerMIMO(nn.Module):
  """MIMO Vision Transformer model."""

  num_classes: int
  patches: Any
  transformer: Any
  hidden_size: int
  ensemble_size: int
  representation_size: Optional[int] = None
  classifier: str = 'token'
  fix_base_model: bool = False

  @nn.compact
  def __call__(self, inputs, *, train):
    """Function of shapes [B*R,h,w,c*E] -> [E*B*R,num_classes]."""
    out = {}

    x = inputs

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

    # TODO(markcollier): Fix base model without using stop_gradient.
    if self.fix_base_model:
      x = jax.lax.stop_gradient(x)

    # Shape: (batch_size, num_classes * ensemble_size).
    x = nn.Dense(self.num_classes * self.ensemble_size,
                 name='head',
                 kernel_init=nn.initializers.zeros)(x)
    # Shape: (batch_size * ensemble_size, num_classes).
    x = jnp.concatenate(jnp.split(x, self.ensemble_size, axis=-1))
    out['logits'] = x
    return x, out


def vision_transformer_mimo(num_classes: int,
                            patches: Any,
                            transformer: Any,
                            hidden_size: int,
                            ensemble_size: int,
                            representation_size: Optional[int] = None,
                            classifier: str = 'token',
                            fix_base_model: bool = False):
  """Builds a BatchEnsemble Vision Transformer (ViT) model."""
  # TODO(dusenberrymw): Add API docs once the config dict in VisionTransformerBE
  # is cleaned up.
  return VisionTransformerMIMO(
      num_classes=num_classes,
      patches=patches,
      transformer=transformer,
      hidden_size=hidden_size,
      ensemble_size=ensemble_size,
      representation_size=representation_size,
      classifier=classifier,
      fix_base_model=fix_base_model)

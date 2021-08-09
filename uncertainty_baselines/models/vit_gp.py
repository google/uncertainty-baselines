# coding=utf-8
# Copyright 2021 The Uncertainty Baselines Authors.
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
from typing import Any, Mapping, Tuple

import edward2.jax as ed
import flax.linen as nn

import uncertainty_baselines.models.vit as vit

# Jax data types.
Array = Any
PRNGKey = Any
Shape = Tuple[int]
Dtype = Any

# Default field value for kwargs, to be used for data class declaration.
default_kwarg_dict = lambda: dataclasses.field(default_factory=dict)


class VisionTransformerGaussianProcess(nn.Module):
  """VisionTransformer with Gaussian process output head."""
  num_classes: int
  use_gp_layer: bool = True
  vit_kwargs: Mapping[str, Any] = default_kwarg_dict()
  gp_layer_kwargs: Mapping[str, Any] = default_kwarg_dict()

  def setup(self):
    # pylint:disable=not-a-mapping
    self.vit_backbone = vit.VisionTransformer(
        num_classes=self.num_classes, **self.vit_kwargs)

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
    x_vit, out = self.vit_backbone(inputs=inputs, train=train)
    if not self.use_gp_layer:
      return x_vit, out

    # Extracts head input from parent class.
    x = out['head_input']

    if self.vit_kwargs['representation_size'] is not None:
      # TODO(jereliu): Replace Dense with spectral normalization.
      x = nn.Dense(features=self.vit_kwargs['representation_size'],
                   name='pre_logits')(x)
      out['pre_logits'] = x
      x = nn.tanh(x)
    else:
      x = vit.IdentityLayer(name='pre_logits')(x)
      out['pre_logits'] = x

    # Makes output: a tuple of logits, covmat, and optionally random features.
    x_gp = self.gp_layer(x, **gp_kwargs)

    out['logits'] = x_gp[0]
    out['covmat'] = x_gp[1]
    if len(x_gp) > 2:
      out['random_features'] = x_gp[2]

    if not train:
      # During inference, compute posterior mean by adjusting posterior-mode
      # logits prediction with predictive uncertainty.
      logits = ed.nn.utils.mean_field_logits(
          logits=x_gp[0], covmat=x_gp[1], mean_field_factor=mean_field_factor)
    else:
      logits = x_gp[0]

    return logits, out


def vision_transformer_gp(num_classes: int, use_gp_layer: bool,
                          vit_kwargs: Mapping[str, Any],
                          gp_layer_kwargs: Mapping[str, Any]):
  """Builds a Vision Transformer Gaussian process (ViT-GP) model."""
  return VisionTransformerGaussianProcess(
      num_classes=num_classes,
      use_gp_layer=use_gp_layer,
      vit_kwargs=vit_kwargs,
      gp_layer_kwargs=gp_layer_kwargs)

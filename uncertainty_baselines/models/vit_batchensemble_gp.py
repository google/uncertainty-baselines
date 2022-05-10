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

"""Patch Transformerm similar to Gshard paper with BatchEnsemble MLPs."""
import dataclasses

from typing import Any, Callable, Iterable, Mapping, Optional, Tuple

import edward2.jax as ed
import flax.linen as nn
import jax.numpy as jnp

from uncertainty_baselines.models import vit
from uncertainty_baselines.models import vit_batchensemble

DType = type(jnp.float32)
InitializeFn = Callable[[jnp.ndarray, Iterable[int], DType], jnp.ndarray]
Params = Mapping[str, Any]

default_kwarg_dict = lambda: dataclasses.field(default_factory=dict)


class VisionTransformerBEGP(nn.Module):
  """BatchEnsemble Vision Transformer with Gaussian process last layer.

  You must specify either the vertical and horizontal resolution of the patches
  (patch_size), or the number of vertical and horizontal divisions of the input
  image (patch_grid).
  """
  num_classes: int
  transformer: Params
  hidden_size: int
  patch_size: Optional[Tuple[int, int]] = None
  patch_grid: Optional[Tuple[int, int]] = None
  representation_size: Optional[int] = None
  classifier: str = "token"
  head_kernel_init: InitializeFn = nn.initializers.zeros
  use_gp_layer: bool = True
  gp_layer_kwargs: Mapping[str, Any] = default_kwarg_dict()
  train: Optional[bool] = None

  def setup(self):
    # pylint:disable=not-a-mapping
    if self.use_gp_layer:
      self.gp_layer = ed.nn.RandomFeatureGaussianProcess(
          features=self.num_classes, name="head", **self.gp_layer_kwargs)
    # pylint:enable=not-a-mapping

  def patches(self,
              images: jnp.ndarray,
              hidden_size: int,
              patch_size: Optional[Tuple[int, int]] = None,
              patch_grid: Optional[Tuple[int, int]] = None) -> jnp.ndarray:
    n, h, w, _ = images.shape
    if patch_size is None == patch_grid is None:
      raise ValueError(
          "You must specify either patch_size or patch_grid, and not both "
          f"(patch_size = {patch_size}, patch_grid = {patch_grid})")
    elif patch_size is None:
      patch_size = (h // patch_grid[0], w // patch_grid[1])
    x = nn.Conv(
        hidden_size,
        patch_size,
        strides=patch_size,
        padding="VALID",
        name="embedding")(
            images)
    return jnp.reshape(x, [n, -1, hidden_size])

  @nn.compact
  def __call__(self, images: jnp.ndarray, train: Optional[bool] = None,
               mean_field_factor: float = -1., **gp_kwargs):
    train = nn.module.merge_param("train", self.train, train)
    # Convert images to patches.
    x = self.patches(images, self.hidden_size, self.patch_size, self.patch_grid)
    # Add "class" token if necessary.
    n, _, c = x.shape
    if self.classifier == "token":
      cls = self.param("cls", nn.initializers.zeros, (1, 1, self.hidden_size))
      cls = jnp.tile(cls, [n, 1, 1])
      x = jnp.concatenate([cls, x], axis=1)
    # Encode tokens.
    x, extra_info = vit_batchensemble.BatchEnsembleEncoder(
        train=train, name="Transformer", **self.transformer)(
            x)
    # Reduce tokens to a single vector representation.
    if self.classifier == "token":
      # Take the first token's output as representation as in BERT.
      x = x[:, 0]
    elif self.classifier == "gap":
      # Average all tokens.
      x = jnp.mean(x, axis=tuple(range(1, x.ndim - 1)))  # (1,) or (1, 2)
    elif self.classifier == "map":
      probe = self.param("probe", nn.initializers.xavier_uniform(), (1, 1, c))
      # x may have been subject to tiling, n can be different from x.shape[0].
      probe = jnp.tile(probe, [x.shape[0], 1, 1])
      attention = nn.MultiHeadDotProductAttention(
          deterministic=not train,
          num_heads=self.transformer.get("attention", {}).get("num_heads", 1),
          kernel_init=nn.initializers.xavier_uniform())
      x = attention(inputs_q=probe, inputs_kv=x)
      y = nn.LayerNorm()(x)
      y = vit.MlpBlock(
          mlp_dim=self.transformer["mlp_dim"], dropout_rate=0)(
              y, deterministic=not train)
      x = (x + y)[:, 0]
    else:
      raise ValueError(f"Unknown classifier: {self.classifier}")

    if self.representation_size is None:
      x = vit.IdentityLayer(name="pre_logits")(x)
      extra_info["pre_logits"] = x
    else:
      x = nn.Dense(self.representation_size, name="pre_logits")(x)
      extra_info["pre_logits"] = x
      x = nn.tanh(x)

    if self.use_gp_layer:
      x_gp = self.gp_layer(x, **gp_kwargs)
      # Gaussian process layer output: a tuple of logits, covmat, and optionally
      # random features.
      extra_info["covmat"] = x_gp[1]
      if len(x_gp) > 2:
        extra_info["random_features"] = x_gp[2]
      if train:
        x = x_gp[0]
      else:
        # During inference, compute posterior mean by adjusting the original
        # logits with predictive uncertainty.
        x = ed.nn.utils.mean_field_logits(
            logits=x_gp[0], covmat=x_gp[1], mean_field_factor=mean_field_factor)
    else:
      x = nn.Dense(
          self.num_classes, kernel_init=self.head_kernel_init,
          name="batchensemble_head")(
              x)
    return x, extra_info


def vision_transformer_be_gp(
    num_classes: int,
    hidden_size: int,
    transformer: Params,
    patch_size: Optional[Tuple[int, int]] = None,
    patch_grid: Optional[Tuple[int, int]] = None,
    representation_size: Optional[int] = None,
    classifier: str = "token",
    head_kernel_init: InitializeFn = nn.initializers.zeros,
    use_gp_layer: bool = True,
    gp_layer_kwargs: Mapping[str, Any] = default_kwarg_dict(),
    train: Optional[bool] = None):
  """Builds a BatchEnsemble GP Vision Transformer (ViT) model."""
  # TODO(dusenberrymw): Add API docs once the config dict in VisionTransformerBE
  # is cleaned up.
  return VisionTransformerBEGP(
      num_classes=num_classes,
      transformer=transformer,
      hidden_size=hidden_size,
      patch_size=patch_size,
      patch_grid=patch_grid,
      representation_size=representation_size,
      classifier=classifier,
      head_kernel_init=head_kernel_init,
      use_gp_layer=use_gp_layer,
      gp_layer_kwargs=gp_layer_kwargs,
      train=train)

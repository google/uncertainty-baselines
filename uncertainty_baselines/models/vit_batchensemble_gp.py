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

from typing import Any, Callable, Iterable, Mapping, Optional, Sequence, Tuple

from absl import logging
import edward2.jax as ed
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import scipy

from uncertainty_baselines.models import vit
from uncertainty_baselines.models import vit_batchensemble

# TODO(dusenberrymw): Open-source remaining imports.
identity = None
checkpoints_model = None


DType = type(jnp.float32)
InitializeFn = Callable[[jnp.ndarray, Iterable[int], DType], jnp.ndarray]
Params = Mapping[str, Any]

default_kwarg_dict = lambda: dataclasses.field(default_factory=dict)


class PatchTransformerBEGP(nn.Module):
  """Patch transformer with BatchEnsemble and GP last layer.

  You must specify either the vertical and horizontal resolution of the patches
  (patch_size), or the number of vertical and horizontal divisions of the input
  image (patch_grid).
  """
  patch_size: Optional[Tuple[int, int]] = None
  patch_grid: Optional[Tuple[int, int]] = None
  num_classes: int = 1000
  train: Optional[bool] = None
  hidden_size: int = 1024
  representation_size: Optional[int] = None
  transformer: Optional[Params] = None
  classifier: str = "token"
  head_kernel_init: InitializeFn = nn.initializers.zeros
  use_gp_layer: bool = True
  gp_layer_kwargs: Mapping[str, Any] = default_kwarg_dict()

  def setup(self):
    # pylint:disable=not-a-mapping
    if self.use_gp_layer:
      self.gp_layer = ed.nn.RandomFeatureGaussianProcess(
          features=self.num_classes, name="head", **self.gp_layer_kwargs)
    # pylint:enable=not-a-mapping

  @classmethod
  def load(
      cls,
      prefix: str,
      init_params: Mapping[str, Any],
      model_params: Mapping[str, Any],
      partition_specs: Sequence[Any],
      keep_head: bool = False,
  ) -> Mapping[str, Any]:
    """Loads from Transformer checkpoint except head parameters.

    Args:
      prefix: Prefix of the model checkpoint to use.
      init_params: Dictionary with unreplicated parameters of the new model.
      model_params: Dictionary with the configuration of the new model.
      partition_specs: A sequence of PartitionSpecs. They map expert parameter
        names (RegEx patterns) to a TPU layout. Expected to be None or empty.
      keep_head: bool, whether head must be kept or replaced with a random one.

    Returns:
      A new dictionary of params to replace init_params.
    """
    local_devices = sorted(jax.local_devices(), key=lambda device: device.id)
    if partition_specs:
      raise ValueError("Partition specs cannot be used for Batchensemble.")
    restored = None
    if restored is None:
      raise ValueError(f"No valid checkpoints with prefix {prefix!r}")
    # Checkpoints contain FrozenDicts, which are immutable.
    restored_params = flax.core.unfreeze(restored["target"])
    # The following allows implementing both fine-tuning head variants from
    # https://docs.google.com/presentation/d/1mWGpOoCq1TGESg7ZpQwBIxBpEQxWk9cjfeVS_qQi1Gc/edit#slide=id.g9798de2d4d_2_0
    # depending on the value of `representation_size` in the fine-tuning job:
    # - `None` is variant 3 (c-head): drop the whole head and add a nn.Linear.
    # - same number as in pre-training means variant 1 (a-head): keep the head
    #   but reset the last layer (logits) for the new task.
    if model_params["representation_size"] is None:
      if "pre_logits" in restored_params:
        logging.info("Resformer: drop-head variant")
        del restored_params["pre_logits"]
    if not keep_head:
      restored_params["batchensemble_head"]["kernel"] = np.stack(
          [init_params["batchensemble_head"]["kernel"]] * len(local_devices))
      restored_params["batchensemble_head"]["bias"] = np.stack(
          [init_params["batchensemble_head"]["bias"]] * len(local_devices))
    # The following implements "high-res finetuning" for transformer models.
    if "posembed_input" in restored_params.get("Transformer", {}):
      # Rescale the grid of position embeddings. Param shape is (1,N,rep.size)
      posemb = (
          restored_params["Transformer"]["posembed_input"]["pos_embedding"][0])
      posemb_new = init_params["Transformer"]["posembed_input"]["pos_embedding"]
      if posemb.shape != posemb_new.shape:
        logging.info("Resformer: resized variant: %s to %s", posemb.shape,
                     posemb_new.shape)
        ntok_new = posemb_new.shape[1]

        if (model_params.get("cls_token", False) or
            model_params.get("classifier", None) == "token"):
          posemb_tok, posemb_grid = posemb[:, :1], posemb[0, 1:]
          ntok_new -= 1
        else:
          posemb_tok, posemb_grid = posemb[:, :0], posemb[0]

        gs_old = int(np.sqrt(len(posemb_grid)))
        gs_new = int(np.sqrt(ntok_new))
        logging.info("Resformer: grid-size from %s to %s", gs_old, gs_new)
        posemb_grid = posemb_grid.reshape(gs_old, gs_old, -1)

        zoom = (gs_new / gs_old, gs_new / gs_old, 1)
        posemb_grid = scipy.ndimage.zoom(posemb_grid, zoom, order=1)
        posemb_grid = posemb_grid.reshape(1, gs_new * gs_new, -1)
        posemb = jnp.array(np.concatenate([posemb_tok, posemb_grid], axis=1))
        restored_params["Transformer"]["posembed_input"][
            "pos_embedding"] = np.stack([posemb] * len(local_devices))

    return flax.core.freeze(restored_params)

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
    transformer = self.transformer or {}
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
        train=train, name="Transformer", **transformer)(
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
          num_heads=transformer.get("attention", {}).get("num_heads", 1),
          kernel_init=nn.initializers.xavier_uniform())
      x = attention(inputs_q=probe, inputs_kv=x)
      y = nn.LayerNorm()(x)
      y = vit.MlpBlock(
          mlp_dim=transformer["mlp_dim"], dropout_rate=0)(
              y, deterministic=not train)
      x = (x + y)[:, 0]
    else:
      raise ValueError(f"Unknown classifier: {self.classifier}")

    if self.representation_size is None:
      x = identity.IdentityLayer(name="pre_logits")(x)
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

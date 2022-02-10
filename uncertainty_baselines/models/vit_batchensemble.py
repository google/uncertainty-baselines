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

import functools
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

# TODO(dusenberrymw): Open-source remaining imports.
identity = None
checkpoints_model = None


DType = type(jnp.float32)
InitializeFn = Callable[[jnp.ndarray, Iterable[int], DType], jnp.ndarray]
Params = Mapping[str, Any]


class BatchEnsembleMlpBlock(nn.Module):
  """Transformer MLP / feed-forward block with BatchEnsemble layers."""
  mlp_dim: int
  ens_size: int
  random_sign_init: float
  dtype: Optional[DType] = None
  out_dim: Optional[int] = None
  dropout_rate: float = 0.0
  deterministic: Optional[bool] = None
  kernel_init: InitializeFn = nn.initializers.xavier_uniform()
  bias_init: InitializeFn = nn.initializers.normal(stddev=1e-6)

  @nn.compact
  def __call__(self,
               inputs: jnp.ndarray,
               deterministic: Optional[bool] = None) -> jnp.ndarray:
    """Applies BatchEnsemble MlpBlock module."""
    deterministic = nn.module.merge_param("deterministic", self.deterministic,
                                          deterministic)
    dtype = self.dtype or inputs.dtype
    inputs = jnp.asarray(inputs, self.dtype)
    out_dim = self.out_dim or inputs.shape[-1]
    x = ed.nn.DenseBatchEnsemble(
        self.mlp_dim,
        self.ens_size,
        activation=None,
        alpha_init=ed.nn.utils.make_sign_initializer(self.random_sign_init),
        gamma_init=ed.nn.utils.make_sign_initializer(self.random_sign_init),
        kernel_init=self.kernel_init,
        bias_init=self.bias_init,
        name="Dense_0",
        dtype=dtype)(
            inputs)
    x = nn.gelu(x)
    x = nn.Dropout(rate=self.dropout_rate, deterministic=deterministic)(x)
    output = ed.nn.DenseBatchEnsemble(
        out_dim,
        self.ens_size,
        activation=None,
        alpha_init=ed.nn.utils.make_sign_initializer(self.random_sign_init),
        gamma_init=ed.nn.utils.make_sign_initializer(self.random_sign_init),
        kernel_init=self.kernel_init,
        bias_init=self.bias_init,
        name="Dense_1",
        dtype=dtype)(
            x)
    output = nn.Dropout(
        rate=self.dropout_rate, deterministic=deterministic)(
            output)
    return output


class Encoder1DBlock(nn.Module):
  """Transformer encoder layer.

  If mlp_class returns a tuple (e.g. TokenMoeBlock) with auxiliary information,
  this also returns the auxiliary information.
  """
  mlp_class: Callable  # pylint: disable=g-bare-generic
  num_heads: int
  dtype: Optional[DType] = None
  dropout_rate: float = 0.0
  attention_dropout_rate: float = 0.0

  @nn.compact
  def __call__(self,
               inputs: jnp.ndarray,
               *,
               deterministic: Optional[bool] = None):
    """Applies Encoder1Dlock module."""
    assert inputs.ndim == 3, f"Expected (batch, seq, hidden) got {inputs.shape}"

    x = nn.LayerNorm(dtype=self.dtype, name="LayerNorm_0")(inputs)
    x = nn.MultiHeadDotProductAttention(
        dtype=self.dtype,
        kernel_init=nn.initializers.xavier_uniform(),
        broadcast_dropout=False,
        deterministic=deterministic,
        name="MultiHeadDotProductAttention_1",
        num_heads=self.num_heads,
        dropout_rate=self.attention_dropout_rate)(x, x)
    x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=deterministic)
    x = x + inputs

    # MLP block.
    y = nn.LayerNorm(dtype=self.dtype, name="LayerNorm_2")(x)
    y = self.mlp_class(name="MlpBlock_3")(y, deterministic=deterministic)

    return x + y


class BatchEnsembleEncoder(nn.Module):
  """Transformer Model Encoder with BE MLP blocks every two layers.

  This replicates the encoder described in Gshard paper, modulo the
  differences described in the class `BatchEnsembleMlpBlock` above.

  Attributes:
    num_layers: number of layers (a.k.a. number of blocks in the encoder).
    mlp_dim: dimension of the mlp on top of attention block
    num_heads: number of heads for the attention layer.
    dtype: the dtype of the computation (default: same as inputs).
    dropout_rate: dropout rate.
    attention_dropout_rate: dropout rate for the attention.
    train: True if the module is used for training.
    be_layers: Sequence of layers where BE MLPs are included. If None, use BE
      MLP blocks in every other layer (1, 3, 5, ...). First layer is 0.
  """
  num_layers: int
  mlp_dim: int
  ens_size: int
  random_sign_init: float
  num_heads: int
  dtype: Optional[DType] = None
  dropout_rate: float = 0.0
  attention_dropout_rate: float = 0.0
  train: Optional[bool] = None
  be_layers: Optional[Sequence[int]] = None

  @nn.compact
  def __call__(self,
               inputs: jnp.ndarray,
               inputs_positions: Optional[jnp.ndarray] = None,
               train: Optional[bool] = None):
    """Applies Transformer model on the inputs."""
    train = nn.module.merge_param("train", self.train, train)
    dtype = self.dtype or inputs.dtype
    assert inputs.ndim == 3  # (batch, len, emb)
    # List indicating which MLPs to substitute with BatchEnsemble MLPs.
    be_layers = self.be_layers
    if be_layers is None:
      be_layers = list(range(1, self.num_layers, 2))
    if self.ens_size > 1 and not be_layers:
      raise ValueError(
          "Must have `ens_size = 1` when not using any BE layers, but received "
          f"ens_size = {self.ens_size}.")

    def is_first_be_layer(lyr: int) -> bool:
      if be_layers:
        return lyr == min(be_layers)
      return False

    x = vit.AddPositionEmbs(
        name="posembed_input", posemb_init=nn.initializers.normal(stddev=0.02))(
            inputs)
    x = nn.Dropout(rate=self.dropout_rate, deterministic=not self.train)(x)

    be_params = dict(
        ens_size=self.ens_size, random_sign_init=self.random_sign_init)
    mlp_params = dict(dtype=dtype, name="mlp")
    mlp_params_dense = dict(
        dropout_rate=self.dropout_rate, mlp_dim=self.mlp_dim)
    mlp_dense = functools.partial(vit.MlpBlock, **mlp_params,
                                  **mlp_params_dense)
    be_block = functools.partial(BatchEnsembleMlpBlock, **mlp_params,
                                 **mlp_params_dense, **be_params)
    extra_info = dict()
    for lyr in range(self.num_layers):
      encoder_block = functools.partial(
          Encoder1DBlock,
          num_heads=self.num_heads,
          dtype=dtype,
          dropout_rate=self.dropout_rate,
          attention_dropout_rate=self.attention_dropout_rate,
          name=f"encoderblock_{lyr}")
      if lyr in be_layers:
        # We need to tile inputs before the first BE layer.
        if is_first_be_layer(lyr):
          x = jnp.tile(x, [self.ens_size] + [1] * (x.ndim - 1))

        x = encoder_block(mlp_class=be_block)(x, deterministic=not train)
      else:
        x = encoder_block(mlp_class=mlp_dense)(x, deterministic=not train)
    encoded = nn.LayerNorm(name="encoder_norm")(x)

    return encoded, extra_info


class PatchTransformerBE(nn.Module):
  """Patch transformer with BE layers in the encoder.

  You must specify either the vertical and horizontal resolution of the patches
  (patch_size).
  """
  patches: Any
  num_classes: int = 1000
  train: Optional[bool] = None
  hidden_size: int = 1024
  representation_size: Optional[int] = None
  transformer: Optional[Params] = None
  classifier: str = "token"
  head_kernel_init: InitializeFn = nn.initializers.zeros

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

  def embed(self,
            images: jnp.ndarray,
            hidden_size: int,
            patch_size: Tuple[int, int]) -> jnp.ndarray:
    x = nn.Conv(
        hidden_size,
        patch_size,
        strides=patch_size,
        padding="VALID",
        name="embedding")(
            images)
    n, h, w, c = x.shape
    x = jnp.reshape(x, [n, h * w, c])
    return x

  @nn.compact
  def __call__(self, images: jnp.ndarray, train: Optional[bool] = None):
    train = nn.module.merge_param("train", self.train, train)
    transformer = self.transformer or {}
    # Convert images to patches.
    x = self.embed(images, self.hidden_size, self.patches.size)
    # Add "class" token if necessary.
    n, _, c = x.shape
    if self.classifier == "token":
      cls = self.param("cls", nn.initializers.zeros, (1, 1, self.hidden_size))
      cls = jnp.tile(cls, [n, 1, 1])
      x = jnp.concatenate([cls, x], axis=1)
    # Encode tokens.
    x, extra_info = BatchEnsembleEncoder(
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
      x = ed.nn.DenseBatchEnsemble(
          self.representation_size,
          self.transformer.get("ens_size"),
          activation=None,
          alpha_init=ed.nn.utils.make_sign_initializer(
              self.transformer.get("random_sign_init")),
          gamma_init=ed.nn.utils.make_sign_initializer(
              self.transformer.get("random_sign_init")),
          name="pre_logits")(x)
      extra_info["pre_logits"] = x
      x = nn.tanh(x)

    x = ed.nn.DenseBatchEnsemble(
        self.num_classes,
        self.transformer.get("ens_size"),
        activation=None,
        alpha_init=ed.nn.utils.make_sign_initializer(
            self.transformer.get("random_sign_init")),
        gamma_init=ed.nn.utils.make_sign_initializer(
            self.transformer.get("random_sign_init")),
        kernel_init=self.head_kernel_init,
        name="batchensemble_head")(x)
    return x, extra_info

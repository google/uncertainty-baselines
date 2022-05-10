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

"""BatchEnsemble Vision Transformer (ViT) model."""

from typing import Any, Callable, Iterable, Mapping, Optional, Sequence, Tuple

import edward2.jax as ed
import flax.linen as nn
import jax.numpy as jnp

from uncertainty_baselines.models import vit

DType = type(jnp.float32)
InitializeFn = Callable[[jnp.ndarray, Iterable[int], DType], jnp.ndarray]
Params = Mapping[str, Any]


class IdentityLayer(nn.Module):
  """Identity layer, convenient for giving a name to an array."""

  @nn.compact
  def __call__(self, x):
    return x


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
  """Transformer encoder layer."""
  mlp_dim: int
  num_heads: int
  ens_size: int
  random_sign_init: float
  ensemble_attention: bool = False
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
    # TODO(trandustin): Remove `ensemble_attention` hparam once we no longer
    # need checkpoints that only apply BE on the FF block.
    if self.ensemble_attention:
      x = ed.nn.MultiHeadDotProductAttentionBE(
          dtype=self.dtype,
          kernel_init=nn.initializers.xavier_uniform(),
          broadcast_dropout=False,
          deterministic=deterministic,
          name="MultiHeadDotProductAttention_1",
          num_heads=self.num_heads,
          ens_size=self.ens_size,
          alpha_init=ed.nn.utils.make_sign_initializer(self.random_sign_init),
          gamma_init=ed.nn.utils.make_sign_initializer(self.random_sign_init),
          dropout_rate=self.attention_dropout_rate)(x, x)
    else:
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
    y = BatchEnsembleMlpBlock(
        mlp_dim=self.mlp_dim,
        ens_size=self.ens_size,
        random_sign_init=self.random_sign_init,
        dtype=self.dtype,
        name="MlpBlock_3",
        dropout_rate=self.dropout_rate)(y, deterministic=deterministic)

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
  ensemble_attention: bool = False
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

    extra_info = dict()
    for lyr in range(self.num_layers):
      params = dict(
          mlp_dim=self.mlp_dim,
          num_heads=self.num_heads,
          dtype=dtype,
          dropout_rate=self.dropout_rate,
          attention_dropout_rate=self.attention_dropout_rate,
          name=f"encoderblock_{lyr}")
      if lyr in be_layers:
        # We need to tile inputs before the first BE layer.
        if is_first_be_layer(lyr):
          x = jnp.tile(x, [self.ens_size] + [1] * (x.ndim - 1))

        x = Encoder1DBlock(
            ens_size=self.ens_size,
            random_sign_init=self.random_sign_init,
            ensemble_attention=self.ensemble_attention,
            **params)(x, deterministic=not train)
      else:
        x = vit.Encoder1DBlock(**params)(x, deterministic=not train)
    encoded = nn.LayerNorm(name="encoder_norm")(x)

    return encoded, extra_info


class VisionTransformerBE(nn.Module):
  """BatchEnsemble Vision Transformer model.

  You must specify either the vertical and horizontal resolution of the patches
  (patch_size).
  """
  num_classes: int
  patches: Any
  transformer: Params
  hidden_size: int
  representation_size: Optional[int] = None
  classifier: str = "token"
  head_kernel_init: InitializeFn = nn.initializers.zeros
  train: Optional[bool] = None

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
      x = IdentityLayer(name="pre_logits")(x)
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


def vision_transformer_be(
    num_classes: int,
    patches: Any,
    transformer: Params,
    hidden_size: int,
    representation_size: Optional[int] = None,
    classifier: str = "token",
    head_kernel_init: InitializeFn = nn.initializers.zeros,
    train: Optional[bool] = None):
  """Builds a BatchEnsemble Vision Transformer (ViT) model."""
  # TODO(dusenberrymw): Add API docs once the config dict in VisionTransformerBE
  # is cleaned up.
  return VisionTransformerBE(
      num_classes=num_classes,
      patches=patches,
      transformer=transformer,
      hidden_size=hidden_size,
      representation_size=representation_size,
      classifier=classifier,
      head_kernel_init=head_kernel_init,
      train=train)

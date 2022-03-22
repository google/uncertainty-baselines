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

"""Vision Transformer with Heteroskedastic, GP, and BatchEnsemble."""
from typing import Any, Callable, Iterable, Mapping, Optional, Tuple

import edward2.jax as ed
import flax.linen as nn
import jax
import jax.numpy as jnp
import uncertainty_baselines.models.vit as vit
import uncertainty_baselines.models.vit_batchensemble as vit_batchensemble

# Jax data types
Array = Any
DType = type(jnp.float32)
InitializeFn = Callable[[jnp.ndarray, Iterable[int], DType], jnp.ndarray]


# TODO(dusenberrymw,zmariet): Clean up and generalize these log marginal probs.
def log_average_softmax_probs(logits: jnp.ndarray) -> jnp.ndarray:
  # TODO(zmariet): dedicated eval loss function.
  ens_size, _, _ = logits.shape
  log_p = jax.nn.log_softmax(logits)  # (ensemble_size, batch_size, num_classes)
  log_p = jax.nn.logsumexp(log_p, axis=0) - jnp.log(ens_size)
  return log_p


def log_average_sigmoid_probs(logits: jnp.ndarray) -> jnp.ndarray:
  ens_size, _, _ = logits.shape
  log_p = jax.nn.log_sigmoid(logits)  # (ensemble_size, batch_size, num_classes)
  log_p = jax.nn.logsumexp(log_p, axis=0) - jnp.log(ens_size)
  log_not_p = jax.nn.log_sigmoid(-logits)
  log_not_p = jax.nn.logsumexp(log_not_p, axis=0) - jnp.log(ens_size)
  log_p = log_p - log_not_p
  return log_p


class VisionTransformerHetGPBE(nn.Module):
  """Vision transformer."""

  num_classes: int
  patches: Any
  # BatchEnsemble's hparams appear in transformer.
  transformer: Any
  hidden_size: int
  representation_size: Optional[int] = None
  classifier: str = 'token'
  fix_base_model: bool = False
  head_kernel_init: InitializeFn = nn.initializers.zeros

  # Heteroscedastic
  multiclass: bool = False  # also used for BatchEnsemble
  temperature: float = 1.0
  mc_samples: int = 1000
  num_factors: int = 0
  param_efficient: bool = True
  return_locs: bool = False

  # GP
  use_gp: bool = False
  covmat_momentum: float = 0.999
  ridge_penalty: float = 1.0
  mean_field_factor: float = -1.0

  @nn.compact
  def __call__(self,
               inputs: Array,
               train: bool,
               **kwargs) -> Tuple[Array, Mapping[str, Any]]:
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

    x, _ = vit_batchensemble.BatchEnsembleEncoder(
        name='Transformer', **self.transformer)(x, train=train)
    out['transformed'] = x

    if self.classifier == 'token':
      x = x[:, 0]
    elif self.classifier == 'gap':
      x = jnp.mean(x, axis=list(range(1, x.ndim - 1)))  # (1,) or (1,2)
    else:
      raise ValueError(f'Invalid classifier={self.classifier}')

    out['head_input'] = x

    if self.representation_size is not None:
      x = ed.nn.DenseBatchEnsemble(
          self.representation_size,
          self.transformer.get('ens_size'),
          activation=None,
          alpha_init=ed.nn.utils.make_sign_initializer(
              self.transformer.get('random_sign_init')),
          gamma_init=ed.nn.utils.make_sign_initializer(
              self.transformer.get('random_sign_init')),
          name='pre_logits')(x)
      out['pre_logits'] = x
      x = nn.tanh(x)
    else:
      x = vit.IdentityLayer(name='pre_logits')(x)
      out['pre_logits'] = x

    # TODO(markcollier): Fix base model without using stop_gradient.
    if self.fix_base_model:
      x = jax.lax.stop_gradient(x)

    if self.use_gp:
      if self.covmat_momentum < 0.:
        gp_layer_kwargs = {'covmat_kwargs': {'momentum': None}}
      else:
        gp_layer_kwargs = {'covmat_kwargs': {'momentum': self.covmat_momentum}}

      if self.multiclass:
        raise NotImplementedError('Multi-class HetSNGP layer not available.')
      else:
        gp_layer = ed.nn.MCSigmoidDenseFASNGPBE(
            num_outputs=self.num_classes,
            num_factors=self.num_factors,
            temperature=self.temperature,
            parameter_efficient=self.param_efficient,
            train_mc_samples=self.mc_samples,
            test_mc_samples=self.mc_samples,
            ens_size=self.transformer.get('ens_size'),
            logits_only=True, name='head', **gp_layer_kwargs)
      x_gp = gp_layer(x, training=train, **kwargs)

      # Gaussian process layer output: a tuple of logits, covmat, and optionally
      # random features.
      out['logits'] = x_gp[0]
      out['covmat'] = x_gp[1]

      logits = x_gp[0]
    else:
      # Note we're using non-BE layers.
      if self.multiclass:
        output_layer = ed.nn.MCSoftmaxDenseFABE(
            self.num_classes,
            self.num_factors,
            self.temperature,
            self.param_efficient,
            self.mc_samples,
            self.mc_samples,
            logits_only=True,
            return_locs=self.return_locs,
            alpha_init=ed.nn.utils.make_sign_initializer(
                self.transformer.get('random_sign_init')),
            gamma_init=ed.nn.utils.make_sign_initializer(
                self.transformer.get('random_sign_init')),
            kernel_init=self.head_kernel_init,
            name='head')
      else:
        output_layer = ed.nn.MCSigmoidDenseFABE(
            num_outputs=self.num_classes,
            num_factors=self.num_factors,
            temperature=self.temperature,
            parameter_efficient=self.param_efficient,
            train_mc_samples=self.mc_samples,
            test_mc_samples=self.mc_samples,
            logits_only=True,
            return_locs=self.return_locs,
            alpha_init=ed.nn.utils.make_sign_initializer(
                self.transformer.get('random_sign_init')),
            gamma_init=ed.nn.utils.make_sign_initializer(
                self.transformer.get('random_sign_init')),
            kernel_init=self.head_kernel_init,
            name='head')
      logits = output_layer(x)
      out['logits'] = logits

    return logits, out

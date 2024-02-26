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

"""Random-feature Gaussian Process model with T5 backbone."""
from typing import Any, Optional
import warnings

import edward2.jax as ed
import flax.core
import flax.linen as nn
import jax
import jax.numpy as jnp
import t5x.examples.t5.layers as t5_layers
import t5x.examples.t5.network as t5_network

# Jax data types.
Array = Any


# This class follows `network.Decoder` implementation.
class GaussianProcessDecoder(nn.Module):
  """T5 Decoder with Gaussian Process output head."""
  config: t5_network.T5Config
  shared_embedding: nn.Module
  # Gaussian process parameters.
  use_gp_layer: bool = True
  covmat_momentum: float = -1.
  normalize_input: bool = True
  # NOTE: We let mean_field_factor as in the constructor to
  # preserve the signature of `__call__` method.
  mean_field_factor: float = -1.
  ridge_penalty: float = 1.
  steps_per_epoch: Optional[int] = None

  def setup(self):
    if self.config.logits_via_embedding:
      warnings.warn('Sharing the embedding weights in the decoder output '
                    'layer is not supported in the GP decoder.',
                    RuntimeWarning)
    # pylint:disable=not-a-mapping
    if self.use_gp_layer:
      covmat_momentum = None if self.covmat_momentum < 0. else self.covmat_momentum
      gp_layer_kwargs = dict(normalize_input=self.normalize_input,
                             covmat_kwargs=dict(momentum=covmat_momentum))
      self.gp_layer = ed.nn.RandomFeatureGaussianProcess(
          features=self.config.vocab_size, name='gp_head', **gp_layer_kwargs)
    # pylint:enable=not-a-mapping

  def _apply_gp_layer(self, y, train=True):
    initializing = self.is_mutable_collection('params')
    if initializing:
      rng = self.make_rng('params')
      x_gp, variables = self.gp_layer.init_with_output(
          rng, y, return_full_covmat=False)
      variables = flax.core.unfreeze(variables)
      params = variables.pop('params')
      # We put parameters into the 'params' scope. With this, all parameters
      # of gp_layer will have the same signatures as if we call gp_layer(y).
      self.scope.put_variable('params', self.gp_layer.name, params)
      # We also put the remaining variables (e.g. the kernel and bias
      # variables of the random fourier feature module) into the 'params'
      # scope.
      variables['step'] = jnp.array(0., dtype=jnp.float32)
      variables = flax.core.freeze(variables)
      self.scope.put_variable('params', 'gp_head_state', variables)
    else:
      gp_params = self.scope.get_variable('params', self.gp_layer.name)
      gp_state = self.scope.get_variable('params', 'gp_head_state')
      gp_state = jax.tree_util.tree_map(
          lambda x: x.astype(jnp.float32), gp_state)
      gp_state = flax.core.unfreeze(gp_state)
      step = gp_state.pop('step')
      if self.covmat_momentum < 0 and (self.steps_per_epoch is not None):
        # Reset precision matrix at the start of the new epoch.
        reset_covmat = (step % self.steps_per_epoch) < 0.5
      else:
        reset_covmat = 0.
      covmat_collection_name = self.gp_layer.covmat_layer.collection_name
      covmat_state = gp_state[covmat_collection_name]['covmat_layer']
      prec_mat_old = covmat_state['precision_matrix']
      prec_mat_new = (
          (1. - reset_covmat) * prec_mat_old +
          reset_covmat * jnp.eye(prec_mat_old.shape[0]) * self.ridge_penalty)
      covmat_state['precision_matrix'] = prec_mat_new

      # We will stop gradient here so that the gradient of the loss with
      # respect to those states will be zero.
      variables = jax.lax.stop_gradient(gp_state)
      variables['params'] = gp_params
      variables = flax.core.freeze(variables)
      if train:
        x_gp, new_state = self.gp_layer.apply(
            variables,
            y,
            return_full_covmat=False,
            mutable=list(gp_state.keys()))
        # Update step.
        new_state = flax.core.unfreeze(new_state)
        new_state['step'] = step + 1.
        new_state = flax.core.freeze(new_state)
        self.sow('intermediates', 'gp_head_state_new', new_state)
      else:
        x_gp = self.gp_layer.apply(variables, y, return_full_covmat=False)

    return x_gp

  # Different from ViT implementation, we will store the intermediate values
  # using `self.sow('intermediates', ...)` to preserve the signature of this
  # method. More precisely, only logits will be returned.
  @nn.compact
  def __call__(self,
               encoded,
               decoder_input_tokens,
               decoder_positions=None,
               decoder_mask=None,
               encoder_decoder_mask=None,
               deterministic=False,
               decode=False,
               max_decode_length=None) -> Array:
    cfg = self.config
    assert decoder_input_tokens.ndim == 2  # [batch, len]
    rel_emb = t5_layers.RelativePositionBiases(
        num_buckets=32,
        max_distance=128,
        num_heads=cfg.num_heads,
        dtype=cfg.dtype,
        embedding_init=nn.initializers.variance_scaling(1.0, 'fan_avg',
                                                        'uniform'),
        name='relpos_bias')

    # [batch, length] -> [batch, length, emb_dim]
    y = self.shared_embedding(decoder_input_tokens.astype('int32'))
    y = nn.Dropout(
        rate=cfg.dropout_rate, broadcast_dims=(-2,))(
            y, deterministic=deterministic)
    y = y.astype(cfg.dtype)

    for lyr in range(cfg.num_decoder_layers):
      # [batch, length, emb_dim] -> [batch, length, emb_dim]
      y = t5_network.DecoderLayer(
          config=cfg,
          relative_embedding=rel_emb,
          name=f'layers_{lyr}')(
              y,
              encoded,
              decoder_mask=decoder_mask,
              encoder_decoder_mask=encoder_decoder_mask,
              deterministic=deterministic,
              decode=decode,
              max_decode_length=max_decode_length)

    y = t5_layers.LayerNorm(dtype=cfg.dtype, name='decoder_norm')(y)
    y = nn.Dropout(
        rate=cfg.dropout_rate, broadcast_dims=(-2,))(
            y, deterministic=deterministic)

    # NOTE: This is the place that will be different from the T5 transformer.
    self.sow('intermediates', 'pre_logits', y)
    # [batch, length, emb_dim] -> [batch, length, vocab_size]
    if not self.use_gp_layer:
      if cfg.logits_via_embedding:
        # Use the transpose of embedding matrix for logit transform.
        logits = self.shared_embedding.attend(y)
        # Correctly normalize pre-softmax logits for this shared case.
        logits = logits / jnp.sqrt(y.shape[-1])
      else:
        logits = t5_layers.DenseGeneral(
            cfg.vocab_size,
            dtype=jnp.float32,  # Use float32 for stabiliity.
            kernel_axes=('embed', 'vocab'),
            name='logits_dense')(
                y)
      self.sow('intermediates', 'logits', logits)
    else:
      # NOTE: In t5x.trainer, dropout_rng is not None when training and is
      # None when evaluating. In t5x.models.EncoderDecoderModel, when using
      # the transformer's apply method to get the logits, a flag
      # `enable_dropout` is provided. It is None if dropout_rng is None.
      # Finally, in t5_network.Transformer.decode method, `deterministic` is
      # set to `not enable_dropout`. So we can use this flag to decide if we
      # are in the train mode.
      # TODO(phandu): Consider to drop this treat and expose the `train`
      # argument to this call method, when the pipeline is in better shape.
      train = not deterministic

      # Using Gaussian process output layer.
      # This is the only place that T5-GP differs from deterministic T5.
      # TODO(phandu): Consider adding a new class field like
      # `store_random_features` for `return_random_features` argument
      # of RandomFeatureGaussianProcess.
      x_gp = self._apply_gp_layer(y, train=train)

      # Gaussian process layer output: a tuple of logits, covmat, and
      # optionally random features.
      self.sow('intermediates', 'logits', x_gp[0])
      covmat = x_gp[1]
      if covmat is not None:
        # The gp_layer considers all dimensions except the last one as
        # batch dimensions and returns a diagonal covariance matrix whose
        # size is the size (i.e. product) of those batch dimensions.
        # So we need to reshape here.
        covmat = covmat.reshape(x_gp[0].shape[:-1])
      self.sow('intermediates', 'covmat', covmat)

      if not train:
        # During inference, compute posterior mean by adjusting the original
        # logits with predictive uncertainty.
        logits = ed.nn.utils.mean_field_logits(
            logits=x_gp[0],
            covmat=covmat,
            mean_field_factor=self.mean_field_factor)
      else:
        logits = x_gp[0]

    return logits


class TransformerGaussianProcess(t5_network.Transformer):
  """T5 Transformer with Gaussian Process output head."""
  config: t5_network.T5Config
  # Gaussian process parameters.
  use_gp_layer: bool = True
  covmat_momentum: float = -1.
  normalize_input: bool = True
  mean_field_factor: float = -1.
  ridge_penalty: float = 1.
  steps_per_epoch: Optional[int] = None

  def setup(self):
    cfg = self.config
    self.shared_embedding = t5_layers.Embed(
        num_embeddings=cfg.vocab_size,
        features=cfg.emb_dim,
        dtype=cfg.dtype,
        attend_dtype=jnp.float32,  # for logit training stability
        embedding_init=nn.initializers.normal(stddev=1.0),
        one_hot=True,
        name='token_embedder')

    self.encoder = t5_network.Encoder(
        config=cfg, shared_embedding=self.shared_embedding)
    self.decoder = GaussianProcessDecoder(
        config=cfg,
        shared_embedding=self.shared_embedding,
        use_gp_layer=self.use_gp_layer,
        covmat_momentum=self.covmat_momentum,
        normalize_input=self.normalize_input,
        mean_field_factor=self.mean_field_factor,
        ridge_penalty=self.ridge_penalty,
        steps_per_epoch=self.steps_per_epoch)

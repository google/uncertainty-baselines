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

"""Tests for TransformerGaussianProcess."""

import dataclasses

from absl.testing import absltest
import flax
import jax
import jax.numpy as jnp
import numpy as np
import t5x.examples.t5.network as t5_network
from uncertainty_baselines.models import t5_gp


class T5Test(absltest.TestCase):

  def setUp(self):
    super().setUp()

    batch, max_decode_len, input_len, vocab_size, emb_dim = 2, 3, 5, 10, 4
    config = t5_network.T5Config(
        vocab_size=vocab_size,
        emb_dim=emb_dim,
        num_heads=2,
        num_encoder_layers=1,
        num_decoder_layers=1,
        head_dim=2,
        mlp_dim=8)
    self.module = t5_gp.TransformerGaussianProcess(
        config=config,
        use_gp_layer=True,
        mean_field_factor=2.,
        steps_per_epoch=2)
    self.data = {
        'encoder_input_tokens':
            jax.random.randint(
                jax.random.PRNGKey(0), (batch, input_len), 0, vocab_size),
        'decoder_input_tokens':
            jax.random.randint(
                jax.random.PRNGKey(1), (batch, max_decode_len), 0, vocab_size),
        'decoder_target_tokens':
            jax.random.randint(
                jax.random.PRNGKey(2), (batch, max_decode_len), 0, vocab_size),
    }
    self.init_variables = self.module.init(
        jax.random.PRNGKey(3), **self.data, enable_dropout=False)

  def test_not_use_gp_layer(self):
    # Assert that variables are the same as deterministic ones.
    module = dataclasses.replace(self.module, use_gp_layer=False)
    init_variables = module.init(
        jax.random.PRNGKey(0), **self.data, enable_dropout=False)
    # Get upstream init variables.
    upstream_module = t5_network.Transformer(config=module.config)
    expected_variables = upstream_module.init(
        jax.random.PRNGKey(0), **self.data, enable_dropout=False)
    jax.tree_util.tree_map(np.testing.assert_allclose, init_variables,
                           expected_variables)

  def test_transformer_train(self):
    cfg = self.module.config
    target_shape = self.data['decoder_target_tokens'].shape
    logits, variables = self.module.apply(
        self.init_variables,
        **self.data,
        enable_dropout=True,
        rngs={'dropout': jax.random.PRNGKey(42)},
        mutable='intermediates')
    # Flax nn.Module.sow collects intermediate values in a tuple,
    # so we need to use `v[0]` to get the desired value.
    out = {k: v[0] for k, v in variables['intermediates']['decoder'].items()}
    self.assertEqual(out['pre_logits'].shape, target_shape + (cfg.emb_dim,))
    self.assertEqual(logits.shape, target_shape + (cfg.vocab_size,))
    np.testing.assert_allclose(out['logits'], logits)

  def test_transformer_eval(self):
    cfg = self.module.config
    target_shape = self.data['decoder_target_tokens'].shape
    logits, variables = self.module.apply(
        self.init_variables,
        **self.data,
        enable_dropout=False,
        mutable='intermediates')
    out = {k: v[0] for k, v in variables['intermediates']['decoder'].items()}
    self.assertEqual(out['pre_logits'].shape, target_shape + (cfg.emb_dim,))
    self.assertEqual(logits.shape, target_shape + (cfg.vocab_size,))
    np.testing.assert_allclose(out['logits'].shape, logits.shape)

    # Make sure that when evaluating a gp module, the returned logits
    # and out['logits'] are different.
    self.assertFalse(jnp.allclose(logits, out['logits'], rtol=0.1))
    self.assertEqual(out['covmat'].shape, target_shape)

    # Check if covmat is batch-compatible.
    for i in range(target_shape[0]):
      # Get a slice of data at index `i`.
      data_i = {k: v[i:i + 1] for k, v in self.data.items()}
      covmat_i = self.module.apply(
          self.init_variables,
          **data_i,
          enable_dropout=False,
          mutable='intermediates')[1]['intermediates']['decoder']['covmat'][0]
      self.assertEqual(covmat_i.shape, data_i['decoder_target_tokens'].shape)
      # Check if the covmat at data_i is the same as the original covmat_i.
      np.testing.assert_allclose(covmat_i[0], out['covmat'][i], rtol=1E-5)

  def test_mutable_state_become_params(self):
    head_state_flatten = flax.traverse_util.flatten_dict(
        flax.core.unfreeze(
            self.init_variables['params']['decoder']['gp_head_state']))
    self.assertEqual(
        set(head_state_flatten),
        {('laplace_covariance', 'covmat_layer', 'precision_matrix'),
         ('random_features', 'hidden_layer', 'bias'),
         ('random_features', 'hidden_layer', 'kernel'), ('step',)})

  def test_mutable_state_grad_zeros(self):
    # This test ensures that no gradient w.r.t. mutable state leaked to the
    # logits output.

    def get_logits_sum(params):
      variables = flax.core.unfreeze(self.init_variables)
      variables['params'] = params
      logits = self.module.apply(
          self.init_variables,
          **self.data,
          enable_dropout=True,
          rngs={'dropout': jax.random.PRNGKey(42)})
      return logits.sum()

    params = self.init_variables['params']
    grads = jax.grad(get_logits_sum)(params)['decoder']['gp_head_state']
    jax.tree_util.tree_map(
        lambda g: np.testing.assert_allclose(g, np.zeros_like(g)), grads)

  def test_mutable_state_update_intermediates(self):
    _, variables = self.module.apply(
        self.init_variables,
        **self.data,
        enable_dropout=True,
        rngs={'dropout': jax.random.PRNGKey(42)},
        mutable='intermediates')
    state_new = variables['intermediates']['decoder']['gp_head_state_new'][0]
    state_flatten = flax.traverse_util.flatten_dict(
        flax.core.unfreeze(state_new))
    self.assertEqual(
        set(state_flatten),
        {('laplace_covariance', 'covmat_layer', 'precision_matrix'),
         ('random_features', 'hidden_layer', 'bias'),
         ('random_features', 'hidden_layer', 'kernel'), ('step',)})
    state = self.init_variables['params']['decoder']['gp_head_state']
    jax.tree_util.tree_map(np.testing.assert_allclose,
                           state_new['random_features'],
                           state['random_features'])
    np.testing.assert_allclose(state_new['step'], state['step'] + 1.)

  def test_reset_precision_matrix(self):
    steps_per_epoch = self.module.steps_per_epoch
    params = {'params': self.init_variables['params']}
    state_first_step = None
    for i in range(steps_per_epoch + 1):
      j = i % steps_per_epoch
      data_i = {k: v[j:j + 1] for k, v in self.data.items()}
      _, variables = self.module.apply(
          params,
          **data_i,
          enable_dropout=True,
          rngs={'dropout': jax.random.PRNGKey(j)},
          mutable='intermediates')
      state_new = variables['intermediates']['decoder']['gp_head_state_new'][0]
      params = flax.core.unfreeze(params)
      params['params']['decoder']['gp_head_state'] = state_new
      params = flax.core.freeze(params)
      state_new = flax.core.unfreeze(state_new)
      state_new.pop('step')
      if i == 0:
        state_first_step = state_new
      elif i == steps_per_epoch:
        # If precision matrix is reset, we should expect to get the same
        # state_new as state_first_step, except for `step` value.
        jax.tree_util.tree_map(np.testing.assert_allclose, state_new,
                               state_first_step)
      else:
        self.assertFalse(
            jnp.allclose(
                state_new['laplace_covariance']['covmat_layer']
                ['precision_matrix'],
                state_first_step['laplace_covariance']['covmat_layer']
                ['precision_matrix'],
                rtol=0.1))


if __name__ == '__main__':
  absltest.main()

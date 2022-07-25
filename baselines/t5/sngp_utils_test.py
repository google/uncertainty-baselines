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

"""Tests for sngp_utils."""

import functools

from absl.testing import absltest
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import sngp_utils  # local file import from baselines.t5


class SNGPUtilsTest(absltest.TestCase):

  def setUp(self):
    super().setUp()

    class Head(nn.Module):

      @nn.compact
      def __call__(self, encoder_input, decoder_input, target, **kwargs):

        def head_state_init_fn(rng):
          return {
              'x': jax.random.normal(rng, encoder_input.shape[1:]),
              'y': jnp.array(3.)
          }

        self.param('gp_head_state', head_state_init_fn)
        head_state_new = {'x': jnp.arange(3.), 'y': jnp.array(1.)}
        self.sow('intermediates', 'gp_head_state_new', head_state_new)
        return jnp.zeros(target.shape + (3,))

    self.model = sngp_utils.EncoderDecoderGPModel(Head(), None, None, None)
    self.batch = {
        'encoder_input_tokens': jnp.array([[1, 2, 0], [2, 0, 1]]),
        'decoder_input_tokens': jnp.array([[2, 1, 0, 1], [0, 2, 1, 1]]),
        'decoder_target_tokens': jnp.array([[1, 2, 0, 2], [2, 1, 1, 0]])
    }
    input_shape = {k: v.shape for k, v in self.batch.items()}
    self.params = self.model.get_initial_variables(
        jax.random.PRNGKey(0), input_shape)['params']
    loss_fn = functools.partial(
        self.model.loss_fn, batch=self.batch, dropout_rng=jax.random.PRNGKey(1))
    self.grads = jax.grad(lambda x: loss_fn(x)[0])(self.params)

  def test_model_update_gp_state(self):
    _, state = self.model.module.apply({'params': self.params},
                                       self.batch['encoder_input_tokens'],
                                       self.batch['decoder_input_tokens'],
                                       self.batch['decoder_target_tokens'],
                                       mutable=['intermediates'])
    jax.tree_util.tree_map(np.testing.assert_allclose,
                           state['intermediates']['gp_head_state_new'][0],
                           flax.core.unfreeze(self.grads['gp_head_state']))

  def test_adafactor_update_gp_state(self):
    optimizer_def = sngp_utils.AdafactorGP()
    init_state = optimizer_def.init_state(self.params)
    new_params, _ = optimizer_def.apply_gradient(optimizer_def.hyper_params,
                                                 self.params, init_state,
                                                 self.grads)
    self.assertEqual(new_params, self.grads)


if __name__ == '__main__':
  absltest.main()

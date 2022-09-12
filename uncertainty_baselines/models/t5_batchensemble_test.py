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

"""Tests for T5 BatchEnsemble Transformer."""

from absl.testing import absltest
import jax
import t5x.examples.t5.network as t5_network
from uncertainty_baselines.models import t5_batchensemble


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
    self.module = t5_batchensemble.TransformerBE(
        config=config, ens_size=3, random_sign_init=0.5, be_decoder_layers=(0,))
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

  def test_logits_shape(self):
    cfg = self.module.config
    target_shape = self.data['decoder_target_tokens'].shape
    logits, _ = self.module.apply(
        self.init_variables,
        **self.data,
        enable_dropout=True,
        rngs={'dropout': jax.random.PRNGKey(42)},
        mutable=True)

    tiled_target_shape = (3 * target_shape[0],) + target_shape[1:]

    self.assertEqual(logits.shape, tiled_target_shape + (cfg.vocab_size,))


if __name__ == '__main__':
  absltest.main()

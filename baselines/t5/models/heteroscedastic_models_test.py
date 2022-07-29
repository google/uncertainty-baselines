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

"""Tests for Heteroscedastic EncoderDecoder model."""
from unittest import mock
from absl.testing import absltest
from absl.testing import parameterized

import jax.numpy as jnp
import seqio
from models import heteroscedastic_models  # local file import from baselines.t5


class EncoderDecoderClassifierModelTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.num_class = 2
    self.vocab_size = 4
    self.target_len = 1
    self.batch_size = 2

    self.vocab = seqio.PassThroughVocabulary(size=self.vocab_size)

  def test_score_batch_shape(self):
    """Checks if score_batch indeed returns class logits."""
    # Defines vocab ids for label tokens
    label_token_ids = jnp.array([0, 3])

    # Defines batch inputs.
    encoder_input_tokens = jnp.ones((self.batch_size, 3))
    # For this test, decoder input and target tokens are dummy values.
    decoder_input_tokens = jnp.array([[1, 2, 1, 0], [0, 1, 0, 2]])
    decoder_target_tokens = jnp.array([[1], [0]])
    decoder_loss_weights = jnp.array([[1], [0]])

    batch = {
        'encoder_input_tokens': encoder_input_tokens,
        'decoder_input_tokens': decoder_input_tokens,
        'decoder_target_tokens': decoder_target_tokens,
        'decoder_loss_weights': decoder_loss_weights
    }

    # Sets up mock model.
    logits_size = self.batch_size * self.target_len * self.vocab_size
    logits = jnp.arange(0, logits_size).reshape(
        (self.batch_size, self.target_len, self.vocab_size))
    params = {'foo': jnp.zeros(3)}

    mock_transformer = mock.Mock()
    mock_transformer.apply.return_value = logits
    mock_transformer.dtype = jnp.float32

    def mock_init(self):
      self.module = mock_transformer
      self.label_token_ids = label_token_ids
      self.temperature = 1.

    with mock.patch.object(
        heteroscedastic_models.EncoderDecoderHeteroscedasticClassifierModel,
        '__init__',
        new=mock_init):
      model = heteroscedastic_models.EncoderDecoderHeteroscedasticClassifierModel(
      )
      res = model.score_batch(params, batch)

    # Checks if score_batch() outputs class-specific logit scores.
    # expected_res = logits[:, :, label_token_ids]
    self.assertEqual(res.shape,
                     (self.batch_size, self.target_len, len(label_token_ids)))


if __name__ == '__main__':
  absltest.main()

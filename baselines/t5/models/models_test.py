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

"""Tests for models."""
import functools
from unittest import mock
from absl.testing import absltest
from absl.testing import parameterized

import jax
import jax.numpy as jnp
import numpy as np
import seqio
from t5x import decoding
from models import models  # local file import from baselines.t5


class EncoderDecoderClassifierModelTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.num_class = 2
    self.vocab_size = 4
    self.target_len = 1
    self.batch_size = 2

    self.vocab = seqio.PassThroughVocabulary(size=self.vocab_size)

  def test_get_label_token_ids(self):
    """Checks if get_class_token_ids returns correct result."""
    # A toy vocab whose token names and token IDs are the same integers.
    output_vocab = seqio.PassThroughVocabulary(size=self.vocab_size)

    label_tokens = jnp.array([0, 3])
    mock_transformer = mock.Mock()

    def mock_init(self):
      self.module = mock_transformer
      self.label_tokens = label_tokens
      self._output_vocabulary = output_vocab

    with mock.patch.object(
        models.EncoderDecoderClassifierModel, '__init__', new=mock_init):
      model = models.EncoderDecoderClassifierModel()
      label_token_ids = model._get_label_token_ids()

    # token_ids and label_tokens should be equal under PassThroughVocabulary.
    np.testing.assert_array_equal(label_token_ids, label_tokens)

  def test_get_label_token_ids_when_whole_vocab_used(self):
    """Checks if get_class_token_ids returns correct result."""
    # A toy vocab whose token names and token IDs are the same integers.
    output_vocab = seqio.PassThroughVocabulary(size=self.vocab_size)

    label_tokens = None
    mock_transformer = mock.Mock()

    def mock_init(self):
      self.module = mock_transformer
      self.label_tokens = label_tokens
      self._output_vocabulary = output_vocab

    with mock.patch.object(
        models.EncoderDecoderClassifierModel, '__init__', new=mock_init):
      model = models.EncoderDecoderClassifierModel()
      label_token_ids = model._get_label_token_ids()

    # token_ids and the whole vocabulary should be equal under
    # PassThroughVocabulary.
    np.testing.assert_array_equal(label_token_ids, jnp.array([0, 1, 2, 3]))

  def test_score_batch(self):
    """Checks if score_batch indeed returns class logits."""
    # Defines vocab ids for label tokens
    label_token_ids = jnp.array([0, 3])

    # Defines batch inputs.
    encoder_input_tokens = jnp.ones((self.batch_size, 3))
    # For this test, decoder input and target tokens are dummy values.
    decoder_input_tokens = jnp.array([[1, 2, 1, 0], [0, 1, 0, 2]])
    decoder_target_tokens = jnp.array([[
        1,
    ], [
        0,
    ]])
    decoder_loss_weights = jnp.array([[
        1,
    ], [
        0,
    ]])

    batch = {
        'encoder_input_tokens': encoder_input_tokens,
        'decoder_input_tokens': decoder_input_tokens,
        'decoder_target_tokens': decoder_target_tokens,
        'decoder_loss_weights': decoder_loss_weights
    }

    # Sets up mock model.
    logits = jnp.arange(0, 8).reshape(
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
        models.EncoderDecoderClassifierModel, '__init__', new=mock_init):
      model = models.EncoderDecoderClassifierModel()
      res = model.score_batch(params, batch)

    # Checks if score_batch() outputs class-specific logit scores.
    expected_res = logits[:, :, label_token_ids]
    np.testing.assert_allclose(res, expected_res, atol=1e-6)

  def test_partial_map(self):
    x = jnp.array([2.])
    batched_x = jnp.array([2., 2.])
    y = jnp.array([3., 4.])
    actual = models._partial_map(sum, (x, y))
    expected = jax.lax.map(sum, (batched_x, y))
    np.testing.assert_array_equal(actual, expected)


class EncoderDecoderBeamScoreModelTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.vocab_size = 4
    self.target_len = 2
    self.batch_size = 2

  @parameterized.named_parameters(('return_score', True),
                                  ('predict_only', False))
  def test_predict_batch(self, return_scores):
    # Defines batch inputs.
    encoder_input_tokens = jnp.ones((self.batch_size, 3))
    output_vocab = seqio.PassThroughVocabulary(size=self.vocab_size)

    # For this test, decoder input and target tokens are dummy values.
    decoder_input_tokens = jnp.array([[1, 2, 1, 0], [0, 1, 0, 2]])
    decoder_target_tokens = jnp.array([[1, 1], [0, 3]])
    decoder_loss_weights = jnp.array([[1, 2], [0, 0.5]])

    batch = {
        'encoder_input_tokens': encoder_input_tokens,
        'decoder_input_tokens': decoder_input_tokens,
        'decoder_target_tokens': decoder_target_tokens,
        'decoder_loss_weights': decoder_loss_weights
    }

    # Sets up mock model.
    logits = jnp.arange(0, 16).reshape(
        (self.batch_size, self.target_len, self.vocab_size))
    scores = jnp.arange(16, 32).reshape(
        (self.batch_size, self.target_len, self.vocab_size))

    # TODO(jereliu): Change to mock.create_autospec() after the transformer
    # implementation is stablized.
    mock_transformer = mock.Mock()
    mock_predict_batch_with_aux_fn = mock.Mock()
    mock_predict_batch_with_aux_fn.return_value = (logits, {'scores': scores})

    def mock_init(self):
      self.module = mock_transformer
      self._output_vocabulary = output_vocab
      self._decode_fn = functools.partial(decoding.beam_search, num_decodes=4)
      self.predict_batch_with_aux = mock_predict_batch_with_aux_fn

    with mock.patch.object(
        models.EncoderDecoderBeamScoreModel, '__init__', new=mock_init):
      model = models.EncoderDecoderBeamScoreModel()

    actual_output = model.predict_batch({}, batch, return_scores=return_scores)
    if return_scores:
      self.assertTupleEqual(actual_output, (logits, scores))
    else:
      np.testing.assert_array_equal(actual_output, logits)

  def test_score_batch(self):
    """Checks if score_batch indeed returns class logits."""
    # Defines batch inputs.
    label_token_ids = jnp.array([0, 3])
    encoder_input_tokens = jnp.ones((self.batch_size, 3))

    # For this test, decoder input and target tokens are dummy values.
    decoder_input_tokens = jnp.array([[1, 2, 1, 0], [0, 1, 0, 2]])
    decoder_target_tokens = jnp.array([[1, 1], [0, 3]])
    decoder_loss_weights = jnp.array([[1, 2], [0, 0.5]])

    batch = {
        'encoder_input_tokens': encoder_input_tokens,
        'decoder_input_tokens': decoder_input_tokens,
        'decoder_target_tokens': decoder_target_tokens,
        'decoder_loss_weights': decoder_loss_weights
    }

    # Sets up mock model.
    logits = jnp.arange(0, 16).reshape(
        (self.batch_size, self.target_len, self.vocab_size))
    intermediates = {}
    params = {'foo': jnp.zeros(3)}

    mock_transformer = mock.Mock()
    mock_transformer.apply.return_value = (logits, intermediates)
    mock_transformer.dtype = jnp.float32

    def mock_init(self):
      self.module = mock_transformer
      self.label_token_ids = label_token_ids
      self.temperature = 1.

    with mock.patch.object(
        models.EncoderDecoderBeamScoreModel, '__init__', new=mock_init):
      model = models.EncoderDecoderBeamScoreModel()
      res, res_intermediates = model.score_batch(
          params, batch, return_intermediates=True)

    # Checks if score_batch() outputs correct logits shape.
    np.testing.assert_array_equal(res.shape, (2,))

    # Checks intermediates with token entropies are correct shape.
    np.testing.assert_array_equal(
        res_intermediates['entropy']['token_entropy'].shape, (2, 2))


if __name__ == '__main__':
  absltest.main()

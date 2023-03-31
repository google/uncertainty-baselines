# coding=utf-8
# Copyright 2023 The Uncertainty Baselines Authors.
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

"""Tests for metrics."""

from absl.testing import absltest
from absl.testing import parameterized

import jax.numpy as jnp
import numpy as np
from t5.evaluation import test_utils

from data import metrics  # local file import from baselines.t5
from data.tasks import utils as task_utils  # local file import from baselines.t5


class BinaryClassificationTest(test_utils.BaseMetricsTest):

  def test_binary_classification(self):
    label_tokens = ('neg', 'pos')
    targets = ['neg', 'pos', 'neg', 'pos', 'neg']
    scores = jnp.ones((5, 1, 2)) * 0.5
    threshold = 0.49

    result_dict = metrics.binary_classification(
        targets,
        scores,
        label_tokens=label_tokens,
        prediction_threshold=threshold)

    expected_dict = {
        'accuracy': 40.0,
        'nll': 0.693147,
        'f1': 57.14285714,
        'auroc': 0.5,
        'auprc': 0.4,
        'auroc_temperature_adjusted': 0.5,
        'auprc_temperature_adjusted': 0.4,
        'ece': 0.1,
        'calib_auroc': 0.5,
        'calib_auprc': 0.6,
        'collab_auprc_1%': 0.4,
        'collab_auprc_2%': 0.4,
        'collab_auprc_5%': 0.4,
        'collab_auprc_10%': 0.4,
        'collab_auroc_1%': 0.5,
        'collab_auroc_2%': 0.5,
        'collab_auroc_5%': 0.5,
        'collab_auroc_10%': 0.5,
    }

    self.assertDictClose(result_dict, expected_dict, places=4)

  def test_binary_classification_input_shape(self):
    """Tests if function correctly detects irregular shapes."""
    targets = ['neg', 'pos', 'neg', 'pos', 'neg']
    scores = jnp.ones((5, 1, 2)) * 0.5
    label_tokens = ('neg', 'pos')
    threshold = 0.49

    scores_rank_2 = jnp.ones((5, 2)) * 0.5
    scores_extra_output_len = jnp.ones((5, 2, 2)) * 0.5
    scores_extra_extra_class = jnp.ones((5, 1, 3)) * 0.5
    label_tokens_extra_class = ('neg', 'pos', 'neutral')

    expected_error_msg = ('`label_tokens` should only contain 2 labels. Got 3: '
                          "('neg', 'pos', 'neutral')")
    with self.assertRaisesWithLiteralMatch(ValueError, expected_error_msg):
      _ = metrics.binary_classification(
          targets,
          scores,
          label_tokens=label_tokens_extra_class,
          prediction_threshold=threshold)

    expected_error_msg = (
        '`scores` should be a 3-D tensor with shape '
        '(batch_size, output_len, vocab_size). Got shape (5, 2)')
    with self.assertRaisesWithLiteralMatch(ValueError, expected_error_msg):
      _ = metrics.binary_classification(
          targets,
          scores_rank_2,
          label_tokens=label_tokens,
          prediction_threshold=threshold)

    expected_error_msg = (
        'For binary classification, the output len should be 1. Got 2')
    with self.assertRaisesWithLiteralMatch(ValueError, expected_error_msg):
      _ = metrics.binary_classification(
          targets,
          scores_extra_output_len,
          label_tokens=label_tokens,
          prediction_threshold=threshold)

    expected_error_msg = (
        'For binary classification, the num_class should be 2. Got 3')
    with self.assertRaisesWithLiteralMatch(ValueError, expected_error_msg):
      _ = metrics.binary_classification(
          targets,
          scores_extra_extra_class,
          label_tokens=label_tokens,
          prediction_threshold=threshold)


class SequenceClassificationTest(test_utils.BaseMetricsTest):

  def test_sequence_classification(self):
    label_tokens = ('tok1', 'tok2', 'tok3')
    targets = ['tok1 tok2', 'tok3 tok2']
    scores = jnp.array([[[0.8, 0.1, 0.1], [0.1, 0.8, 0.1]],
                        [[0.1, 0.7, 0.2], [0.1, 0.9, 0.1]]])

    result_entropy = metrics.sequence_classification(
        targets, scores, label_tokens=label_tokens,
        uncertainty_type='entropy', return_collab_accuracy=True)

    result_cond_entropy = metrics.sequence_classification(
        targets,
        scores,
        label_tokens=label_tokens,
        uncertainty_type='conditional_entropy',
        return_collab_accuracy=True)

    expected_result_entropy = {
        'accuracy': 50.0,
        'nll': 0.68580,
        'auroc': 1.0,
        'auprc': 1.0,
        'ece': 0.4963092803955078,
        'calib_auroc': 1.0,
        'calib_auprc': 1.0,
        'collab_acc_1%': 0.5,
        'collab_acc_2%': 0.5,
        'collab_acc_5%': 0.5,
        'collab_acc_10%': 0.5,
    }

    expected_result_cond_entropy = {
        'accuracy': 50.0,
        'nll': 0.73715,
        'auroc': 0.5,
        'auprc': 0.5,
        'ece': 0.14639335870742798,
        'calib_auroc': 0.5,
        'calib_auprc': 0.5,
        'collab_acc_1%': 0.5,
        'collab_acc_2%': 0.5,
        'collab_acc_5%': 0.5,
        'collab_acc_10%': 0.5,
    }

    self.assertDictClose(result_entropy, expected_result_entropy, places=4)
    self.assertDictClose(
        result_cond_entropy, expected_result_cond_entropy, places=4)

  def test_sequence_classification_input_shape(self):
    """Tests if function correctly detects irregular shapes."""
    label_tokens = ('tok1', 'tok2', 'tok3')
    targets = ['tok1 tok2', 'tok3 tok2']
    scores = jnp.array([[[0.8, 0.1, 0.1], [0.1, 0.8, 0.1]],
                        [[0.1, 0.7, 0.2], [0.1, 0.9, 0.1]]])

    scores_rank_2 = jnp.ones((3, 2)) * 0.5
    label_tokens_extra_class = ('tok1', 'tok2', 'tok3', 'tok4')
    illegal_uncertainty_type = 'mutual_information'
    targets_extra_output_len = ['tok1 tok2', 'tok3 tok2 tok1']

    expected_error_msg = (
        '`scores` should be a 3-D tensor with shape '
        '(batch_size, output_len, num_class). Got shape (3, 2)')
    with self.assertRaisesWithLiteralMatch(ValueError, expected_error_msg):
      _ = metrics.sequence_classification(
          targets, scores_rank_2, label_tokens=label_tokens)

    expected_error_msg = (
        'The number of classes in score tensor (3) does not match the number '
        'of label tokens (4).')
    with self.assertRaisesWithLiteralMatch(ValueError, expected_error_msg):
      _ = metrics.sequence_classification(
          targets, scores, label_tokens=label_tokens_extra_class)

    expected_error_msg = (
        'The uncertainty type must be one of ("entropy", "conditional_entropy")'
        '. Got "mutual_information".')
    with self.assertRaisesWithLiteralMatch(ValueError, expected_error_msg):
      _ = metrics.sequence_classification(
          targets,
          scores,
          label_tokens=label_tokens,
          uncertainty_type=illegal_uncertainty_type)

    expected_error_msg = (
        'Expects 2 tokens in the target string "tok3 tok2 tok1". Got 3: '
        '[\'tok3\', \'tok2\', \'tok1\'].')
    with self.assertRaisesWithLiteralMatch(ValueError, expected_error_msg):
      _ = metrics.sequence_classification(
          targets_extra_output_len, scores, label_tokens=label_tokens)


class SequenceClassificationBeamTest(test_utils.BaseMetricsTest,
                                     parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.vocab = task_utils.get_default_vocab()
    self.targets = ['A', 'H']
    self.beam_predictions_str = [['G', 'C', 'A'], ['C', 'H', 'A']]

    self.greedy_scores = jnp.array([0, 0])  # Ignored by beam functions.
    self.beam_predictions = [[[350], [205], [71]],
                             [[205], [454], [71]]]
    self.beam_probs = np.array([[[.3], [.3], [.4]],
                                [[.2], [.2], [.6]]])

    self.beam_scores = jnp.log(self.beam_probs)

    self.scores = [
        (self.greedy_scores[0], self.beam_predictions[0], self.beam_scores[0]),
        (self.greedy_scores[1], self.beam_predictions[1], self.beam_scores[1]),
    ]

  def test_extract_beam_results(self):
    beam_predictions, beam_scores = (
        metrics._extract_beam_results_from_scores(self.scores))

    self.assertListEqual(beam_predictions, self.beam_predictions)
    np.testing.assert_array_equal(np.array(beam_scores),
                                  np.array(self.beam_scores))

  @parameterized.parameters(
      ('all_beam', 3), ('non_top1_beam', 2), ('top1_beam', 1))
  def test_process_beam_by_type(self, beam_type, expected_len):
    pred_processed, score_processed = metrics._process_beam_by_type(
        self.beam_predictions, self.beam_scores, beam_type=beam_type)

    self.assertEqual(np.array(pred_processed).shape, (2, expected_len, 1))
    self.assertEqual(np.array(score_processed).shape, (2, expected_len, 1))

  @parameterized.parameters(
      ('all_beam', 2), ('non_top1_beam', 1), ('top1_beam', 1))
  def test_compute_beam_correctness(self, beam_type, num_correct):
    beam_correctness = metrics._compute_beam_correctness(
        self.targets, self.beam_predictions,
        vocab=self.vocab, beam_type=beam_type)

    self.assertEqual(np.sum(beam_correctness), num_correct)

  @parameterized.parameters(
      ('all_beam', 'probability', [1., 1.]),
      ('all_beam', 'margin', [0.1, 0.4]),
      # We will compute the true all_beam entropy explicitly.
      ('all_beam', 'entropy', [None, None]),
      ('non_top1_beam', 'probability', [0.6, 0.4]),
      ('non_top1_beam', 'entropy', [0.3, 0.2]),
      ('top1_beam', 'probability', [0.4, 0.6]))
  def test_compute_beam_uncertainty(
      self, beam_type, uncertainty_type, expected_confidence):

    beam_confidences = metrics._compute_beam_uncertainty(
        self.beam_predictions, self.beam_scores,
        beam_type, uncertainty_type)

    # Compute all_beam entropy explicitly due to lacking a convienient form.
    if beam_type == 'all_beam' and uncertainty_type == 'entropy':
      expected_confidence = np.sum(self.beam_probs * self.beam_scores,
                                   axis=(1, 2))
      expected_confidence = np.exp(expected_confidence)

    np.testing.assert_allclose(np.array(beam_confidences),
                               np.array(expected_confidence), atol=1e-3)


class Seq2seqTest(test_utils.BaseMetricsTest, parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.vocab = task_utils.get_default_vocab()
    self.targets = ['The hat is nice', 'The house is large']
    self.predictions = ['The bag is very nice', 'Doors are small']
    self.token_scores = [[-2, -3, -4, -7, -5, -1, -3, -10],
                         [-3, -4, -2, -3, -9, -2, -1, -11]]
    self.beam_scores = [-22, -12]

  def test_beam_scores_agree_with_token_scores(self):
    for score, beam_score, pred in zip(self.token_scores, self.beam_scores,
                                       self.predictions):
      seq_length = len(pred.split(' ')) + 1
      assert sum(score[:seq_length]) == beam_score

  def test_sequence_level_uncertainty_metrics(self):
    actual_metrics = metrics.seq2seq_uncertainty_metrics(
        self.targets, self.predictions, {'scores': self.beam_scores})
    expected_metrics = {
        'sequence_ece': 3e-06,
        'sequence_weighted_ece': 3.04e-6,
        'sequence_calib_auroc': 0.0,
        'sequence_calib_auprc': 1.0
    }
    self.assertDictClose(actual_metrics, expected_metrics, places=4)

  def test_token_level_uncertainty_metrics(self):
    actual_metrics = metrics.seq2seq_uncertainty_metrics(
        self.targets, self.predictions, {'scores': self.token_scores})
    expected_metrics = {
        'token_ece': 0.21678,
        'token_weighted_ece': 0.1838,
        'token_calib_auroc': 0.69048,
        'token_calib_auprc': 0.79448,
        'sequence_ece': 3e-06,
        'sequence_weighted_ece': 3.04e-6,
        'sequence_calib_auroc': 0.0,
        'sequence_calib_auprc': 1.0
    }
    self.assertDictClose(actual_metrics, expected_metrics, places=4)


if __name__ == '__main__':
  absltest.main()

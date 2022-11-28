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

"""Tests for utils."""

from absl.testing import absltest
from absl.testing import parameterized

import numpy as np
import t5.data

import utils  # local file import from baselines.t5

DEFAULT_SENT_STRS = [
    'That could cost him the <extra_id_99> chance to influence the outcome.',
    'the board <extra_id_0> mightn"t obtain an offer from bidders/'
]

DEFAULT_SENT_IDS = [[
    466, 228, 583, 376, 8, 32000, 1253, 12, 2860, 8, 6138, 5
], [8, 1476, 32099, 429, 29, 121, 17, 3442, 46, 462, 45, 6894, 588, 7, 87]]

DEFAULT_SCORES = [-0.031, -0.027]


class BeamPredictionsTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    # We want to make sure the function works correctly with this
    # sepecific vocabulary.
    self.vocab = t5.data.get_default_vocabulary()

    self.predictions_list = DEFAULT_SENT_IDS
    self.scores_list = DEFAULT_SCORES

    self.test_predictions = {
        'top_1': self.predictions_list[1],
        'top_2': self.predictions_list,
    }
    self.test_scores = {
        'dict_len_1': dict(scores=self.scores_list[:1]),
        'dict_len_2': dict(scores=self.scores_list[:2]),
        'list_len_1': self.scores_list[:1],
        'list_len_2': self.scores_list[:2],
    }

    self.expected_output_dict = {
        'prediction_0': DEFAULT_SENT_STRS[1].encode('utf-8'),
        'prediction_1': DEFAULT_SENT_STRS[0].encode('utf-8'),
        'prediction_0_ids': DEFAULT_SENT_IDS[1],
        'prediction_1_ids': DEFAULT_SENT_IDS[0]
    }

  @parameterized.named_parameters(
      ('top1_score_list', 'top_1', 'list', 1),
      ('topk_score_list', 'top_2', 'list', 2),
      ('top1_score_dict', 'top_1', 'dict', 1),
      ('topk_score_dict', 'top_2', 'dict', 2),
      ('top1_no_score', 'top_1', None, 1),
      ('top2_no_score', 'top_2', None, 2))
  def test_process_beam_prediction_outputs(
      self, prediction_type, score_type, beam_size):
    expected_dict = self.expected_output_dict
    output_beam_scores = score_type is not None

    # Adjusts function input.
    if not output_beam_scores:
      test_input = self.test_predictions[prediction_type]
    else:
      score_name = f'{score_type}_len_{beam_size}'
      test_input = (self.test_predictions[prediction_type],
                    self.test_scores[score_name])

    # Adjusts expected output according to test type.
    if beam_size == 1:
      expected_dict.pop('prediction_1')
      expected_dict.pop('prediction_1_ids')
    if output_beam_scores:
      expected_dict['beam_scores'] = self.scores_list[:beam_size][::-1]

    # Performs test.
    output_dict = utils.process_beam_prediction_outputs(
        test_input,
        vocabulary=self.vocab,
        output_beam_scores=output_beam_scores)

    self.assertEqual(output_dict, expected_dict)

  @parameterized.named_parameters(
      ('non_tuple_or_list_input', np.array(DEFAULT_SENT_IDS[0]), None,
       r'Output of predict_batch\(\) should be either a list or a tuple'),
      ('wrong_score_prediction_order', DEFAULT_SCORES, DEFAULT_SENT_IDS,
       r'output from predict_batch\(\) must be a list of integer token ids'),
      ('score_list_only', DEFAULT_SCORES, None,
       r'output from predict_batch\(\) must be a list of integer token ids'),
      ('score_dict_only', dict(scores=DEFAULT_SCORES), None,
       r'Output of predict_batch\(\) should be either a list or a tuple'),
      ('3-tuple', (DEFAULT_SENT_IDS, DEFAULT_SCORES, [0.92]), None,
       r'Output tuple of predict_batch\(\) should be a 2-tuple'),
      ('tuple_prediction_no_score', tuple(DEFAULT_SENT_IDS + [0, 3, 1]), None,
       r'Output tuple of predict_batch\(\) should be a 2-tuple'),
      ('tuple_prediction_with_score', tuple(DEFAULT_SENT_IDS), DEFAULT_SCORES,
       r'prediction output from predict_batch\(\) must be a list'),
      ('wrong_token_id_format', [[1., 0., 123., 2.]], [-1.],
       r'prediction output from predict_batch\(\) must be a list of integer'),
      ('wrong_score_format', DEFAULT_SENT_IDS, tuple(DEFAULT_SCORES),
       r'scores output from predict_batch\(\) should be either list, dict, or a jax device array'
      ),
      ('wrong_score_dict_key', DEFAULT_SENT_IDS, dict(score=DEFAULT_SCORES),
       r'score dictionary from predict_batch\(\) must contain key `scores`'),
      (
          'mismatch_len',
          DEFAULT_SENT_IDS,
          DEFAULT_SCORES + [0.95],
          r'Number of decoded samples contained in `beam_predictions` and '  # pylint:disable=implicit-str-concat
          r'`beam_scores` should equal'))
  def test_process_beam_prediction_outputs_type_error(
      self, test_prediction, test_score, expected_regex):
    """Tests the type check errors in process_beam_prediction_outputs."""
    if test_score is None:
      test_input = test_prediction
    else:
      test_input = (test_prediction, test_score)

    with self.assertRaisesRegex(ValueError, expected_regex):
      utils.process_beam_prediction_outputs(test_input,
                                            vocabulary=self.vocab)


if __name__ == '__main__':
  absltest.main()

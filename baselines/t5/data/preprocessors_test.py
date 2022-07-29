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

"""Tests for data.preprocessors."""
from absl.testing import absltest
from absl.testing import parameterized

import seqio
import tensorflow as tf

from data import preprocessors  # local file import from baselines.t5

mock = absltest.mock
assert_dataset = seqio.test_utils.assert_dataset


class ToxicCommentsPreprocessorsTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      ('zero', 0., '<extra_id_0>'), ('small float', 0.2, '<extra_id_0>'),
      ('on threshold', 0.7, '<extra_id_0>'),
      ('barely above threshold', 0.701, '<extra_id_1>'),
      ('one', 1., '<extra_id_1>'),
      ('above one', 1.1, '<extra_id_1>'))
  def test_toxicity_processorr_binary_classification(
      self, toxicity_score, expected_label):
    og_dataset = tf.data.Dataset.from_tensor_slices({
        'text': ['A comment.'],
        'toxicity': [toxicity_score]
    })

    output_observed = preprocessors.toxic_comments_preprocessor_binary_classification(
        og_dataset, threshold=0.7)
    output_expected = {'inputs': 'A comment.', 'targets': expected_label}

    assert_dataset(output_observed, output_expected)

  @parameterized.named_parameters(
      ('zero', 0., '0'), ('small float', 0.2, '0'), ('on threshold', 0.7, '0'),
      ('barely above threshold', 0.701, '1'), ('one', 1., '1'),
      ('above one', 1.1, '1'))
  def test_toxicity_processor_classification(self, toxicity_score,
                                             expected_label):
    og_dataset = tf.data.Dataset.from_tensor_slices({
        'text': ['A comment.'],
        'toxicity': [toxicity_score]
    })

    output_observed = preprocessors.toxic_comments_preprocessor_rank_classification(
        og_dataset, threshold=0.7)
    output_expected = {
        'inputs': 'A comment.',
        'choice1': '0',
        'choice2': '1',
        'label': int(expected_label)
    }

    assert_dataset(output_observed, output_expected)


class NaLUEPreprocessorsTest(tf.test.TestCase, parameterized.TestCase):

  def setUp(self):
    super().setUp()

    # Below intent names corresponds to <extra_id_X> for X from 0 to 8.
    self.intent_names = ('vertical_0', 'vertical_1', 'domain_0', 'domain_1',
                         'domain_2', 'intent_0', 'intent_1', 'intent_2',
                         'intent_3')
    self.intent_tokens = ('v_0', 'v_1', 'd_0', 'd_1', 'd_2', 'i_0', 'i_1',
                          'i_2', 'i_3')

    self.input_example = {
        'vertical': tf.constant('vertical_1'),
        'domain': tf.constant('domain_0'),
        'intent': tf.constant('intent_1'),
        'sentence': tf.constant('some utterance.')
    }

    self.unk_token = '???'

  @parameterized.named_parameters(('vertical_name', 'vertical_0', 'v_0'),
                                  ('domain_name', 'domain_2', 'd_2'),
                                  ('wrong_case_vocab', 'Intent_3', '???'),
                                  ('unknown_vocab', 'intent_4', '???'))
  def test_make_intent_to_token_map(self, key, expected_value):
    """Tests if make_intent_to_token_map makes a correct HashTable."""
    unk_token = '???'
    token_map = preprocessors.make_intent_to_token_map(
        self.intent_names, self.intent_tokens, unk_token=unk_token)

    map_key = tf.constant(key)
    self.assertEqual(token_map[map_key], expected_value)

  def test_tokenize_compositional_intents(self):
    token_map = preprocessors.make_intent_to_token_map(
        self.intent_names, self.intent_tokens, unk_token=self.unk_token)
    observed_output = preprocessors.tokenize_compositional_intents(
        self.input_example, token_map)

    self.assertEqual(observed_output, 'v_1 d_0 i_1')

  def test_nalue_preprocessors_classification(self):
    expected_output = {
        'inputs': tf.constant('some utterance.'),
        'targets': tf.constant('v_1 d_0 i_1')
    }

    input_dataset = tf.data.Dataset.from_tensor_slices({
        'vertical': ['vertical_1'],
        'domain': ['domain_0'],
        'intent': ['intent_1'],
        'sentence': ['some utterance.']
    })
    token_map = preprocessors.make_intent_to_token_map(
        self.intent_names, self.intent_tokens, unk_token=self.unk_token)
    observed_output = preprocessors.nalue_preprocessors_classification(
        input_dataset, token_map)

    assert_dataset(observed_output, expected_output)


class NLIPreprocessorsTest(tf.test.TestCase, parameterized.TestCase):
  """Tests for natural language inference preprocessors."""

  def setUp(self):
    super().setUp()
    # Set NLI as a binary classification problem. Only 'entailment' corresponds
    # to the positive class.
    self.label_names = ('entailment', 'neutral', 'contradiction',
                        'non-entailment')
    self.label_tokens = ('<extra_id_1>', '<extra_id_0>', '<extra_id_0>',
                         '<extra_id_0>')
    self.default_token = '<extra_id_0>'

    # Examples from MNLI or from HANS.
    self.mnli_example = {
        'hypothesis': b'The scientist supported the doctors.',
        'idx': 16399,
        'label': 1,
        'premise': b'The doctors supported the scientist.'
    }

    self.hans_example = {
        'gold_label':
            tf.constant('non-entailment'),
        'sentence1_binary_parse':
            tf.constant(
                '( ( The doctors ) ( ( supported ( the scientist ) ) . ) )'),
        'sentence2_binary_parse':
            tf.constant(
                '( ( The scientist ) ( ( supported ( the doctors ) ) . ) )'),
        'sentence1_parse':
            tf.constant(
                '(ROOT (S (NP (DT The) (NNS doctors)) (VP (VBD supported) (NP (DT the) (NN scientist))) (. .)))'
            ),
        'sentence2_parse':
            tf.constant(
                '(ROOT (S (NP (DT The) (NN scientist)) (VP (VBD supported) (NP (DT the) (NNS doctors))) (. .)))'
            ),
        'sentence1':
            tf.constant('The doctors supported the scientist .'),
        'sentence2':
            tf.constant('The scientist supported the doctors .'),
        'pairID':
            tf.constant('ex0'),
        'heuristic':
            tf.constant('lexical_overlap'),
        'subcase':
            tf.constant('ln_subject/object_swap'),
        'template':
            tf.constant('temp1')
    }

    self.intent_to_token_map = preprocessors.make_intent_to_token_map(
        self.label_names, self.label_tokens, unk_token=self.default_token)

  @parameterized.named_parameters(('mnli', 'mnli'), ('hans', 'hans'))
  def test_process_nli_inputs(self, data_type):
    ex = self.mnli_example if data_type == 'mnli' else self.hans_example
    expected_inputs = tf.constant(
        'premise: The doctors supported the scientist. '
        'hypothesis: The scientist supported the doctors.')
    observed_inputs = preprocessors.process_nli_inputs(ex, data_type=data_type)

    self.assertEqual(expected_inputs, observed_inputs)

  @parameterized.named_parameters(('mnli', 'mnli'), ('hans', 'hans'))
  def nli_preprocessors_classification_data_type_error(self, data_type):
    unknown_type = 'snli'
    ex = self.mnli_example if data_type == 'mnli' else self.hans_example

    with self.assertRaisesRegex(
        ValueError, 'data_type must be one of ("mnli", "hans"). Got "snli".'):
      _ = preprocessors.process_nli_inputs(ex, data_type=unknown_type)

  @parameterized.named_parameters(
      ('mnli_entailment', 'mnli', 0, '<extra_id_1>'),
      ('mnli_neutral', 'mnli', 1, '<extra_id_0>'),
      ('mnli_contradiction', 'mnli', 2, '<extra_id_0>'),
      ('hans_entailment', 'hans', 'entailment', '<extra_id_1>'),
      ('hans_nonentailment', 'hans', 'nonentailment', '<extra_id_0>'),
      ('hans_neutral', 'hans', 'neutral', '<extra_id_0>'),
      ('hans_contradiction', 'hans', 'contradiction', '<extra_id_0>'),
      ('hans_unknown_label', 'hans', 'presupposition', '<extra_id_0>'))
  def nli_preprocessors_classification(self, data_type, label, expected_token):
    if data_type == 'mnli':
      ex = self.mnli_example
      ex['label'] = tf.constant(label)
    else:
      ex = self.hans_example
      ex['gold_label'] = tf.constant(label)
    input_dataset = tf.data.Dataset.from_tensor_slices(ex)

    expected_output = {
        'inputs':
            tf.constant('premise: The doctors supported the scientist. '
                        'hypothesis: The scientist supported the doctors.'),
        'targets':
            tf.constant(expected_token)
    }

    observed_output = preprocessors.nli_preprocessors_classification(
        input_dataset,
        intent_to_token_map=self.intent_to_token_map,
        data_type=data_type)

    assert_dataset(observed_output, expected_output)

  @parameterized.named_parameters(('mnli', 'mnli'), ('hans', 'hans'))
  def nli_preprocessors_classification_input_error(self, data_type):
    unknown_type = 'snli'
    ex = self.mnli_example if data_type == 'mnli' else self.hans_example
    input_dataset = tf.data.Dataset.from_tensor_slices(ex)

    with self.assertRaisesRegex(
        ValueError, 'data_type must be one of ("mnli", "hans"). Got "snli".'):
      _ = preprocessors.nli_preprocessors_classification(
          input_dataset,
          intent_to_token_map=self.intent_to_token_map,
          data_type=unknown_type)

if __name__ == '__main__':
  absltest.main()

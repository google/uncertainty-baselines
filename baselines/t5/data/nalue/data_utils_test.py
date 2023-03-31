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

"""Tests for data_utils."""
import os

from absl.testing import absltest
import seqio
import t5.data

from data.nalue import data_utils  # local file import from baselines.t5

# Expected tokens under default t5 vocab (i.e., a standard SentencePiece
# vocabulary with 100 extra ids).
NALUE_INTENT_NAMES = data_utils.get_nalue_intent_names()
NALUE_INTENT_TOKENS_DEFAULT_T5_VOCAB = (
    'AI', 'A', 'ATM', 'Accent', 'Accept', 'Account', '<extra_id_0>', 'Activity',
    'firm', 'Age', 'Alarm', 'Answer', 'Arriv', 'Audio', 'Automat', 'Balance',
    'Bank', 'Trans', 'Basic', '<extra_id_1>', 'Bill', 'B', 'Book', 'T', 'U',
    'Bo', 'Bot', 'Business', 'By', 'Car', 'Check', 'Calcul', 'Calendar', 'Call',
    'Cal', 'Can', 'ist', 'Card', 'Char', 'Com', 'D', 'L', 'Not', 'S', 'Type',
    'Carr', 'Cash', 'Change', 'Pass', 'Per', 'Charge', 'Coffee', 'Con',
    'Contact', '<extra_id_2>', 'less', 'version', 'Cook', 'Country', 'Credit',
    'mit', 'core', 'Currency', '<extra_id_3>', 'Ex', 'Current', 'Date', 'Dec',
    'Definition', 'Delivery', 'Est', 'Deposit', 'Device', 'Dictionary', 'Dim',
    'Direction', 'Dis', 'Info', 'Distance', 'Don', 'K', 'Email', 'Event',
    '<extra_id_4>', '<extra_id_5>', 'Explain', 'Extra', 'F', 'Fee',
    '<extra_id_6>', 'Find', 'Flight', 'Flip', 'Food', 'For', 'Free', 'Fun',
    'Function', 'Game', 'Gas', 'General', 'Get', 'Good', 'Hello', 'Hobby',
    'Hotel', 'How', 'Bu', 'To', 'Identity', 'Improve', 'Income', '<extra_id_7>',
    'Ingredient', 'Insurance', 'Interest', 'International', 'Issue', 'Jo',
    'Jump', 'Language', 'Level', 'Light', 'Limit', 'Link', 'List', 'Location',
    'Lost', 'Maintenance', 'Manufacturer', 'Me', 'Meaning', 'Measure', 'Media',
    '<extra_id_8>', 'Meta', 'Mile', 'Movie', 'Music', 'Name', 'Navigation',
    'Neg', 'New', 'News', 'Next', 'Update', 'Nutrition', 'O', 'Off', 'Oil',
    'On', 'Order', '<extra_id_9>', 'Origin', 'P', 'Pay', 'day', 'Payment',
    'App', 'ending', 'Pet', 'Phone', 'Physical', 'Pin', 'Play', 'list', 'Plug',
    'Podcast', 'Post', 'Pre', 'Pressure', 'Product', '<extra_id_10>', 'Radio',
    'Rate', 'Receive', 'Recipe', 'Re', '<extra_id_11>', 'fund', '<extra_id_12>',
    'Remove', 'Rental', 'Repeat', 'Report', 'Request', 'Reservation', 'set',
    'Restaurant', '<extra_id_13>', 'Review', 'Rewards', 'Ride', 'Roll', 'Over',
    'Rou', 'Send', 'Set', 'Setting', 'Settings', 'Share', 'Shopping', 'Skip',
    'Small', 'Smart', 'Social', 'Spar', 'Speaker', 'Speed', 'Spe', 'Spend',
    'Statement', 'Status', 'Stock', 'Sto', 'Sub', 'Support', '<extra_id_14>',
    'Take', 'Tax', 'Termin', 'Text', 'Thanks', 'Time', '<extra_id_15>', 'Tire',
    'Top', 'Up', 'Traffic', 'Train', 'Transaction', 'Transfer', 'Translat',
    'Transport', 'Travel', '<extra_id_16>', 'Trip', '<extra_id_17>', 'Un', 'co',
    '<extra_id_18>', '<extra_id_19>', 'Used', 'User', 'Utility',
    '<extra_id_20>', 'Va', 'Ver', 'Very', 'Virtual', 'Work', 'Visa', 'Volume',
    'M', '<extra_id_21>', 'W', 'Weather', 'We', 'When', 'Whi', 'Why',
    '<extra_id_22>', 'mount', 'change')


mock = absltest.mock

TEST_DATA_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'test_data')


def build_sentencepiece_vocab(extra_ids=0):
  # The valid tokens for this vocab are:
  # [' ⁇ ','e', 'a', 's', 'o', 'i', 'l', 'test', 'th', ':', 'c', 'm', 'p',
  # 'te', 'w', 'at', 'w', 'h', 'd', 'n', 'r', 't', 'v',]
  return seqio.SentencePieceVocabulary(
      os.path.join(TEST_DATA_DIR, 'sentencepiece', 'sentencepiece.model'),
      extra_ids=extra_ids)


class DataUtilsTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.t5_vocab = t5.data.get_default_vocabulary()
    self.custom_tokens = [f'<extra_id_{i}>' for i in range(99)]

  def test_check_custom_token_validity_nonsingular_token(self):
    test_vocab = build_sentencepiece_vocab(extra_ids=100)

    # '<extra_id_100>' doesn't exist in standard SentencePiece vocab. However
    # since some of '<extra_id_100>''s letters do exist in the vocabulary tokens
    # of the test_vocab, it will be tokenized into multiple tokens.
    invalid_custom_tokens = ['<extra_id_99>', '<extra_id_100>']

    with self.assertRaisesRegex(
        ValueError, '"<extra_id_100>" cannot be tokenized into a single token'):
      _ = data_utils.check_custom_token_validity(
          invalid_custom_tokens, test_vocab)

  def test_check_custom_token_validity_out_of_vocab_token(self):
    # For a token that doesn't exist in the vocabulary tokens of the
    # SentencePiece vocab, it will be tokenized into ' ⁇ '. However, notice that
    # under the standard seqio.SentencePiece decoding algorithm tend to add a
    # placeholder before the unknown token, making this above behavior not easy
    # to reproduce. Therefore we use mock here.
    out_of_vocab_tokens = ['x', 'y', 'z']

    mock_vocab = mock.create_autospec(seqio.SentencePieceVocabulary)
    mock_vocab.encode.return_value = [3]
    mock_vocab.decode.return_value = ' ⁇ '

    with self.assertRaisesRegex(
        ValueError, '"x" is tokenized into a different token " ⁇ "'):
      _ = data_utils.check_custom_token_validity(
          out_of_vocab_tokens, mock_vocab)

  def test_make_intent_tokens_sample_intents(self):
    """Tests if make_intent_tokens correctly tokenizes a toy example."""
    intent_names = [
        'The', 'Quick', 'Brown', 'Fox', 'Jumps', 'Over', 'Lazy', 'Dog'
    ]
    intent_tokens = [
        'The', 'Quick', 'Brown', 'Fox', 'Jump', 'Over', 'La', 'Dog'
    ]

    intent_to_token, token_to_intent = data_utils.make_intent_tokens(
        intent_names, self.t5_vocab, self.custom_tokens)

    expected_intent_to_token = dict(zip(intent_names, intent_tokens))
    expected_token_to_intent = dict(zip(intent_tokens, intent_names))
    self.assertDictEqual(intent_to_token, expected_intent_to_token)
    self.assertDictEqual(token_to_intent, expected_token_to_intent)

  def test_make_intent_tokens_full_nalue_intents(self):
    """Checks if make_intent_tokens correctly tokenizes all NaLUE intents."""
    expected_tokens = NALUE_INTENT_TOKENS_DEFAULT_T5_VOCAB
    intent_to_token, token_to_intent = data_utils.make_intent_tokens(
        NALUE_INTENT_NAMES, self.t5_vocab, self.custom_tokens)

    observed_intents = list(token_to_intent.values())
    observed_tokens = list(intent_to_token.values())

    self.assertSequenceEqual(observed_intents, NALUE_INTENT_NAMES)
    self.assertSequenceEqual(observed_tokens, expected_tokens)

  def test_make_intent_tokens_invalid_custom_tokens(self):
    """Tests if make_intent_tokens detects invalid custom tokens."""
    invalid_custom_tokens = self.custom_tokens + ['invalid_token']
    intent_names = ['Card', 'CardInfo', 'CardLimit', 'Info', 'Limit']

    with self.assertRaisesRegex(
        ValueError, '"invalid_token" cannot be tokenized into a single token'):
      _ = data_utils.make_intent_tokens(intent_names, self.t5_vocab,
                                        invalid_custom_tokens)

  def test_make_intent_tokens_insufficient_custom_tokens(self):
    """Tests if make_intent_tokens correctly detects insufficient tokens."""
    intent_names = ['ABCD', 'ABC', 'AB', 'A']
    insufficient_custom_tokens = ['<extra_id_0>']

    with self.assertRaisesRegex(
        ValueError,
        'needs a custom token, however all custom_tokens are already used'):
      _ = data_utils.make_intent_tokens(intent_names, self.t5_vocab,
                                        insufficient_custom_tokens)

  def test_make_intent_tokens_duplicate_intents(self):
    # The below intent names contains two 'The'.
    duplicate_intent_names = [
        'The', 'Quick', 'Brown', 'Fox', 'Jumps', 'Over', 'The', 'Lazy', 'Dog'
    ]

    with self.assertRaisesRegex(ValueError,
                                '`intent_names` contains duplicates.'):
      _ = data_utils.make_intent_tokens(duplicate_intent_names, self.t5_vocab,
                                        self.custom_tokens)

if __name__ == '__main__':
  absltest.main()

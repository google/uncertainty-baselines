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

"""Data utils."""
from typing import Dict, List, Sequence, Tuple

from absl import logging
import numpy as np
import seqio

# A tuple of all possible intent label names to be used by NALUE task.
_NALUE_INTENT_NAMES = (
    'AIName', 'APR', 'ATM', 'Accent', 'Accept', 'Account', 'Activate',
    'Activity', 'Affirm', 'Age', 'Alarm', 'Answers', 'Arrival', 'Audiobook',
    'AutomateHowto', 'Balance', 'Bank', 'BankTransferChargeInfo',
    'BasicService', 'Basics', 'Bill', 'Blocked', 'Book', 'BookTaxi', 'BookUber',
    'Boss', 'BotInfo', 'Business', 'ByAppHowto', 'ByCardHowTo', 'ByCheckHowTo',
    'Calculator', 'Calendar', 'Call', 'Calorie', 'Cancel', 'CarAssistant',
    'CardAboutExpire', 'CardChargeInfo', 'CardCompromised', 'CardDamaged',
    'CardLost', 'CardNotWorking', 'CardSwallowed', 'CardType', 'CarryOn',
    'CashWithdraw', 'Change', 'ChangePassword', 'ChangePersonalDetail',
    'ChargedTwice', 'Coffee', 'Confirm', 'ContactAdd', 'ContactQuery',
    'ContactlessNotWorking', 'Conversion', 'CookTime', 'Country', 'CreditCard',
    'CreditLimit', 'CreditScore', 'Currency', 'Currency  ', 'CurrencyExchange',
    'CurrentLocation', 'DateTime', 'Declined', 'Definition', 'DeliveryEstimate',
    'DeliveryEstimateInfo', 'Deposit', 'Device', 'Dictionary', 'Dim',
    'Direction', 'DisposableCard', 'DisposableCardLimitInfo', 'Distance',
    'DontCare', 'DontKnow', 'Email', 'Event', 'Expiration', 'ExpirationDate',
    'Explain', 'ExtraCharge', 'Failed', 'FeeCharged', 'FeeInfo', 'FindPhone',
    'Flight', 'FlipCoin', 'Food', 'ForgottenPassword', 'Freeze', 'FunFact',
    'Function', 'Game', 'Gas', 'General', 'GetCard', 'Goodbye', 'Hello',
    'Hobby', 'Hotel', 'How', 'HowBusy', 'HowTo', 'Identity', 'ImproveInfo',
    'Income', 'Info', 'Ingredient', 'Insurance', 'InterestRateInfo',
    'InternationalFees', 'Issue', 'Joke', 'JumpStart', 'Language', 'Level',
    'Light', 'LimitInfo', 'Linking', 'List', 'Location', 'LostLuggage',
    'Maintenance', 'Manufacturer', 'Meal', 'MeaningOfLife', 'Measurement',
    'Media', 'Message', 'Meta', 'Mileage', 'Movie', 'Music', 'Name',
    'Navigation', 'Negate', 'NewCard', 'News', 'NextVacation', 'NotUpdated',
    'Nutrition', 'OOS', 'Off', 'OilChange', 'On', 'Order', 'OrderChecks',
    'Origin', 'PTO', 'Pay', 'Payday', 'Payment', 'PaymentApp', 'Pending',
    'Pets', 'Phone', 'PhysicalCard', 'Pin', 'Play', 'Playlist', 'PlugType',
    'Podcast', 'Post', 'Preference', 'Pressure', 'Productivity', 'Query',
    'Radio', 'RateInfo', 'ReceiveHowto', 'Recipe', 'Recommend', 'Redeem',
    'Refund', 'Reminder', 'Remove', 'RentalCar', 'Repeat', 'ReportFraud',
    'Request', 'Reservation', 'Reset', 'Restaurant', 'Reverted', 'Review',
    'Rewards', 'Ride', 'RollDice', 'RollOver', 'RoutingNumberInfo', 'Send',
    'Set', 'Setting', 'Settings', 'ShareLocation', 'Shopping', 'Skip',
    'SmallTalk', 'SmartHome', 'Social', 'SpareCard', 'Speaker', 'Speed',
    'Spelling', 'SpendingQuery', 'Statement', 'Status', 'Stock', 'StolenPhone',
    'Substitute', 'Support', 'Sync', 'TakeOut', 'Tax', 'Terminante', 'Text',
    'Thanks', 'TimeZone', 'Timer', 'Tire', 'TopUp', 'TopUpMethod', 'Traffic',
    'Train', 'TransactionQuery', 'Transfer', 'Translate', 'Transport',
    'TravelAlert', 'TravelNotification', 'TripPlanning', 'Type', 'Unable',
    'UnrecognizedTransaction', 'Up', 'Update', 'Used', 'UserName', 'Utility',
    'Vaccine', 'Vacuum', 'Verify', 'VeryIdentity', 'VirtualCard',
    'VirtualCardNotWorking', 'Visa', 'VolumeDown', 'VolumeMute', 'VolumeUp',
    'W2', 'Weather', 'Wemo', 'When', 'WhisperMode', 'Why', 'Work',
    'WrongAmount', 'WrongExchangeRate')


# A callable for other file to use to retrieve NaLUE intents (e.g., `task.py`)
get_nalue_intent_names = lambda: _NALUE_INTENT_NAMES


def make_intent_tokens(
    intent_names: Sequence[str], vocab: seqio.SentencePieceVocabulary,
    custom_tokens: Sequence[str]) -> Tuple[Dict[str, str], Dict[str, str]]:
  """Identify unique vocab tokens for intent label names.

  For each intent name, this function identifies a unique symbol from the
  SentencePieceVocabulary so every intent label can be decoded into a unique
  token. For example, for two intent names (`CardInfo`, `CardLimit`) that will
  be tokenized into [`Card`, `Info`] and [`Card`, `Limit`], we will use `Card`
  as the token for `CardInfo`, and `Limit` as the token for `CardLimit` so each
  intent correspond to a unique token. For an intent whose tokenized symbols are
  all already used by other intents, we will use an pre-reserved custom token
  from the vocabulary (e.g., extra ids `<extra_id_0>`) as its symbol.

  Args:
    intent_names: A sequence of names of the intent labels.
    vocab: A SentencePieceVocabulary that will be used for tokenizing the output
      sequence.
    custom_tokens: Custom tokens from the SentencePieceVocabulary. Usually these
      are the extra ids of the format "<extra_id_X>".

  Returns:
    intent_to_token: A mapping that converts intent names to tokens.
    token_to_intent: A mapping that converts tokens to intent names.

  Raises:
    ValueError: If `intent_names` contains duplicate names.
    ValueError: If the list of custom tokens are not part of vocab.
    ValueError: If the list of custom tokens are already exhausted when a new
      custom token is needed.
  """

  if len(intent_names) != len(np.unique(intent_names)):
    raise ValueError('`intent_names` contains duplicates. '
                     'Please make sure the intent names are unique.')

  # Creates a copy of the custom_tokens that is a mutable stack for later use.
  custom_tokens_available = list(custom_tokens)
  check_custom_token_validity(custom_tokens_available, vocab)

  # Assign each intent names an unique token.
  intent_to_token, token_to_intent = dict(), dict()
  for intent_name in intent_names:
    candidate_tokens = get_vocab_tokens(intent_name, vocab)

    for token in candidate_tokens:
      # Assigns a token to the intent if it (1) is un-used, (2) corresponds
      # to a single id under vocab.encode(), and (3) does not belong to a custom
      # special token.
      is_unused = token_to_intent.get(token, None) is None
      is_singular = len(vocab.encode(token)) == 1
      not_custom_token = token not in custom_tokens_available
      suitable_token_found = is_unused and is_singular and not_custom_token

      if suitable_token_found:
        intent_to_token[intent_name] = token
        token_to_intent[token] = intent_name
        logging.info('%s: assign token %s from candidates %s', intent_name,
                     token, candidate_tokens)
        break

    # Otherwise, assign a custom token to this intent.
    if not suitable_token_found:
      if not custom_tokens_available:
        raise ValueError(f'{intent_name} needs a custom token, however all '
                         'custom_tokens are already used.')
      custom_token = custom_tokens_available.pop(0)
      intent_to_token[intent_name] = custom_token
      token_to_intent[custom_token] = intent_name

      logging.info(
          '%s: all candidate tokens %s are either already used or '
          'does not correspond to a unique id, assigning to it a '
          'custom token: %s', intent_name, candidate_tokens, custom_token)

  return intent_to_token, token_to_intent


def get_vocab_tokens(input_str: str,
                     vocab: seqio.SentencePieceVocabulary) -> List[str]:
  """Splits an input string into a list of its SentencePiece tokens."""
  return [vocab.decode([i]) for i in vocab.encode(input_str)]


def check_custom_token_validity(custom_tokens: Sequence[str],
                                vocab: seqio.SentencePieceVocabulary):
  """Makes sure that all custom_tokens are part of the vocabulary.

  Args:
    custom_tokens: A list of special tokens that should correspond to unique
      tokens under the `vocab` object.
    vocab: A seqio.SentencePieceVocabulary object.

  Raises:
    ValueError: If any of the custom_tokens is tokenized into multiple tokens.
    ValueError: If any of the custom_tokens is tokenized into a different token.
  """
  for token in custom_tokens:
    token_recoded = get_vocab_tokens(token, vocab)
    if len(token_recoded) != 1:
      raise ValueError(
          f'custom token "{token}" cannot be tokenized into a single token: '
          f'{token_recoded}. It is not part of the valid vocabulary.')
    elif token != token_recoded[0]:
      raise ValueError(
          f'custom token "{token}" is tokenized into a different token '
          f'"{token_recoded[0]}". It is not part of the valid vocabulary.')

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

"""Preprocessors for T5 Tasks."""
from typing import Dict, Mapping, Sequence, Text, Union

import seqio
import tensorflow as tf

AUTOTUNE = tf.data.experimental.AUTOTUNE

NALUE_INPUT_NAME = 'sentence'
NALUE_OUTPUT_NAMES = ('vertical', 'domain', 'intent')

T5Text = Union[tf.Tensor, Text]


def toxic_comments_preprocessor_binary_classification(
    dataset: tf.data.Dataset,
    label_tokens: Sequence[str] = ('<extra_id_0>', '<extra_id_1>'),
    threshold: float = 0.5) -> tf.data.Dataset:
  """Converts a toxicity detection dataset to classification format.

  Toxicity detection task maps a sentence to a binary class of '<extra_id_0>'
  (non-toxic) and '<extra_id_1>' (toxic). A floating toxicity score (e.g., 0.3)
  will be converted to binary using a threshold.

  For example, a typical example might look like
  {
      'text': 'Some text.',
      'toxicity': 0.3,
  }

  This example would be transformed to
  {
      'inputs': 'Some text.',
      'targets': '<extra_id_0>',
  }

  Args:
    dataset: a tf.data.Dataset containing examples to process.
    label_tokens: Strings indicating the two classes of the binary labels. They
      should correspond to one of the extra tokens in SentencePiece tokenizer.
    threshold: the binary threshold for converting a continuous score (e.g.,
      0.3) to toxicity label.

  Returns:
    A mapped toxicity detection dataset in text2text format.
  """
  label_tokens = tf.constant(label_tokens)

  def _map_fn(ex: Mapping[str, tf.Tensor]) -> Mapping[str, tf.Tensor]:

    label_index = tf.cast(ex['toxicity'] > threshold, tf.int32)
    label_string = tf.gather(label_tokens, label_index)

    return {'inputs': ex['text'], 'targets': label_string}

  return dataset.map(_map_fn, num_parallel_calls=AUTOTUNE)


def toxic_comments_preprocessor_rank_classification(
    dataset: tf.data.Dataset,
    all_labels: Sequence[str] = ('0', '1'),
    input_feature: str = 'text',
    target_feature: str = 'toxicity',
    threshold: float = 0.5) -> tf.data.Dataset:
  """Reformats toxicity dataset to use rank classification preprocessor.

  Adapted from privacy.research.hark.t5.preprocessors.binary_classification.

  In this method, we convert examples having a `text` and a `toxicity` feature
  to a format that is subsequently consumed by a rank classification formatter.
  The `rank_classification_formatter` in T5 preprocessors then consumes
  the output features and creates two examples with `inputs` as input and each
  of `choice1` and `choice2` as targets. Each combination is then scored given
  the ground-truth `label`.

  Input data format:
    {
       'text': 'Some text.',
       'toxicity': 0.3,
    }

  This function will return example of the format:
    {
       'inputs': 'Some text.',
       'choice1': '0',
       'choice2': '1',
       ’label‘: 0
    }

  Args:
    dataset: A dataset to process.
    all_labels: Strings indicating the two classes of the binary labels.
    input_feature: Input feature name.
    target_feature: Target feature name.
    threshold: the binary threshold for converting a continuous score
      (e.g., 0.7) to toxicity label.

  Returns:
    A dataset preprocessed with the format listed above.
  """

  def _map_fn(ex: Mapping[str, tf.Tensor]) -> Mapping[str, tf.Tensor]:

    return {
        'inputs': ex[input_feature],
        'choice1': tf.constant(all_labels[0], dtype=tf.string),
        'choice2': tf.constant(all_labels[1], dtype=tf.string),
        'label': tf.cast(ex[target_feature] > threshold, dtype=tf.int32)
    }

  return dataset.map(_map_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)


def make_intent_to_token_map(intent_names: Sequence[str],
                             intent_tokens: Sequence[str],
                             unk_token: str) -> tf.lookup.StaticHashTable:
  """Creates a StaticHashTable that maps intent names to special tokens.

  Different from Dict, the StaticHashTable supports value-based key lookup based
  on a tf.Tensor string, which is important from working with tf.data-based
  input processors.

  Args:
    intent_names: A sequence of possible intent names.
    intent_tokens: A sequence of SentencePiece tokens corresponding to the
      intent names.
    unk_token: The token for unknown vocabulary values.

  Returns:
    A StaticHashTable that maps intent_names to special tokens in the format of
    '<extra_id_X>' where X is an integer.
  """
  mapping_initializer = tf.lookup.KeyValueTensorInitializer(
      keys=tf.constant(intent_names), values=tf.constant(intent_tokens))
  intent_to_token_mapping = tf.lookup.StaticHashTable(
      mapping_initializer, default_value=tf.constant(unk_token))

  return intent_to_token_mapping


def tokenize_compositional_intents(
    example: Mapping[str, tf.Tensor],
    intent_to_token_map: tf.lookup.StaticHashTable,
    separator: str = ' ') -> tf.string:
  """Converts an intent tuple into a string of special tokens.

  This function extracts intent names from an example with fields listed in
  NALUE_OUTPUT_NAMES (e.g., "vertical", "domain", "intent"), convert them into
  special tokens and then concatenantes them into a string. For example:

  Input data format:
    {
      'vertical': 'Answers',
      'domain': 'Dictionary',
      'intent': 'Translate'
    }

  We will then extract the intent names 'Answers', 'Dictionary', 'Translate'
  and convert them into special tokens using intent_to_token_map.

  Output data format:

    '<token_1> <token_2> <token_3>'. where `<token_X>` are the tokens
    corresponding to the names 'Answers', 'Dictionary', 'Translate' as defined
    in intent_to_token_map.


  Args:
    example: An input example with fields ('vertical', 'domain', 'intent').
    intent_to_token_map: A StaticHashTable that maps intent name to the
      corresponding special token.
    separator: The string separater that is used to join special tokens into a
      string.

  Returns:
    A string with format '<token_1> <token_2> <token_3>'.
  """
  intent_tokens = []

  # Extracts intent_name, and convert them into intent_tokens.
  for output_field in NALUE_OUTPUT_NAMES:
    intent_name = example[output_field]
    intent_token = intent_to_token_map[intent_name]

    intent_tokens.append(intent_token)

  return tf.strings.reduce_join(intent_tokens, separator=separator)


@seqio.map_over_dataset
def nalue_preprocessors_classification(
    example: Mapping[str, tf.Tensor],
    intent_to_token_map: tf.lookup.StaticHashTable) -> Dict[str, tf.Tensor]:
  """Preprocess NALUE examples into text2text format.

  Input data format:
    {
      'sentence': 'some sentence'.
      'vertical': 'Answers',
      'domain': 'Dictionary',
      'intent': 'Translate'
    }

  This function will convert the intent names in ('vertical', 'domain',
  'intent') into corresponding singular tokens for
  seqio.SentencePieceVocabulary() (e.g., a sentence piece that will be encoded
  into a single by seqio.SentencePieceVocabulary().encode()), and then
  concatenate them together as the output string.

  Output data format:
    {
      'inputs': 'some sentence'.
      'targets': '<token_1> <token_2> <token_3>',
    }

  In this way, after tokenization, the target sequence that the model should
  predict will be a sequence of three tokens (vertical_name_token,
  domain_name_token, intent_name_token).

  Args:
    example: A NALUE example with fields ('sentence', 'vertical', 'domain',
      'intent').
    intent_to_token_map: A mapping that maps intent names in the ('vertical',
      'domain', 'intent') fields to special tokens.

  Returns:
    A processed example.
  """
  inputs = example[NALUE_INPUT_NAME]
  targets = tokenize_compositional_intents(example, intent_to_token_map)

  return {'inputs': inputs, 'targets': targets}


def process_nli_inputs(example: Mapping[str, T5Text],
                       data_type: str = 'mnli') -> tf.Tensor:
  """Processes the sentence-pair inputs from MNLI and HANS examples.

  This function converts the sentence-pair inputs from NLI examples from MNLI or
  HANS datasets into a unified format. Specifically,

  Input data format from MNLI:
    {
      'premise': 'some sentence 1.',
      'hypothesis': 'some sentence 2.',
    }

  Input data format from HANS (notice the extra blank before period):
    {
      'sentence1': 'some sentence 1 .',
      'sentence2': 'some sentence 2 .',
    }

  Output:

   'premise: some sentence 1. hypothesis: some sentence 2.'

  Args:
    example: An input example with fields ('premise', 'hypothesis') if
      `data_type = 'mnli'`) or ('sentence1', 'sentence2') if `data_type='hans'`.
    data_type: The source dataset, can only be 'mnli' or 'hans'.

  Returns:
    A string in the format 'premise: {1ST_SENTECE}. hypothesis: {2ND_SENTECE}.'

  Raises:
    ValueError: If data_type is not one of ('mnli', 'hans').
  """
  supported_data_type = ('mnli', 'hans')
  if data_type not in supported_data_type:
    raise ValueError(
        f'data_type must be one of {supported_data_type}. Got "{data_type}".')

  # Extracts sentence from example.
  if data_type == 'mnli':
    premise = example['premise']
    hypothesis = example['hypothesis']
  else:
    # Remove the space before period.
    process_hans_str = lambda s: tf.strings.regex_replace(s, ' .$', '.')

    premise = process_hans_str(example['sentence1'])
    hypothesis = process_hans_str(example['sentence2'])

  # Concatenant sentences following t5.data.preprocessors.glue().
  strs_to_join = ['premise:', premise, 'hypothesis:', hypothesis]

  return tf.strings.join(strs_to_join, separator=' ')


@seqio.map_over_dataset
def nli_preprocessors_classification(
    example: Mapping[str, tf.Tensor],
    intent_to_token_map: tf.lookup.StaticHashTable,
    data_type: str = 'mnli',
    mnli_label_names: Sequence[str] = ('entailment', 'neutral', 'contradiction')
) -> Dict[str, tf.Tensor]:
  """Preprocess NLI examples from MNLI or HANS into classification format.

  Input data format (MNLI):
    {
      'premise': 'some sentence 1.'.
      'hypothesis': 'some sentence 2.'.
      "label": 1,
    }

  Input data format (HANS):
    {
      'sentence1': 'some sentence 1 .'.
      'sentence2': 'some sentence 2 .'.
      "gold_label": 'entailment',
    }

  Output data format:
    {
      'inputs': 'premise: {1ST_SENTENCE} hypothesis: {2ND_SENTENCE}',
      'targets': '<token_1>',
    }

  Args:
    example: An NLI example from MNLI or HANS.
    intent_to_token_map: A mapping that maps intent names ('entailment',
      'non-entailment', 'neutral', 'contradiction') into binary tokens. (i.e.,
      'entailment' into class 1, otherwise to class 0).
    data_type: The source dataset, can only be 'mnli' or 'hans'.
    mnli_label_names: a ordered list of MNLI label names corresponding to class
      index.

  Returns:
    A processed example.

  Raises:
    ValueError: If data_type is not one of ('mnli', 'hans').
  """
  sentence = process_nli_inputs(example, data_type=data_type)

  if data_type == 'mnli':
    label_name = tf.gather(mnli_label_names, example['label'])
  else:
    label_name = example['gold_label']
  label_token = intent_to_token_map[label_name]

  return {'inputs': sentence, 'targets': label_token}

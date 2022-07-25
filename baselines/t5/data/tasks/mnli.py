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

"""Task specification for Natural Language Inference (NLI) tasks.

Given a pair of sentences ('premise' and 'hypothesis'), the task of Natural
Language Inference (NLI) can be stated as a binary prediction problem answering
whether the premise entails the hypothesis.

The in-domain and OOD examples are taken from the Multi-Genre Natural Language
Inference (MNLI) dataset [1]. It contains a 'matched' split containing in-domain
examples across 9 domains, and a 'mismatched' split containing examples from
9 different domains.

The subpopulation shift datasets are coming from the HANS dataset [2]. It
contains 30k examples that contains three types of common surface-level
heuristics (lexical overlap, subsequence, constituent) that the machine learning
models are found to exploit.

## References
[1]: Williams, Adina, Nangia, Nikita and Bowman, Samuel. A Broad-Coverage
     Challenge Corpus for Sentence Understanding through Inference. In
     _Proceedings of the 2018 Conference of the North American Chapter of the
     Association for Computational Linguistics_, 2018.
     https://aclanthology.org/N18-1101/
[2]: R. Thomas McCoy, Ellie Pavlick, Tal Linzen. Right for the Wrong Reasons:
     Diagnosing Syntactic Heuristics in Natural Language Inference. In
     _Proceedings of the 57th Annual Meeting of the Association for
     Computational Linguistics_, 2019.
     https://aclanthology.org/P19-1334/

"""
import functools
import os
from typing import Sequence, Mapping, Union

import seqio
from t5.data import preprocessors as t5_preprocessors
import tensorflow as tf

import metrics as ub_metrics  # local file import from baselines.t5.data
import preprocessors as ub_preprocessors  # local file import from baselines.t5.data
import utils as task_utils  # local file import from baselines.t5.data.tasks

TaskRegistry = seqio.TaskRegistry

# Vocabulary.
MNLI_VOCAB = task_utils.get_default_vocab()

# Set the file path to the HANS dataset here.
_HANS_FILE_PATH = ''

_HANS_DEFAULT_DATA_TYPE = ('all',)
# 30 challenging subpopulations (arranged in the format of
# {heuristic} {subcases}) from HANS dataset.
_HANS_SUBPOP_DATA_TYPE = (
    'constituent_ce_adverb', 'constituent_ce_after_since_clause',
    'constituent_ce_conjunction', 'constituent_ce_embedded_under_since',
    'constituent_ce_embedded_under_verb', 'constituent_cn_adverb',
    'constituent_cn_after_if_clause', 'constituent_cn_disjunction',
    'constituent_cn_embedded_under_if', 'constituent_cn_embedded_under_verb',
    'lexical_overlap_le_around_prepositional_phrase',
    'lexical_overlap_le_around_relative_clause',
    'lexical_overlap_le_conjunction', 'lexical_overlap_le_passive',
    'lexical_overlap_le_relative_clause', 'lexical_overlap_ln_conjunction',
    'lexical_overlap_ln_passive', 'lexical_overlap_ln_preposition',
    'lexical_overlap_ln_relative_clause',
    'lexical_overlap_ln_subject_object_swap', 'subsequence_se_PP_on_obj',
    'subsequence_se_adjective', 'subsequence_se_conjunction',
    'subsequence_se_relative_clause_on_obj', 'subsequence_se_understood_object',
    'subsequence_sn_NP_S', 'subsequence_sn_NP_Z',
    'subsequence_sn_PP_on_subject', 'subsequence_sn_past_participle',
    'subsequence_sn_relative_clause_on_subject')

_HANS_TSV_FIELD_NAMES = ('gold_label', 'sentence1_binary_parse',
                         'sentence2_binary_parse', 'sentence1_parse',
                         'sentence2_parse', 'sentence1', 'sentence2', 'pairID',
                         'heuristic', 'subcase', 'template')

# Define HANS meta data to be used by the data provider. Note that the
# validation split is set to be the same as the test split for easy performance
# monitoring.
_HANS_FILE_PATTERNS = {}
_HANS_NUM_EXAMPLES = {}

for data_type in _HANS_DEFAULT_DATA_TYPE + _HANS_SUBPOP_DATA_TYPE:
  _HANS_FILE_PATTERNS[data_type] = {
      'train':
          os.path.join(_HANS_FILE_PATH, f'heuristics_train_{data_type}.tsv'),
      'validation':
          os.path.join(_HANS_FILE_PATH, f'heuristics_test_{data_type}.tsv'),
      'test':
          os.path.join(_HANS_FILE_PATH, f'heuristics_test_{data_type}.tsv')
  }

  _HANS_NUM_EXAMPLES[data_type] = {
      'train': 1000 if data_type in _HANS_SUBPOP_DATA_TYPE else 30000,
      'validation': 1000 if data_type in _HANS_SUBPOP_DATA_TYPE else 30000,
      'test': 1000 if data_type in _HANS_SUBPOP_DATA_TYPE else 30000,
  }

# Define label space and corresponding tokens. Notice that we are converting
# NLI into a binary prediction task. Therefore any label that is not
# 'entailment' will be treated as negative class and given the SentencePiece
# token '<extra_id_0>').
_LABEL_NAMES = ('entailment', 'negation', 'contradiction', 'non-entailment')
_LABEL_TOKENS = ('<extra_id_1>', '<extra_id_0>', '<extra_id_0>', '<extra_id_0>')

hash_init = tf.lookup.KeyValueTensorInitializer(_LABEL_NAMES, _LABEL_TOKENS)
_NLI_INTENT_TO_TOKEN_MAP = tf.lookup.StaticHashTable(
    hash_init, default_value='<extra_id_0>')


# Define register functions.
def register_hans_dataset(
    data_name: str,
    split_names: Sequence[str],
    file_patterns: Mapping[str, str],
    file_num_examples: Mapping[str, int],
    output_class_tokens: Sequence[str] = ('<extra_id_0>', '<extra_id_1>')):
  """Registers HANS datasets using TextLineDataSource provider."""
  # Filters file metadata by split_names.
  file_patterns = {k: v for k, v in file_patterns.items() if k in split_names}
  file_num_examples = {
      k: v for k, v in file_num_examples.items() if k in split_names
  }

  # Creates data source and registers dataset.
  data_source = seqio.dataset_providers.TextLineDataSource(
      file_patterns, skip_header_lines=1, num_input_examples=file_num_examples)
  hans_tsv_preprocessor = functools.partial(
      t5_preprocessors.parse_tsv, field_names=_HANS_TSV_FIELD_NAMES)
  hans_example_preprocessor = functools.partial(
      ub_preprocessors.nli_preprocessors_classification,
      intent_to_token_map=_NLI_INTENT_TO_TOKEN_MAP,
      data_type='hans')
  hans_metrics = functools.partial(
      ub_metrics.binary_classification, label_tokens=output_class_tokens)

  TaskRegistry.add(
      data_name,
      source=data_source,
      preprocessors=[
          hans_tsv_preprocessor,
          hans_example_preprocessor,
          seqio.preprocessors.tokenize,
          seqio.CacheDatasetPlaceholder(),
      ],
      postprocess_fn=None,
      metric_fns=[hans_metrics],
      output_features=task_utils.get_output_features_classification(
          vocab=MNLI_VOCAB),
  )


def register_mnli_dataset(
    data_name: str,
    tfds_name: str,
    tfds_splits: Union[Sequence[str], Mapping[str, str]],
    output_class_tokens: Sequence[str] = ('<extra_id_0>', '<extra_id_1>')):
  """Registers MNLI datasets using TFDSDataSource provider."""
  tfds_source = seqio.TfdsDataSource(tfds_name=tfds_name, splits=tfds_splits)
  mnli_preprocessor = functools.partial(
      ub_preprocessors.nli_preprocessors_classification,
      intent_to_token_map=_NLI_INTENT_TO_TOKEN_MAP,
      data_type='mnli',
      mnli_label_names=('entailment', 'neutral', 'contradiction'))
  mnli_metrics = functools.partial(
      ub_metrics.binary_classification, label_tokens=output_class_tokens)

  TaskRegistry.add(
      data_name,
      source=tfds_source,
      preprocessors=[
          mnli_preprocessor,
          seqio.preprocessors.tokenize,
          seqio.CacheDatasetPlaceholder(),
      ],
      postprocess_fn=None,
      metric_fns=[mnli_metrics],
      output_features=task_utils.get_output_features_classification(
          vocab=MNLI_VOCAB),
  )


# A callable to be used to define mixtures.
get_hans_subpopulation_types = lambda: _HANS_SUBPOP_DATA_TYPE

# Registers in-domain datasets.
# NOTE: The test split of glue/mnli does not have labels,
# so we will use validation source for it.
register_mnli_dataset(
    'mnli',
    tfds_name='glue/mnli:2.0.0',
    tfds_splits={
        'train': 'train',
        'validation': 'validation_matched',
        'test': 'validation_matched'
    })

# Register out-of-domain eval datasets.
register_mnli_dataset(
    'mnli_mismatched',
    tfds_name='glue/mnli:2.0.0',
    tfds_splits={
        'validation': 'validation_mismatched',
        'test': 'validation_mismatched'
    })

# Register sub-population eval datasets.
register_hans_dataset(
    'hans_all',
    split_names=('validation', 'test'),
    file_patterns=_HANS_FILE_PATTERNS['all'],
    file_num_examples=_HANS_NUM_EXAMPLES['all'])

for subpopulation_type in _HANS_SUBPOP_DATA_TYPE:
  register_hans_dataset(
      f'hans_{subpopulation_type}',
      split_names=('validation', 'test'),
      file_patterns=_HANS_FILE_PATTERNS[subpopulation_type],
      file_num_examples=_HANS_NUM_EXAMPLES[subpopulation_type])

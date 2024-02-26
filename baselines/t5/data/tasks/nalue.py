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

"""Task specification for Compositional Natural Language Evaluation (NaLUE)."""
import functools
import os
from typing import Sequence, Mapping, Dict, List, Union

import seqio
from t5.data import preprocessors as t5_preprocessors

from data import metrics as ub_metrics  # local file import from baselines.t5
from data import preprocessors as ub_preprocessors  # local file import from baselines.t5
from data.nalue import data_utils  # local file import from baselines.t5
from data.tasks import utils as task_utils  # local file import from baselines.t5

TaskRegistry = seqio.TaskRegistry

# Change below to the path to dataset files.
NALUE_FILE_PATH = ''

# Vocabulary.
NALUE_VOCAB = task_utils.get_default_vocab()
NALUE_VOCAB_EXTRA_IDS = [f'<extra_id_{i}>' for i in range(100)]

NALUE_TSV_FIELD_NAMES = ('orig_intent', 'sentence', 'vertical', 'domain',
                         'intent', 'data_source')

# Data provider.
NALUE_FILE_PATTERNS_IND = {
    'train': os.path.join(NALUE_FILE_PATH, 'ind_train.tsv'),
    'validation': os.path.join(NALUE_FILE_PATH, 'ind_valid.tsv'),
    'test': os.path.join(NALUE_FILE_PATH, 'ind_test.tsv')
}
NALUE_FILE_PATTERNS_OOS = {
    'train': os.path.join(NALUE_FILE_PATH, 'oos_train.tsv'),
    'validation': os.path.join(NALUE_FILE_PATH, 'oos_valid.tsv'),
    'test': os.path.join(NALUE_FILE_PATH, 'oos_test.tsv')
}
NALUE_FILE_PATTERNS_NEAR_OOS = {
    'train': os.path.join(NALUE_FILE_PATH, 'near_oos_train.tsv'),
    'validation': os.path.join(NALUE_FILE_PATH, 'near_oos_valid.tsv'),
    'test': os.path.join(NALUE_FILE_PATH, 'near_oos_test.tsv')
}
NALUE_FILE_PATTERNS_TAIL_INTENT = {
    'train': os.path.join(NALUE_FILE_PATH, 'tail_intent_train.tsv'),
    'validation': os.path.join(NALUE_FILE_PATH, 'tail_intent_valid.tsv'),
    'test': os.path.join(NALUE_FILE_PATH, 'tail_intent_test.tsv')
}
NALUE_FILE_PATTERNS_IND_AND_OOS = {
    split: [NALUE_FILE_PATTERNS_IND[split], NALUE_FILE_PATTERNS_OOS[split]]
    for split in ['train', 'validation', 'test']
}
NALUE_FILE_PATTERNS_IND_AND_NEAR_OOS = {
    split:
    [NALUE_FILE_PATTERNS_IND[split], NALUE_FILE_PATTERNS_NEAR_OOS[split]]
    for split in ['train', 'validation', 'test']
}

NALUE_NUM_EXAMPLES_IND = {'train': 29022, 'validation': 5022, 'test': 7752}
NALUE_NUM_EXAMPLES_OOS = {'train': 100, 'validation': 100, 'test': 1000}
NALUE_NUM_EXAMPLES_NEAR_OOS = {'train': 3437, 'validation': 571, 'test': 868}
NALUE_NUM_EXAMPLES_TAIL_INTENT = {'train': 3719, 'validation': 560, 'test': 740}
NALUE_NUM_EXAMPLES_IND_AND_OOS = {
    split: NALUE_NUM_EXAMPLES_IND[split] + NALUE_NUM_EXAMPLES_OOS[split]
    for split in ['train', 'validation', 'test']
}
NALUE_NUM_EXAMPLES_IND_AND_NEAR_OOS = {
    split: NALUE_NUM_EXAMPLES_IND[split] + NALUE_NUM_EXAMPLES_NEAR_OOS[split]
    for split in ['train', 'validation', 'test']
}


# Output labels. Defines the intent label names and its corresponding tokens for
# the t5 vocab.
intent_to_token_str_map, _ = data_utils.make_intent_tokens(
    data_utils.get_nalue_intent_names(),
    vocab=NALUE_VOCAB,
    custom_tokens=NALUE_VOCAB_EXTRA_IDS)

NALUE_INTENT_NAMES = list(intent_to_token_str_map.keys())
NALUE_INTENT_TOKENS = list(intent_to_token_str_map.values())

get_nalue_intent_tokens = (  # A callable to be used by gin model config.
    lambda: NALUE_INTENT_TOKENS)

intent_to_token_tensor_map = (  # A lookup HashTable to be used by preprocessor.
    ub_preprocessors.make_intent_to_token_map(
        NALUE_INTENT_NAMES,
        NALUE_INTENT_TOKENS,
        unk_token=NALUE_VOCAB.decode([NALUE_VOCAB.unk_id])))

NALUE_OOS_INTENT = 'OOS'
NALUE_OOS_INTENT_TOKEN = intent_to_token_str_map[NALUE_OOS_INTENT]
NALUE_OOS_INTENT_TOKEN_ID = NALUE_INTENT_TOKENS.index(NALUE_OOS_INTENT_TOKEN)


# Register datasets.
# TODO(jereliu): Add a post-processor converting token back to intent names.
# TODO(jereliu): Add metrics for evalutating token-level sequence uncertainty.
def register_nalue_dataset(data_name: str, split_names: Sequence[str],
                           file_patterns: Union[Mapping[str, str],
                                                Dict[str, List[str]]],
                           file_num_examples: Mapping[str, int]):
  """Registers NaLUE datasets."""
  # Filters file metadata by split_names.
  file_patterns = {k: v for k, v in file_patterns.items() if k in split_names}
  file_num_examples = {
      k: v for k, v in file_num_examples.items() if k in split_names
  }

  # Creates data source and registers dataset.
  data_source = seqio.dataset_providers.TextLineDataSource(
      file_patterns, skip_header_lines=1, num_input_examples=file_num_examples)

  # Computes token-level uncertainty metric for greedy prediction.
  nalue_top1_metrics = functools.partial(
      ub_metrics.sequence_classification,
      label_tokens=NALUE_INTENT_TOKENS,
      oos_token_id=NALUE_OOS_INTENT_TOKEN_ID)

  # Computes sequence-level uncertainty metric for top-K beam prediction.
  nalue_metric_fns = [nalue_top1_metrics]
  for beam_type in ('all_beam', 'non_top1_beam', 'top1_beam'):
    for uncertainty_type in ('probability', 'margin', 'entropy'):
      # Top-1 beam does not support margin and entropy metrics.
      if beam_type == 'top1_beam' and uncertainty_type != 'probability':
        continue
      # Makes sure to compute accuracy only once for each beam type.
      return_accuracy = uncertainty_type == 'probability'
      # Registers the calibration metric.
      topk_calibration_metric = functools.partial(
          ub_metrics.sequence_classification_beam_metrics,
          vocab=NALUE_VOCAB,
          beam_type=beam_type,
          uncertainty_type=uncertainty_type,
          return_accuracy=return_accuracy)

      # Registers the collaboration metric.
      topk_collaboration_metric = functools.partial(
          ub_metrics.topk_collaborative_accuracy,
          vocab=NALUE_VOCAB,
          beam_type=beam_type,
          uncertainty_type=uncertainty_type)

      nalue_metric_fns.extend(
          [topk_calibration_metric, topk_collaboration_metric])

  TaskRegistry.add(
      data_name,
      source=data_source,
      preprocessors=[
          functools.partial(
              t5_preprocessors.parse_tsv, field_names=NALUE_TSV_FIELD_NAMES),
          functools.partial(
              ub_preprocessors.nalue_preprocessors_classification,
              intent_to_token_map=intent_to_token_tensor_map),
          seqio.preprocessors.tokenize,
          seqio.CacheDatasetPlaceholder(),
      ],
      postprocess_fn=None,
      metric_fns=nalue_metric_fns,
      output_features=task_utils.get_output_features_classification(
          vocab=NALUE_VOCAB),
  )


# In-domain training dataset.
register_nalue_dataset(
    'nalue',
    split_names=('train', 'validation', 'test'),
    file_patterns=NALUE_FILE_PATTERNS_IND,
    file_num_examples=NALUE_NUM_EXAMPLES_IND)

# Out-of-scope evaluation dataset.
register_nalue_dataset(
    'nalue_standard_oos',
    split_names=('validation', 'test'),
    file_patterns=NALUE_FILE_PATTERNS_OOS,
    file_num_examples=NALUE_NUM_EXAMPLES_OOS)

register_nalue_dataset(
    'nalue_near_oos',
    split_names=('validation', 'test'),
    file_patterns=NALUE_FILE_PATTERNS_NEAR_OOS,
    file_num_examples=NALUE_NUM_EXAMPLES_NEAR_OOS)

# Subpopulation shift (i.e., tail-intent) evaluation dataset.
register_nalue_dataset(
    'nalue_tail_intent',
    split_names=('validation', 'test'),
    file_patterns=NALUE_FILE_PATTERNS_TAIL_INTENT,
    file_num_examples=NALUE_NUM_EXAMPLES_TAIL_INTENT)

# Out-of-scope mixed with In-domain evaluation dataset.
register_nalue_dataset(
    'nalue_ind_and_standard_oos',
    split_names=('validation', 'test'),
    file_patterns=NALUE_FILE_PATTERNS_IND_AND_OOS,
    file_num_examples=NALUE_NUM_EXAMPLES_IND_AND_OOS)

register_nalue_dataset(
    'nalue_ind_and_near_oos',
    split_names=('validation', 'test'),
    file_patterns=NALUE_FILE_PATTERNS_IND_AND_NEAR_OOS,
    file_num_examples=NALUE_NUM_EXAMPLES_IND_AND_NEAR_OOS)

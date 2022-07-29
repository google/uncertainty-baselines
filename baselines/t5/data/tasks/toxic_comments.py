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

"""Task specification for toxic comments."""
import functools
from typing import Iterable, Mapping, Optional, Union

import seqio

from t5.data import postprocessors as t5_postprocessors
from t5.data import preprocessors as t5_preprocessors
from t5.evaluation import metrics as t5_metrics

from data import metrics as ub_metrics  # local file import from baselines.t5
from data import preprocessors as ub_preprocessors  # local file import from baselines.t5
from data.tasks import utils as task_utils  # local file import from baselines.t5

TaskRegistry = seqio.TaskRegistry

DataSplit = Union[Iterable[str], Mapping[str, str]]

DEFAULT_OUTPUT_FEATURES = task_utils.get_output_features_classification()

LABEL_TOKENS = ('<extra_id_0>', '<extra_id_1>')
TOXIC_LABEL_THRESHOLD = 0.5

# TODO(jereliu): Move to `gin` file.
AUC_TEMPERATURES = (1e-3, 1e-2, 0.1, 0.5, 1., 1.5, 2., 2.5, 5., 7.5, 10.)


def _register_toxic_comments_ranking_task(
    task_name: str,
    tfds_name: str,
    toxicity_label_threshold: float = 0.5,
    preprocess_mode: str = 'eval',
    tfds_splits: Optional[DataSplit] = ('train', 'validation', 'test')):
  """Register a toxic comments task as a ranking problem.


  This function casts toxic comments as a binary prediction problem by taking
  advantage of the official `rank_classification` pre-/post-processors.
  Specifically, during preprocessing, it first converts the text2text problem to
  a ranking problem with two choices (using the `binary_classification`
  preprocessor), and then feed it to rank_classification_formatter.

  Args:
    task_name: Name of the seqio task to be added to TaskRegistry.
    tfds_name: Name of the TFDS dataset.
    toxicity_label_threshold: Numeric threshold between (0, 1) to be used for to
      convert the float-valued toxicity score into a binary label.
    preprocess_mode: A string, one of 'train', 'eval', or 'fewshot_eval')
      'train' produces only the correct example(s) based on the label value(s).
      'eval' produces an example for every possible class value, sequentially.
      'fewshot_eval': produces an example for every possible class value,
        batched together for each input example.
    tfds_splits: Data splits to be included for this task.
  """
  tfds_source = seqio.TfdsDataSource(tfds_name=tfds_name, splits=tfds_splits)
  toxicity_rank_classification_preprocessor = functools.partial(
      ub_preprocessors.toxic_comments_preprocessor_rank_classification,
      threshold=toxicity_label_threshold)
  toxicity_rank_classification_formatter = functools.partial(
      t5_preprocessors.rank_classification_formatter,
      inputs_formats='{inputs}',
      targets_formats=['{choice1}', '{choice2}'],
      mode=preprocess_mode)
  toxicity_rank_classification_metrics = functools.partial(
      t5_metrics.rank_classification, num_classes=2)

  TaskRegistry.add(
      task_name,
      source=tfds_source,
      preprocessors=[
          toxicity_rank_classification_preprocessor,
          toxicity_rank_classification_formatter, seqio.preprocessors.tokenize,
          seqio.CacheDatasetPlaceholder(),
          seqio.preprocessors.append_eos_after_trim
      ],
      metric_fns=[toxicity_rank_classification_metrics],
      postprocess_fn=t5_postprocessors.rank_classification,
      output_features=DEFAULT_OUTPUT_FEATURES)


def _register_toxic_comments_classification_task(
    task_name: str,
    tfds_name: str,
    label_tokens: Mapping[str, int],
    toxicity_label_threshold: float = 0.5,
    tfds_splits: Optional[DataSplit] = ('train', 'validation', 'test')):
  """Registers a toxic comments task as a binary classification problem.

  Args:
    task_name: Name of the seqio task to be added to TaskRegistry.
    tfds_name: Name of the TFDS dataset.
    label_tokens: A list of label tokens (e.g.,'<extra_id_0>') for the output
      classes. Shape (2,).
    toxicity_label_threshold: Numeric threshold between (0, 1) to be used for to
      convert the float-valued toxicity score into a binary label.
    tfds_splits: Data splits to be included for this task.
  """
  tfds_source = seqio.TfdsDataSource(tfds_name=tfds_name, splits=tfds_splits)
  toxicity_classification_preprocessor = functools.partial(
      ub_preprocessors.toxic_comments_preprocessor_binary_classification,
      label_tokens=label_tokens,
      threshold=toxicity_label_threshold)
  toxicity_classification_metrics = functools.partial(
      ub_metrics.binary_classification,
      label_tokens=label_tokens,
      prediction_threshold=toxicity_label_threshold,
      auc_temperatures=AUC_TEMPERATURES)

  # Note that for classification task, the preprocessor append_eos_after_trim is
  # not needed since eos token is redundant for the single-token classification
  # outputs.
  TaskRegistry.add(
      task_name,
      source=tfds_source,
      preprocessors=[
          toxicity_classification_preprocessor,
          seqio.preprocessors.tokenize,
          seqio.CacheDatasetPlaceholder(),
      ],
      metric_fns=[toxicity_classification_metrics],
      output_features=DEFAULT_OUTPUT_FEATURES)


_register_toxic_comments_task = functools.partial(
    _register_toxic_comments_classification_task,
    label_tokens=LABEL_TOKENS,
    toxicity_label_threshold=TOXIC_LABEL_THRESHOLD)

# Training data.
_register_toxic_comments_task(
    task_name='wikipedia_talk',
    tfds_name='wikipedia_toxicity_subtypes:0.3.1',
    tfds_splits=('train',),
)

# In-domain evaluation data. Note we will use test split for validation since
# wikipedia_talk data does not have a validation split.
_register_toxic_comments_task(
    task_name='wikipedia_talk_eval_only',
    tfds_name='wikipedia_toxicity_subtypes:0.3.1',
    tfds_splits={
        'validation': 'test',
        'test': 'test'
    },
)

# Out-of-domain evaluation data.
_register_toxic_comments_task(
    task_name='civil_comments_eval_only',
    tfds_name='civil_comments/CivilComments:1.1.2',
    tfds_splits=('validation', 'test'),
)

# Spurious correlation / tail generalization evaluation data, i.e., comments
# with mentions of sensitive social / racial attributes.
_register_toxic_comments_task(
    task_name='civil_comments_identity_eval_only',
    tfds_name='civil_comments/CivilCommentsIdentities:1.1.2',
    tfds_splits=('validation', 'test'),
)

# Data uncertainty evaluation data, i.e., highly ambiguous comments that
# contains different types of covert offensiveness (e.g., microaggression,
# sarcasim, emoticons, etc).
# Note this data only has positive label and does not have a validation split.
_register_toxic_comments_task(
    task_name='civil_comments_covert_eval_only',
    tfds_name='civil_comments/CivilCommentsCovert:1.1.2',
    tfds_splits={'validation': 'test'},
)

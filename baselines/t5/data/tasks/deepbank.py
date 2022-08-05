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

"""Task specification for DeepBank."""
import functools
import os

from typing import Any, Dict, Optional, Text, Sequence

import seqio

import t5.data
import tensorflow as tf

from data import metrics as ub_metrics  # local file import from baselines.t5


TaskRegistry = seqio.TaskRegistry

_DEFAULT_SHUFFLE_BUFFER_SIZE = 10000

_DEFAULT_V0_TRAIN_PATTERNS = []
_DEFAULT_V0_EVAL_PATTERNS = {}
_DEFAULT_V1_TRAIN_PATTERNS = []
_DEFAULT_V1_EVAL_PATTERNS = {}
_DEFAULT_V1_AUG_TRAIN_PATTERNS = []
_DEFAULT_V1_AUG_EVAL_PATTERNS = {}
_DEFAULT_OOD_PATTERNS_VALENCY = {}
_DEFAULT_OOD_PATTERNS_LANG10 = {}
_DEFAULT_OOD_PATTERNS_BROWN = {}
_DEFAULT_OOD_PATTERNS_CSLI = {}
_DEFAULT_OOD_PATTERNS_ESD = {}
_DEFAULT_OOD_PATTERNS_ESSAY = {}
_DEFAULT_OOD_PATTERNS_FRACAS = {}
_DEFAULT_OOD_PATTERNS_MRS = {}
_DEFAULT_OOD_PATTERNS_SEMCOR = {}
_DEFAULT_OOD_PATTERNS_TANAKA = {}
_DEFAULT_OOD_PATTERNS_TREC = {}
_DEFAULT_OOD_PATTERNS_VERBMOBIL = {}
_DEFAULT_OOD_PATTERNS_ECOMMERCE = {}
_DEFAULT_OOD_PATTERNS_LOGON = {}
_DEFAULT_OOD_PATTERNS_WIKI = {}
_DEFAULT_OOD_PATTERNS_BROWN_AUG = {}
_DEFAULT_OOD_PATTERNS_ECOMMERCE_AUG = {}
_DEFAULT_OOD_PATTERNS_ESSAY_AUG = {}
_DEFAULT_OOD_PATTERNS_LOGON_AUG = {}
_DEFAULT_OOD_PATTERNS_SEMCOR_AUG = {}
_DEFAULT_OOD_PATTERNS_TANAKA_AUG = {}
_DEFAULT_OOD_PATTERNS_VERBMOBIL_AUG = {}
_DEFAULT_OOD_PATTERNS_WIKI_AUG = {}

# User must provide train pattern path.
_DEFAULT_V0_TRAIN_PATTERNS = [
    '/train_patterns/v0_train.tfr*',
]
_DEFAULT_V1_TRAIN_PATTERNS = [
    '/train_patterns/v1_train.tfr*',
]
_DEFAULT_V1_AUG_TRAIN_PATTERNS = [
    '/train_patterns/v1_aug_train.tfr*',
]



def dataset_configs(  # pylint:disable=dangerous-default-value
    train_patterns: Sequence[Text] = _DEFAULT_V0_TRAIN_PATTERNS,
    eval_patterns: Dict[Text, Text] = _DEFAULT_V0_EVAL_PATTERNS,
    train_weights: Optional[Sequence[float]] = None) -> Dict[Text, Any]:
  """Returns configurable hyperparams."""
  # TODO(jereliu): Move default args to parsing_dataset function call.
  if not train_weights:
    train_weights = [1.0 / len(train_patterns)] * len(train_patterns)

  if len(train_patterns) != len(train_weights):
    raise ValueError(
        f'{train_patterns} should have the same length as {train_weights}')

  return dict(
      train_patterns=train_patterns,
      train_weights=train_weights,
      eval_patterns=eval_patterns)


# TODO(jereliu): Deprecate in favor of seqio.TFExampleDataSource.
def tfrecord_dataset(data_patterns: Sequence[Text],
                     shuffle_files: bool = False,
                     repeat: bool = False) -> tf.data.Dataset:
  """TFRecord dataset function."""
  data_paths = []
  for dp in data_patterns:
    data_paths.extend(sorted(tf.io.gfile.glob(dp)))
  tf.compat.v1.logging.info(f'Using examples from {data_paths}')
  assert data_paths
  dataset = tf.data.Dataset.from_tensor_slices(tf.constant(data_paths))
  if repeat:
    dataset = dataset.repeat()
  if shuffle_files:
    dataset = dataset.apply(
        # Note: This function is deprecated in favor of
        # tf.data.Dataset.interleave with the `num_parallel_calls` argument.
        tf.data.experimental.parallel_interleave(
            tf.data.TFRecordDataset,
            sloppy=True,  # Note: this may cause a lack of determinism.
            cycle_length=len(data_paths)))
  else:
    dataset = dataset.flat_map(tf.data.TFRecordDataset)

  features = dict(
      inputs=tf.io.FixedLenFeature([], tf.string),
      targets=tf.io.FixedLenFeature([], tf.string))

  dataset = dataset.map(
      functools.partial(tf.io.parse_single_example, features=features),
      num_parallel_calls=tf.data.experimental.AUTOTUNE)
  return dataset


def dataset_mixture(datasets: Sequence[tf.data.Dataset],
                    weights: Sequence[float]) -> tf.data.Dataset:
  """Returns a dataset mixture sampled with `weights`."""
  if len(datasets) == 1:
    return datasets[0]

  dataset = tf.data.experimental.sample_from_datasets(datasets, weights)
  dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

  return dataset


def parsing_dataset(
    split: Text,
    shuffle_files: bool = False,
    seed: Optional[int] = None,
    params: Optional[Dict[Text, Any]] = None) -> tf.data.Dataset:
  """Dataset function for semantic parsing."""
  # `seed` is required in the function signature but unused here.
  del seed

  if split == 'train':
    datasets = [
        tfrecord_dataset(
            data_patterns=[p], shuffle_files=shuffle_files, repeat=True)
        for p in params['train_patterns']
    ]
    return dataset_mixture(datasets, weights=params['train_weights'])

  return tfrecord_dataset(
      data_patterns=[params['eval_patterns'][split]],
      shuffle_files=shuffle_files,
      repeat=False)


# Register parsing tasks for deepbank using t5.data.TaskRegistry
# (i.e., a thin t5.data wrapper around seqio.TaskRegistry).
# Adapted from example_extrapolation codebase at:
# https://github.com/google/example_extrapolation/blob/master/tasks.py
# TODO(jereliu): Deprecate in favor of seqio.TaskRegistry.

# In-domain train and eval data.
deepbank_v0_config = dataset_configs(
    train_patterns=_DEFAULT_V0_TRAIN_PATTERNS,
    eval_patterns=_DEFAULT_V0_EVAL_PATTERNS)
deepbank_v1_config = dataset_configs(
    train_patterns=_DEFAULT_V1_TRAIN_PATTERNS,
    eval_patterns=_DEFAULT_V1_EVAL_PATTERNS)
deepbank_v1_aug_config = dataset_configs(
    train_patterns=_DEFAULT_V1_AUG_TRAIN_PATTERNS,
    eval_patterns=_DEFAULT_V1_AUG_EVAL_PATTERNS)


# Evaluation metrics.
def get_deepbank_metric_fns(data_version='v1'):
  """Returns metrics to be used for deepbank tasks."""
  return [
      functools.partial(ub_metrics.deepbank_metrics,
                        data_version=data_version),
      functools.partial(ub_metrics.deepbank_metrics_v2,
                        data_version=data_version),
      functools.partial(ub_metrics.deepbank_uncertainty_metrics,
                        data_version=data_version),
      ub_metrics.seq2seq_uncertainty_metrics]


# Defines canonical tasks.
t5.data.TaskRegistry.add(
    'deepbank',
    t5.data.Task,
    dataset_fn=functools.partial(parsing_dataset, params=deepbank_v0_config),
    splits=['train', 'validation', 'test'],
    text_preprocessor=None,
    metric_fns=[
        functools.partial(ub_metrics.deepbank_metrics, data_version='v0')
    ],
    shuffle_buffer_size=_DEFAULT_SHUFFLE_BUFFER_SIZE)

t5.data.TaskRegistry.add(
    'deepbank_1.1',
    t5.data.Task,
    dataset_fn=functools.partial(parsing_dataset, params=deepbank_v1_config),
    splits=['train', 'validation', 'test'],
    text_preprocessor=None,
    metric_fns=get_deepbank_metric_fns(data_version='v1'),
    shuffle_buffer_size=_DEFAULT_SHUFFLE_BUFFER_SIZE)

t5.data.TaskRegistry.add(
    'deepbank_1.1_aug',
    t5.data.Task,
    dataset_fn=functools.partial(
        parsing_dataset, params=deepbank_v1_aug_config),
    splits=['train', 'validation', 'test'],
    text_preprocessor=None,
    metric_fns=get_deepbank_metric_fns(data_version='v1'),
    shuffle_buffer_size=_DEFAULT_SHUFFLE_BUFFER_SIZE)

# Registers retrieval-augmented tasks.
RETRIEVAL_DATA_TYPES = ['random_retrieval_on_gold']
RETRIEVAL_DATA_SUBTYPES = [
    f'num_examplar={n}_depth={d}' for n in (1, 3, 5) for d in (1, 2, 3)]  # pylint:disable=g-complex-comprehension
RETRIEVAL_DATA_OOD_NAMES = [
    'brown', 'csli', 'ecommerce', 'esd', 'essay', 'fracas', 'logon', 'mrs',
    'semcor', 'tanaka', 'trec', 'verbmobil', 'wiki'
]

for retrieval_data_type in RETRIEVAL_DATA_TYPES:
  for retrieval_data_subtype in RETRIEVAL_DATA_SUBTYPES:
    retrieval_config = get_retrieval_augmented_data_config(
        data_type=retrieval_data_type, data_subtype=retrieval_data_subtype)

    # Replaces `=` sign since seqio task name does not allow it.
    subtype_name = retrieval_data_subtype.replace('=', '_')
    t5.data.TaskRegistry.add(
        f'deepbank_1.1_{retrieval_data_type}_{subtype_name}',
        t5.data.Task,
        dataset_fn=functools.partial(parsing_dataset, params=retrieval_config),
        splits=['train', 'validation', 'test'],
        text_preprocessor=None,
        metric_fns=get_deepbank_metric_fns(data_version='v1'),
        shuffle_buffer_size=_DEFAULT_SHUFFLE_BUFFER_SIZE)

for retrieval_data_type in RETRIEVAL_DATA_TYPES:
  for retrieval_data_subtype in RETRIEVAL_DATA_SUBTYPES:
    # Replaces `=` sign since seqio task name does not allow it.
    subtype_name = retrieval_data_subtype.replace('=', '_')
    for ood_name in RETRIEVAL_DATA_OOD_NAMES:
      retrieval_ood_config = get_retrieval_augmented_data_config(
          data_type=retrieval_data_type,
          data_subtype=retrieval_data_subtype,
          ood_data_name=ood_name)

      t5.data.TaskRegistry.add(
          f'deepbank_1.1_ood_{ood_name}_{retrieval_data_type}_{subtype_name}',
          t5.data.Task,
          dataset_fn=functools.partial(
              parsing_dataset, params=retrieval_ood_config),
          splits=['validation', 'test'],
          text_preprocessor=None,
          metric_fns=get_deepbank_metric_fns(data_version='v1'),
          shuffle_buffer_size=_DEFAULT_SHUFFLE_BUFFER_SIZE)

# OOD eval data on tail linguistic phenomenon.
ood_valency_config = dataset_configs(
    eval_patterns=_DEFAULT_OOD_PATTERNS_VALENCY)
ood_lang10_config = dataset_configs(
    eval_patterns=_DEFAULT_OOD_PATTERNS_LANG10)

t5.data.TaskRegistry.add(
    'deepbank_ood_valency',
    t5.data.Task,
    dataset_fn=functools.partial(parsing_dataset, params=ood_valency_config),
    splits=['test'],
    text_preprocessor=None,
    metric_fns=[ub_metrics.deepbank_metrics],
    shuffle_buffer_size=_DEFAULT_SHUFFLE_BUFFER_SIZE)

t5.data.TaskRegistry.add(
    'deepbank_ood_lang10',
    t5.data.Task,
    dataset_fn=functools.partial(parsing_dataset, params=ood_lang10_config),
    splits=['test'],
    text_preprocessor=None,
    metric_fns=[ub_metrics.deepbank_metrics],
    shuffle_buffer_size=_DEFAULT_SHUFFLE_BUFFER_SIZE)

# OOD eval data on out-of-domain generalization.
ood_config_patterns = dict(
    brown=_DEFAULT_OOD_PATTERNS_BROWN,
    csli=_DEFAULT_OOD_PATTERNS_CSLI,
    ecommerce=_DEFAULT_OOD_PATTERNS_ECOMMERCE,
    esd=_DEFAULT_OOD_PATTERNS_ESD,
    essay=_DEFAULT_OOD_PATTERNS_ESSAY,
    fracas=_DEFAULT_OOD_PATTERNS_FRACAS,
    logon=_DEFAULT_OOD_PATTERNS_LOGON,
    mrs=_DEFAULT_OOD_PATTERNS_MRS,
    semcor=_DEFAULT_OOD_PATTERNS_SEMCOR,
    tanaka=_DEFAULT_OOD_PATTERNS_TANAKA,
    trec=_DEFAULT_OOD_PATTERNS_TREC,
    verbmobil=_DEFAULT_OOD_PATTERNS_VERBMOBIL,
    wiki=_DEFAULT_OOD_PATTERNS_WIKI)

for data_name, data_pattern in ood_config_patterns.items():
  ood_config = dataset_configs(eval_patterns=data_pattern)
  t5.data.TaskRegistry.add(
      f'deepbank_ood_{data_name}',
      t5.data.Task,
      dataset_fn=functools.partial(parsing_dataset, params=ood_config),
      splits=['validation', 'test'],
      text_preprocessor=None,
      metric_fns=get_deepbank_metric_fns(data_version='v1'),
      shuffle_buffer_size=_DEFAULT_SHUFFLE_BUFFER_SIZE)

# OOD eval data on out-of-domain generalization with data augmentation.
ood_aug_config_patterns = dict(
    brown=_DEFAULT_OOD_PATTERNS_BROWN_AUG,
    ecommerce=_DEFAULT_OOD_PATTERNS_ECOMMERCE_AUG,
    essay=_DEFAULT_OOD_PATTERNS_ESSAY_AUG,
    logon=_DEFAULT_OOD_PATTERNS_LOGON_AUG,
    semcor=_DEFAULT_OOD_PATTERNS_SEMCOR_AUG,
    tanaka=_DEFAULT_OOD_PATTERNS_TANAKA_AUG,
    verbmobil=_DEFAULT_OOD_PATTERNS_VERBMOBIL_AUG,
    wiki=_DEFAULT_OOD_PATTERNS_WIKI_AUG)

for data_name, data_pattern in ood_aug_config_patterns.items():
  ood_config = dataset_configs(eval_patterns=data_pattern)
  t5.data.TaskRegistry.add(
      f'deepbank_ood_aug_{data_name}',
      t5.data.Task,
      dataset_fn=functools.partial(parsing_dataset, params=ood_config),
      splits=['validation', 'test'],
      text_preprocessor=None,
      metric_fns=get_deepbank_metric_fns(data_version='v1'),
      shuffle_buffer_size=_DEFAULT_SHUFFLE_BUFFER_SIZE)

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

"""Register parser tasks."""
import functools
import os

import seqio
import t5.data

from data import metrics as ub_metrics  # local file import from baselines.t5
from data.tasks import deepbank as utils  # local file import from baselines.t5

TaskRegistry = seqio.TaskRegistry

# T5 shuffle buffer size.
_DEFAULT_SHUFFLE_BUFFER_SIZE = 10000

_DEFAULT_SMCAL_TRAIN_PATTERNS = []
_DEFAULT_SMCAL_EVAL_PATTERNS = {}
_DEFAULT_MWOZ_TRAIN_PATTERNS = []
_DEFAULT_MWOZ_EVAL_PATTERNS = {}


# MTOP data (the English subset).
_DEFAULT_MTOP_TRAIN_PATTERNS = [
    '/cns/nm-d/home/jereliu/public/mtop/t5/train.tfr*',
]
_DEFAULT_MTOP_EVAL_PATTERNS = {
    'validation':
        '/cns/nm-d/home/jereliu/public/mtop/t5/dev.tfr*',
    'test':
        '/cns/nm-d/home/jereliu/public/mtop/t5/test.tfr*',
}

# MTOP data (English subset) with output string in penman format.
_DEFAULT_MTOP_PENMAN_TRAIN_PATTERNS = [
    '/cns/nm-d/home/lzi/public/mtop/t5/train.tfr*',
]
_DEFAULT_MTOP_PENMAN_EVAL_PATTERNS = {
    'validation':
        '/cns/nm-d/home/lzi/public/mtop/t5/dev.tfr*',
    'test':
        '/cns/nm-d/home/lzi/public/mtop/t5/test.tfr*',
}

# SNIPS data.
_DEFAULT_SNIPS_TRAIN_PATTERNS = [
    '/cns/nm-d/home/jereliu/public/snips/t5/train.tfr*',
]
_DEFAULT_SNIPS_EVAL_PATTERNS = {
    'validation':
        '/cns/nm-d/home/jereliu/public/snips/t5/dev.tfr*',
    'test':
        '/cns/nm-d/home/jereliu/public/snips/t5/test.tfr*',
}

# SNIPS data with output string in penman format.
_DEFAULT_SNIPS_PENMAN_TRAIN_PATTERNS = [
    '/cns/nm-d/home/lzi/public/snips/t5/train.tfr*',
]
_DEFAULT_SNIPS_PENMAN_EVAL_PATTERNS = {
    'validation':
        '/cns/nm-d/home/lzi/public/snips/t5/dev.tfr*',
    'test':
        '/cns/nm-d/home/lzi/public/snips/t5/test.tfr*',
}

# In-domain training and evaluation data.
smcalflow_config = utils.dataset_configs(
    train_patterns=_DEFAULT_SMCAL_TRAIN_PATTERNS,
    eval_patterns=_DEFAULT_SMCAL_EVAL_PATTERNS)
multiwoz_config = utils.dataset_configs(
    train_patterns=_DEFAULT_MWOZ_TRAIN_PATTERNS,
    eval_patterns=_DEFAULT_MWOZ_EVAL_PATTERNS)
mtop_config = utils.dataset_configs(
    train_patterns=_DEFAULT_MTOP_TRAIN_PATTERNS,
    eval_patterns=_DEFAULT_MTOP_EVAL_PATTERNS)
snips_config = utils.dataset_configs(
    train_patterns=_DEFAULT_SNIPS_TRAIN_PATTERNS,
    eval_patterns=_DEFAULT_SNIPS_EVAL_PATTERNS)

# In-domain penman training and evaluation data.
smcalflow_penman_config = utils.dataset_configs(
    train_patterns=_DEFAULT_SMCAL_PENMAN_TRAIN_PATTERNS,
    eval_patterns=_DEFAULT_SMCAL_PENMAN_EVAL_PATTERNS)
mtop_penman_config = utils.dataset_configs(
    train_patterns=_DEFAULT_MTOP_PENMAN_TRAIN_PATTERNS,
    eval_patterns=_DEFAULT_MTOP_PENMAN_EVAL_PATTERNS)
snips_penman_config = utils.dataset_configs(
    train_patterns=_DEFAULT_SNIPS_PENMAN_TRAIN_PATTERNS,
    eval_patterns=_DEFAULT_SNIPS_PENMAN_EVAL_PATTERNS)


# Evaluation metrics.
def get_dataflow_metric_fns(dataset_name='snips'):
  """Returns metrics to be used for deepbank tasks."""
  return [
      ub_metrics.seq2seq_metrics,
      ub_metrics.seq2seq_uncertainty_metrics,
      functools.partial(ub_metrics.dataflow_metrics,
                        dataset_name=dataset_name),
      functools.partial(ub_metrics.dataflow_uncertainty_metrics,
                        dataset_name=dataset_name)]


def get_retrieval_augmented_data_config(data_type='random_retrieval_on_gold',
                                        data_subtype='num_examplar=1_depth=1'):
  """Prepares retrieval-augmented data config."""
  data_root_path = '/cns/nm-d/home/lzi/public/smcalflow/t5/'
  data_full_path = os.path.join(data_root_path, data_type, data_subtype)
  train_patterns = [f'{data_full_path}/train.tfr*']
  eval_patterns = dict(
      validation=f'{data_full_path}/valid.tfr*',
      test=f'{data_full_path}/valid.tfr*')
  return utils.dataset_configs(
      train_patterns=train_patterns, eval_patterns=eval_patterns)


# Academic parsing tasks for data flow datasets.
# TODO(jereliu): Deprecate in favor of seqio.TaskRegistry.
t5.data.TaskRegistry.add(
    'smcalflow',
    t5.data.Task,
    dataset_fn=functools.partial(
        utils.parsing_dataset, params=smcalflow_config),
    splits={'train': 'train', 'validation': 'validation', 'test': 'validation'},
    text_preprocessor=None,
    metric_fns=[ub_metrics.seq2seq_metrics,
                ub_metrics.seq2seq_uncertainty_metrics],
    shuffle_buffer_size=_DEFAULT_SHUFFLE_BUFFER_SIZE)

t5.data.TaskRegistry.add(
    'smcalflow_penman',
    t5.data.Task,
    dataset_fn=functools.partial(
        utils.parsing_dataset, params=smcalflow_penman_config),
    splits={'train': 'train', 'validation': 'validation', 'test': 'validation'},
    text_preprocessor=None,
    metric_fns=get_dataflow_metric_fns(dataset_name='smcalflow'),
    shuffle_buffer_size=_DEFAULT_SHUFFLE_BUFFER_SIZE)

# Registers retrieval-augmented tasks.
RETRIEVAL_DATA_TYPES = [
    'random_retrieval_on_gold', 'oracle_retrieval_on_gold',
    'uncertain_retrieval_on_gold', 'oracle+uncertain_retrieval_on_gold'
]
RETRIEVAL_DATA_SUBTYPES = [
    f'num_examplar={n}_depth={d}' for n in (1, 3, 5) for d in (1, 2, 3)]  # pylint:disable=g-complex-comprehension
for retrieval_data_type in RETRIEVAL_DATA_TYPES:
  for retrieval_data_subtype in RETRIEVAL_DATA_SUBTYPES:
    retrieval_config = get_retrieval_augmented_data_config(
        data_type=retrieval_data_type, data_subtype=retrieval_data_subtype)
    # Replaces `+` sign since seqio task name does not allow it.
    retrieval_data_type_name = retrieval_data_type.replace('+', '_')

    # Replaces `=` sign since seqio task name does not allow it.
    subtype_name = retrieval_data_subtype.replace('=', '_')

    # Registers both a train-only task and a eval-only task.
    t5.data.TaskRegistry.add(
        f'smcalflow_penman_{retrieval_data_type_name}_{subtype_name}',
        t5.data.Task,
        dataset_fn=functools.partial(
            utils.parsing_dataset, params=retrieval_config),
        splits=['train'],
        text_preprocessor=None,
        metric_fns=get_dataflow_metric_fns(dataset_name='smcalflow'),
        shuffle_buffer_size=_DEFAULT_SHUFFLE_BUFFER_SIZE)

    # Eval-only data, can be used to evaluate the robustness of a
    # retieval-augmented model to different retrieval methods.
    t5.data.TaskRegistry.add(
        f'smcalflow_penman_eval_{retrieval_data_type_name}_{subtype_name}',
        t5.data.Task,
        dataset_fn=functools.partial(
            utils.parsing_dataset, params=retrieval_config),
        splits=['validation', 'test'],
        text_preprocessor=None,
        metric_fns=get_dataflow_metric_fns(dataset_name='smcalflow'),
        shuffle_buffer_size=_DEFAULT_SHUFFLE_BUFFER_SIZE)

t5.data.TaskRegistry.add(
    'multiwoz',
    t5.data.Task,
    dataset_fn=functools.partial(utils.parsing_dataset, params=multiwoz_config),
    splits={
        'train': 'train',
        'validation': 'validation',
        'test': 'test'
    },
    text_preprocessor=None,
    metric_fns=[ub_metrics.seq2seq_metrics,
                ub_metrics.seq2seq_uncertainty_metrics],
    shuffle_buffer_size=_DEFAULT_SHUFFLE_BUFFER_SIZE)

t5.data.TaskRegistry.add(
    'mtop',
    t5.data.Task,
    dataset_fn=functools.partial(utils.parsing_dataset, params=mtop_config),
    splits={
        'train': 'train',
        'validation': 'validation',
        'test': 'test'
    },
    text_preprocessor=None,
    metric_fns=[ub_metrics.seq2seq_metrics,
                ub_metrics.seq2seq_uncertainty_metrics],
    shuffle_buffer_size=_DEFAULT_SHUFFLE_BUFFER_SIZE)

t5.data.TaskRegistry.add(
    'mtop_penman',
    t5.data.Task,
    dataset_fn=functools.partial(
        utils.parsing_dataset, params=mtop_penman_config),
    splits={
        'train': 'train',
        'validation': 'validation',
        'test': 'test'
    },
    text_preprocessor=None,
    metric_fns=get_dataflow_metric_fns(dataset_name='mtop'),
    shuffle_buffer_size=_DEFAULT_SHUFFLE_BUFFER_SIZE)

t5.data.TaskRegistry.add(
    'snips',
    t5.data.Task,
    dataset_fn=functools.partial(utils.parsing_dataset, params=snips_config),
    splits={
        'train': 'train',
        'validation': 'validation',
        'test': 'test'
    },
    text_preprocessor=None,
    metric_fns=[ub_metrics.seq2seq_metrics,
                ub_metrics.seq2seq_uncertainty_metrics],
    shuffle_buffer_size=_DEFAULT_SHUFFLE_BUFFER_SIZE)

t5.data.TaskRegistry.add(
    'snips_penman',
    t5.data.Task,
    dataset_fn=functools.partial(
        utils.parsing_dataset, params=snips_penman_config),
    splits={
        'train': 'train',
        'validation': 'validation',
        'test': 'test'
    },
    text_preprocessor=None,
    metric_fns=get_dataflow_metric_fns(dataset_name='snips'),
    shuffle_buffer_size=_DEFAULT_SHUFFLE_BUFFER_SIZE)

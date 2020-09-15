# coding=utf-8
# Copyright 2020 The Uncertainty Baselines Authors.
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

"""Data loader for General Language Understanding Evaluation (GLUE) benchmark.

This data loader covers binary classification and regression datasets in the
GLUE benchmark. It includes 7 binary classification tasks:

   * Corpus of Linguistic Acceptability (COLA)
   * The Stanford Sentiment Treebank (SST2)
   * Microsoft Research Paraphrase Corpus (MRPC)
   * Quora Question Pairs (QQP)
   * Question NLI (QNLI)
   * Recognizing Textual Entailment (RTE)
   * Winograd NLI (WNLI),

and one regression task:

   * Semantic Textual Similarity Benchmark (STSb)

It does not include 3-way classification tasks, e.g., MNLI.

See https://gluebenchmark.com/ for further detail.


## References:
[1] Alex Wang, Amanpreet Singh, Julian Michael, Felix Hill, Omer Levy,
    Samuel Bowman. GLUE: A Multi-Task Benchmark and Analysis Platform for
    Natural Language Understanding.
    In _Proceedings of the 2018 EMNLP Workshop BlackboxNLP: Analyzing and
    Interpreting Neural Networks for NLP_, 2018.
    https://www.aclweb.org/anthology/W18-5446/
"""

from typing import Any, Dict

import tensorflow as tf
import tensorflow_datasets as tfds

from uncertainty_baselines.datasets import base

_FEATURE_NAME = {
    'glue/cola': ['sentence', None],
    'glue/sst2': ['sentence', None],
    'glue/mrpc': ['sentence1', 'sentence2'],
    'glue/qqp': ['question1', 'question2'],
    'glue/qnli': ['question', 'sentence'],
    'glue/rte': ['sentence1', 'sentence2'],
    'glue/wnli': ['sentence1', 'sentence2'],
    'glue/stsb': ['sentence1', 'sentence2'],
}


class _GLUEDataset(base.BaseDataset):
  """CIFAR dataset builder abstract class."""

  def __init__(self,
               name: str,
               batch_size: int,
               eval_batch_size: int,
               shuffle_buffer_size: int = None,
               num_parallel_parser_calls: int = 64,
               **unused_kwargs: Dict[str, Any]):
    """Create a GLUE tf.data.Dataset builder.

    Args:
      name: the name of this dataset.
      batch_size: the training batch size.
      eval_batch_size: the validation and test batch size.
      shuffle_buffer_size: the number of example to use in the shuffle buffer
        for tf.data.Dataset.shuffle().
      num_parallel_parser_calls: the number of parallel threads to use while
        preprocessing in tf.data.Dataset.map().
    """
    tfds_name = 'glue/' + name
    dataset_info = tfds.builder(tfds_name).info

    num_train_examples = dataset_info.splits['train'].num_examples
    num_validation_examples = dataset_info.splits['validation'].num_examples
    num_test_examples = dataset_info.splits['test'].num_examples

    super(_GLUEDataset, self).__init__(
        name=tfds_name,
        num_train_examples=num_train_examples,
        num_validation_examples=num_validation_examples,
        num_test_examples=num_test_examples,
        batch_size=batch_size,
        eval_batch_size=eval_batch_size,
        shuffle_buffer_size=shuffle_buffer_size,
        num_parallel_parser_calls=num_parallel_parser_calls)

  def _read_examples(self, split: base.Split) -> tf.data.Dataset:
    """Creates a dataset to be processed by _create_process_example_fn."""
    if split == base.Split.TEST:
      return tfds.load(self.name, split='test')
    elif split == base.Split.TRAIN:
      return tfds.load(self.name, split='train')
    else:
      return tfds.load(self.name, split='validation')

  def _create_process_example_fn(self, split: base.Split) -> base.PreProcessFn:
    """Create a pre-process function to return labels and sentence tokens."""

    def _example_parser(example: Dict[str, tf.Tensor]) -> Dict[str, Any]:
      """Parse snetences and labels from a serialized tf.train.Example."""
      idx = example['idx']
      label = example['label']

      # Read in sentences.
      text_a_name, text_b_name = _FEATURE_NAME[self.name]
      text_a = example[text_a_name]

      text_b = None
      if text_b_name:
        text_b = example[text_b_name]

      parsed_example = {
          'text_a': text_a,
          'text_b': text_b,
          'labels': label,
          'idx': idx
      }

      return parsed_example

    return _example_parser


class ColaDataset(_GLUEDataset):
  """COLA dataset builder class."""

  def __init__(self, **kwargs):
    super(ColaDataset, self).__init__(name='cola', **kwargs)


class Sst2Dataset(_GLUEDataset):
  """SST2 dataset builder class."""

  def __init__(self, **kwargs):
    super(Sst2Dataset, self).__init__(name='sst2', **kwargs)


class MrpcDataset(_GLUEDataset):
  """MRPC dataset builder class."""

  def __init__(self, **kwargs):
    super(MrpcDataset, self).__init__(name='mrpc', **kwargs)


class QqpDataset(_GLUEDataset):
  """QQP dataset builder class."""

  def __init__(self, **kwargs):
    super(QqpDataset, self).__init__(name='qqp', **kwargs)


class StsbDataset(_GLUEDataset):
  """STSb dataset builder class."""

  def __init__(self, **kwargs):
    super(StsbDataset, self).__init__(name='stsb', **kwargs)


class QnliDataset(_GLUEDataset):
  """QNLI dataset builder class."""

  def __init__(self, **kwargs):
    super(QnliDataset, self).__init__(name='qnli', **kwargs)


class RteDataset(_GLUEDataset):
  """RTE dataset builder class."""

  def __init__(self, **kwargs):
    super(RteDataset, self).__init__(name='rte', **kwargs)


class WnliDataset(_GLUEDataset):
  """WNLI dataset builder class."""

  def __init__(self, **kwargs):
    super(WnliDataset, self).__init__(name='wnli', **kwargs)


GlueDatasets = {
    'glue/cola': ColaDataset,
    'glue/sst2': Sst2Dataset,
    'glue/mrpc': MrpcDataset,
    'glue/qqp': QqpDataset,
    'glue/qnli': QnliDataset,
    'glue/rte': RteDataset,
    'glue/wnli': WnliDataset,
    'glue/stsb': StsbDataset,
}

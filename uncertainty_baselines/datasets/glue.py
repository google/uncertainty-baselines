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

from typing import Any, Dict, Optional

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


class _GlueDataset(base.BaseDataset):
  """GLUE dataset builder abstract class."""

  def __init__(
      self,
      name: str,
      split: str,
      shuffle_buffer_size: Optional[int] = None,
      num_parallel_parser_calls: int = 64,
      try_gcs: bool = False,
      download_data: bool = False,
      data_dir: Optional[str] = None,
      is_training: Optional[bool] = None,
  ):
    """Create a GLUE tf.data.Dataset builder.

    Args:
      name: the name of this dataset, 'glue/' will be prepended to get the
        dataset from TFDS.
      split: a dataset split, either a custom tfds.Split or one of the
        tfds.Split enums [TRAIN, VALIDAITON, TEST] or their lowercase string
        names.
      shuffle_buffer_size: the number of example to use in the shuffle buffer
        for tf.data.Dataset.shuffle().
      num_parallel_parser_calls: the number of parallel threads to use while
        preprocessing in tf.data.Dataset.map().
      try_gcs: Whether or not to try to use the GCS stored versions of dataset
        files.
      download_data: Whether or not to download data before loading.
      data_dir: Directory to read/write data, that is passed to the
              tfds dataset_builder as a data_dir parameter.
      is_training: Whether or not the given `split` is the training split. Only
        required when the passed split is not one of ['train', 'validation',
        'test', tfds.Split.TRAIN, tfds.Split.VALIDATION, tfds.Split.TEST].
    """
    tfds_name = 'glue/' + name
    dataset_builder = tfds.builder(
        tfds_name, try_gcs=try_gcs, data_dir=data_dir)
    super().__init__(
        name=tfds_name,
        dataset_builder=dataset_builder,
        split=split,
        is_training=is_training,
        shuffle_buffer_size=shuffle_buffer_size,
        num_parallel_parser_calls=num_parallel_parser_calls,
        fingerprint_key='idx',
        download_data=download_data)

  def _create_process_example_fn(self) -> base.PreProcessFn:
    """Create a pre-process function to return labels and sentence tokens."""

    def _example_parser(example: Dict[str, tf.Tensor]) -> Dict[str, Any]:
      """Parse sentences and labels from a serialized tf.train.Example."""
      text_a_name, text_b_name = _FEATURE_NAME[self.name]
      text_a = example[text_a_name]
      text_b = example[text_b_name] if text_b_name else None
      return {
          'text_a': text_a,
          'text_b': text_b,
          'labels': example['label'],
          'idx': example['idx'],
      }

    return _example_parser


class ColaDataset(_GlueDataset):
  """COLA dataset builder class."""

  def __init__(self, **kwargs):
    super().__init__(name='cola', **kwargs)


class Sst2Dataset(_GlueDataset):
  """SST2 dataset builder class."""

  def __init__(self, **kwargs):
    super().__init__(name='sst2', **kwargs)


class MrpcDataset(_GlueDataset):
  """MRPC dataset builder class."""

  def __init__(self, **kwargs):
    super().__init__(name='mrpc', **kwargs)


class QqpDataset(_GlueDataset):
  """QQP dataset builder class."""

  def __init__(self, **kwargs):
    super().__init__(name='qqp', **kwargs)


class StsbDataset(_GlueDataset):
  """STSb dataset builder class."""

  def __init__(self, **kwargs):
    super().__init__(name='stsb', **kwargs)


class QnliDataset(_GlueDataset):
  """QNLI dataset builder class."""

  def __init__(self, **kwargs):
    super().__init__(name='qnli', **kwargs)


class RteDataset(_GlueDataset):
  """RTE dataset builder class."""

  def __init__(self, **kwargs):
    super().__init__(name='rte', **kwargs)


class WnliDataset(_GlueDataset):
  """WNLI dataset builder class."""

  def __init__(self, **kwargs):
    super().__init__(name='wnli', **kwargs)


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

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

# Lint as: python3
"""Data loader for the Multi-Genre Natural Language Inference (MNLI) dataset.

MNLI corpus is a crowd-sourced collection of 433k sentence pairs annotated with
textual entailment labels (3 classes: entailment, contradiction and neutral).
The corpus covers a range of genres of spoken and written text from 5 domains:
fiction, government, telephone, travel, slate (i.e., popular magazine).

It also contains ~20k evaluation data from out-of-domain genre (9/11,
face-to-face, letters, oup, verbatim). This evaluation dataset can be used for
evaluating model robustness under distribution shift and for out-of-domain
detection.

See https://cims.nyu.edu/~sbowman/multinli/ and corpus paper for further detail.

## References:

[1] Adina Williams, Nikita Nangia, Samuel Bowman.
    A Broad-Coverage Challenge Corpus for Sentence Understanding through
    Inference.
    In _Proceedings of the 2018 Conference of the North American Chapter of
    the Association for Computational Linguistics_, 2018.
    https://www.aclweb.org/anthology/N18-1101/
"""

from typing import Any, Dict, Optional

import tensorflow as tf
import tensorflow_datasets as tfds

from uncertainty_baselines.datasets import base


class MnliDataset(base.BaseDataset):
  """Multi-NLI dataset builder class."""

  def __init__(
      self,
      batch_size: int,
      eval_batch_size: int,
      shuffle_buffer_size: int = None,
      num_parallel_parser_calls: int = 64,
      data_dir: Optional[str] = None,
      mode: str = 'matched',
      **unused_kwargs: Dict[str, Any]):
    """Create a GLUE tf.data.Dataset builder.

    Args:
      batch_size: the training batch size.
      eval_batch_size: the validation and test batch size.
      shuffle_buffer_size: the number of example to use in the shuffle buffer
        for tf.data.Dataset.shuffle().
      num_parallel_parser_calls: the number of parallel threads to use while
        preprocessing in tf.data.Dataset.map().
      data_dir: optional dir to save TFDS data to. If none then the local
        filesystem is used. Required for using TPUs on Cloud.
      mode: Type of data to import. If mode = "matched", import the in-domain
        data (glue/mnli_matched). If mode = "mismatched", import the
        out-of-domain data (glue/mnli_mismatched).
    """
    if mode not in ('matched', 'mismatched'):
      raise ValueError('"mode" must be either "matched" or "mismatched".'
                       'Got {}'.format(mode))
    self.mode = mode
    self.validation_split_name = 'validation_' + mode
    self.test_split_name = 'test_' + mode

    tfds_name = 'glue/mnli'
    dataset_info = tfds.builder(tfds_name).info

    num_train_examples = dataset_info.splits[
        'train'].num_examples if mode == 'matched' else 0
    num_validation_examples = dataset_info.splits[
        self.validation_split_name].num_examples
    num_test_examples = dataset_info.splits[self.test_split_name].num_examples

    super(MnliDataset, self).__init__(
        name=tfds_name,
        num_train_examples=num_train_examples,
        num_validation_examples=num_validation_examples,
        num_test_examples=num_test_examples,
        batch_size=batch_size,
        eval_batch_size=eval_batch_size,
        shuffle_buffer_size=shuffle_buffer_size,
        num_parallel_parser_calls=num_parallel_parser_calls,
        data_dir=data_dir)

  def _read_examples(self, split: base.Split) -> tf.data.Dataset:
    """Creates a dataset to be processed by _create_process_example_fn."""
    if split == base.Split.TEST:
      return tfds.load(
          self.name,
          split=self.test_split_name,
          try_gcs=True,
          data_dir=self._data_dir)
    elif split == base.Split.TRAIN:
      if self.mode == 'mismatched':
        raise ValueError('No training data for mismatched domains.')
      return tfds.load(
          self.name,
          split='train',
          try_gcs=True,
          data_dir=self._data_dir)
    else:
      return tfds.load(
          self.name,
          split=self.validation_split_name,
          try_gcs=True,
          data_dir=self._data_dir)

  def _create_process_example_fn(self, split: base.Split) -> base.PreProcessFn:
    """Create a pre-process function to return labels and sentence tokens."""

    def _example_parser(example: Dict[str, tf.Tensor]) -> Dict[str, Any]:
      """Parse sentences and labels from a serialized tf.train.Example."""
      idx = example['idx']
      label = example['label']
      text_a = example['premise']
      text_b = example['hypothesis']

      parsed_example = {
          'text_a': text_a,
          'text_b': text_b,
          'labels': label,
          'idx': idx
      }

      return parsed_example

    return _example_parser

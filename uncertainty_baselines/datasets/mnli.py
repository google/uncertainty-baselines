# coding=utf-8
# Copyright 2021 The Uncertainty Baselines Authors.
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
      split: str,
      shuffle_buffer_size: Optional[int] = None,
      num_parallel_parser_calls: int = 64,
      mode: str = 'matched',
      try_gcs: bool = False,
      download_data: bool = False,
      is_training: Optional[bool] = None):
    """Create an Genomics OOD tf.data.Dataset builder.

    Args:
      split: a dataset split, either a custom tfds.Split or one of the
        tfds.Split enums [TRAIN, VALIDAITON, TEST] or their lowercase string
        names.
      shuffle_buffer_size: the number of example to use in the shuffle buffer
        for tf.data.Dataset.shuffle().
      num_parallel_parser_calls: the number of parallel threads to use while
        preprocessing in tf.data.Dataset.map().
      mode: Type of data to import. If mode = "matched", import the in-domain
        data (glue/mnli_matched). If mode = "mismatched", import the
        out-of-domain data (glue/mnli_mismatched).
      try_gcs: Whether or not to try to use the GCS stored versions of dataset
        files. Currently unsupported.
      download_data: Whether or not to download data before loading. Currently
        unsupported.
      is_training: Whether or not the given `split` is the training split. Only
        required when the passed split is not one of ['train', 'validation',
        'test', tfds.Split.TRAIN, tfds.Split.VALIDATION, tfds.Split.TEST].
    """
    if mode not in ('matched', 'mismatched'):
      raise ValueError('"mode" must be either "matched" or "mismatched".'
                       'Got {}'.format(mode))
    if mode == 'mismatched' and split == tfds.Split.TRAIN:
      raise ValueError('No training data for mismatched domains.')

    if is_training is None:
      is_training = split in ['train', tfds.Split.TRAIN]

    if split == tfds.Split.VALIDATION:
      split = 'validation_' + mode
    if split == tfds.Split.TEST:
      split = 'test_' + mode

    name = 'glue/mnli'
    dataset_builder = tfds.builder(name, try_gcs=try_gcs)
    super().__init__(
        name=name,
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
      idx = example['idx']
      label = example['label']
      text_a = example['premise']
      text_b = example['hypothesis']

      return {
          'text_a': text_a,
          'text_b': text_b,
          'labels': label,
          'idx': idx
      }

    return _example_parser

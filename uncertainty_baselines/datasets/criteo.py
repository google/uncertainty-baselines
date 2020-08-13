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
"""Data loader for the Criteo dataset."""

import os.path
from typing import Any, Dict, Union

import tensorflow.compat.v2 as tf
from uncertainty_baselines.datasets import base


NUM_INT_FEATURES = 13  # Number of Criteo integer features.
NUM_CAT_FEATURES = 26  # Number of Criteo categorical features.
NUM_TOTAL_FEATURES = NUM_INT_FEATURES + NUM_CAT_FEATURES

_INT_KEY_TMPL = 'int-feature-%d'
_CAT_KEY_TMPL = 'categorical-feature-%d'


def _build_dataset(glob_dir: str, is_training: bool) -> tf.data.Dataset:
  cycle_len = 10 if is_training else 1
  dataset = tf.data.Dataset.list_files(glob_dir, shuffle=is_training)
  dataset = dataset.interleave(tf.data.TFRecordDataset, cycle_length=cycle_len)
  return dataset


def feature_name(idx: int) -> str:
  assert 0 < idx <= NUM_TOTAL_FEATURES
  if idx <= NUM_INT_FEATURES:
    return _INT_KEY_TMPL % idx
  return _CAT_KEY_TMPL % idx


def _make_features_spec() -> Dict[str, tf.io.FixedLenFeature]:
  features = {'clicked': tf.io.FixedLenFeature([1], tf.float32)}
  for idx in range(1, NUM_INT_FEATURES + 1):
    features[feature_name(idx)] = tf.io.FixedLenFeature([1], tf.float32, -1)
  for idx in range(NUM_INT_FEATURES + 1, NUM_TOTAL_FEATURES + 1):
    features[feature_name(idx)] = tf.io.FixedLenFeature([1], tf.string, '')
  return features


def apply_randomization(features, label, randomize_prob):
  """Randomize each categorical feature with some probability."""

  for idx in range(NUM_INT_FEATURES + 1, NUM_TOTAL_FEATURES + 1):
    key = feature_name(idx)

    def rnd_tok():
      return tf.as_string(
          tf.random.uniform(tf.shape(features[key]), 0, 99999999, tf.int32))  # pylint: disable=cell-var-from-loop

    # Ignore lint since tf.cond should evaluate lambda immediately.
    features[key] = tf.cond(tf.random.uniform([]) < randomize_prob,
                            rnd_tok,
                            lambda: features[key])  # pylint: disable=cell-var-from-loop
  return features, label


class CriteoDataset(base.BaseDataset):
  """Criteo dataset builder class."""

  def __init__(
      self,
      batch_size: int,
      eval_batch_size: int,
      shuffle_buffer_size: int = None,
      num_parallel_parser_calls: int = 64,
      dataset_dir: str = None,
      **unused_kwargs: Dict[str, Any]):
    """Create a Criteo tf.data.Dataset builder.

    Args:
      batch_size: the training batch size.
      eval_batch_size: the validation and test batch size.
      shuffle_buffer_size: the number of example to use in the shuffle buffer
        for tf.data.Dataset.shuffle().
      num_parallel_parser_calls: the number of parallel threads to use while
        preprocessing in tf.data.Dataset.map().
      dataset_dir: path to a directory containing the Criteo datasets, with
        filenames train-*-of-*', 'validate.tfr', 'test.tfr'.
    """
    self._dataset_dir = dataset_dir
    super(CriteoDataset, self).__init__(
        name='criteo',
        num_train_examples=int(37e6),
        num_validation_examples=4420308,
        num_test_examples=4420309,
        batch_size=batch_size,
        eval_batch_size=eval_batch_size,
        shuffle_buffer_size=shuffle_buffer_size,
        num_parallel_parser_calls=num_parallel_parser_calls)

  def _read_examples(self, split: base.Split) -> tf.data.Dataset:
    if split == base.Split.TRAIN:
      return _build_dataset(
          glob_dir=os.path.join(self._dataset_dir, 'train-*-of-*'),
          is_training=True)
    if split == base.Split.VAL:
      return _build_dataset(
          glob_dir=os.path.join(self._dataset_dir, 'validation-*-of-*'),
          is_training=False)
    return _build_dataset(
        glob_dir=os.path.join(self._dataset_dir, 'test-*-of-*'),
        is_training=False)

  def _create_process_example_fn(
      self,
      split: Union[float, base.Split]) -> base.PreProcessFn:

    def _example_parser(example: tf.train.Example) -> Dict[str, tf.Tensor]:
      """Parse features and labels from a serialized tf.train.Example."""
      features_spec = _make_features_spec()
      features = tf.io.parse_example(example, features_spec)
      features = {k: tf.squeeze(v, axis=0) for k, v in features.items()}
      labels = tf.cast(features.pop('clicked'), tf.int32)

      if isinstance(split, float):
        if split < 0.0 or split > 1.0:
          raise ValueError('shift_level not in [0, 1]: {}'.format(split))
        features, labels = apply_randomization(features, labels, split)

      return {'features': features, 'labels': labels}

    return _example_parser

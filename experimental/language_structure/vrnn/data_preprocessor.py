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

"""Helper functions for training/evaluating models."""

import collections
from typing import Any, Dict, Optional, Sequence, Union

import more_itertools
import tensorflow as tf
from uncertainty_baselines.datasets import base
from uncertainty_baselines.datasets import dialog_state_tracking
from vrnn import utils  # local file import from experimental.language_structure

USR_UTT_RAW_NAME = dialog_state_tracking.USR_UTT_RAW_NAME
SYS_UTT_RAW_NAME = dialog_state_tracking.SYS_UTT_RAW_NAME
USR_UTT_NAME = dialog_state_tracking.USR_UTT_NAME
SYS_UTT_NAME = dialog_state_tracking.SYS_UTT_NAME
STATE_LABEL_NAME = dialog_state_tracking.STATE_LABEL_NAME
DIAL_TURN_ID_NAME = dialog_state_tracking.DIAL_TURN_ID_NAME
DOMAIN_LABEL_NAME = dialog_state_tracking.DOMAIN_LABEL_NAME
TRAIN_SAMPLE_MASK_NAME = dialog_state_tracking.TRAIN_SAMPLE_MASK_NAME

INPUT_ID_NAME = 'input_word_ids'
INPUT_MASK_NAME = 'input_mask'

_UtteranceFeatureType = Dict[str, tf.Tensor]


def create_utterance_features(
    inputs: tf.Tensor) -> Sequence[_UtteranceFeatureType]:
  """Retrives utterance features from the dataset."""
  features = []
  for key in [USR_UTT_NAME, SYS_UTT_NAME]:
    features.append({
        INPUT_ID_NAME: inputs[key],
        INPUT_MASK_NAME: tf.sign(inputs[key]),
    })
  return features


def create_bert_utterance_features_fn(bert_preprocess_model: tf.keras.Model):
  """Creates utterance feature function for BERT embedding."""

  def _merge_utterance_features(
      features: Sequence[_UtteranceFeatureType]) -> _UtteranceFeatureType:
    """Merges features by the names."""
    merged_features = collections.defaultdict(list)
    for feature in features:
      for key, value in feature.items():
        merged_features[key].append(value)

    for key in merged_features:
      merged_features[key] = tf.stack(merged_features[key], axis=1)
    return merged_features

  def _create_bert_utterance_features(
      inputs: tf.Tensor) -> Sequence[_UtteranceFeatureType]:
    """Converts utterances into features for BERT embedding."""
    features = []
    for key in [USR_UTT_RAW_NAME, SYS_UTT_RAW_NAME]:
      features_by_step = bert_preprocess_model(tf.unstack(inputs[key], axis=1))
      merged_features = _merge_utterance_features(features_by_step)
      features.append(merged_features)
    return features

  return _create_bert_utterance_features


class DataPreprocessor:
  """Class preprocessing dialog_state_tracking datasets."""

  def __init__(self,
               encoder_feature_fn: Any,
               decoder_feature_fn: Any,
               num_states: int,
               labeled_dialog_turn_ids: Optional[tf.Tensor] = None,
               in_domains: Optional[tf.Tensor] = None):
    self._encoder_feature_fn = encoder_feature_fn
    self._decoder_feature_fn = decoder_feature_fn

    self._num_states = num_states
    self._labeled_dialog_turn_ids = labeled_dialog_turn_ids
    self._in_domains = in_domains

  def create_feature_and_label(self, inputs: tf.Tensor):
    """Creates the features and labels for training and evaluating."""
    encoder_input_1, encoder_input_2 = self._encoder_feature_fn(inputs)
    decoder_input_1, decoder_input_2 = self._decoder_feature_fn(inputs)

    label_id = inputs[STATE_LABEL_NAME]
    if self._labeled_dialog_turn_ids is None or DIAL_TURN_ID_NAME not in inputs:
      label_mask = tf.sign(label_id)
    else:
      label_mask = tf.cast(
          utils.value_in_tensor(inputs[DIAL_TURN_ID_NAME],
                                self._labeled_dialog_turn_ids),
          dtype=tf.int32)

    initial_state = tf.zeros_like(
        tf.tile(label_id[:, :1], [1, self._num_states]), dtype=tf.float32)
    initial_sample = tf.ones_like(
        tf.tile(label_id[:, :1], [1, self._num_states]), dtype=tf.float32)

    domain = inputs[DOMAIN_LABEL_NAME]
    if self._in_domains is not None:
      ind_mask = tf.cast(
          utils.value_in_tensor(domain, self._in_domains), dtype=tf.int32)
    else:
      ind_mask = None

    if TRAIN_SAMPLE_MASK_NAME in inputs:
      train_sample_mask = inputs[TRAIN_SAMPLE_MASK_NAME]
    else:
      train_sample_mask = None

    return (encoder_input_1, encoder_input_2, decoder_input_1, decoder_input_2,
            label_id, label_mask, initial_state, initial_sample,
            (domain, ind_mask), train_sample_mask)


def get_full_dataset_outputs(dataset_builder: base.BaseDataset) -> tf.Tensor:
  """Returns the entire dataset."""
  dataset = dataset_builder.load(batch_size=dataset_builder.num_examples)
  return more_itertools.first(dataset)


def create_dataset(
    dataset_builder: base.BaseDataset, batch_size: int, process_fn: Any,
    distributed_strategy: tf.distribute.Strategy, distributed: bool
) -> Union[tf.data.Dataset, tf.distribute.DistributedDataset]:
  """Creates (optionally distributed) dataset from dataset_builder and process_fn."""
  dataset = dataset_builder.load(batch_size=batch_size).map(process_fn)
  if distributed:
    dataset = distributed_strategy.experimental_distribute_dataset(dataset)
  return dataset

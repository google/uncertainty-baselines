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

"""Utility functions used in drug cardiotoxicity task training."""
import dataclasses
import json
import logging
import os
from typing import Any, List, Optional

import robustness_metrics as rm
import tensorflow as tf
import tensorflow_datasets as tfds

from uncertainty_baselines.datasets.drug_cardiotoxicity import DrugCardiotoxicityDataset


@dataclasses.dataclass(frozen=True)
class MPNNParameters:
  """Model Parameters used in MPNN architecture.

  Attributes:
    num_classes: Int, number of output classes.
    num_layers: Int, number of Message Passing layers.
    message_layer_size: Int, dimension of message representation.
    readout_layer_size: Int, dimension of graph level readout representation.
    use_gp_layer: Bool, whether to use Gaussian Process layer as classifier.
    learning_rate: Float, learning rate.
    num_epochs: Int, number of epoch for the entire training process.
    augmentations: List of str, representing augmentation function names.
    steps_per_epoch: Int, number of training batches to take in one epoch.
  """
  num_classes: int = 2
  num_layers: int = 2
  message_layer_size: int = 32
  readout_layer_size: int = 32
  use_gp_layer: bool = False
  learning_rate: float = 0.001
  num_epochs: int = 100
  augmentations: Optional[List[str]] = None
  steps_per_epoch: Optional[int] = None


@dataclasses.dataclass(frozen=True)
class GATParameters:
  """Model Parameters used in GAT architecture.

  Attributes:
    num_classes: Int, number of output classes.
    attention_heads: Int, number of graph attention heads per layer.
    out_node_feature_dim: Int, dimension (integer) of node level features
      outcoming from the attention layer.
    readout_layer_size: Int, dimension of graph level readout representation.
    constant_attention: Bool, whether to use constant attention.
    dropout_rate: Float, dropout rate.
    learning_rate: Float, learning rate.
    num_epochs: Int, number of epoch for the entire training process.
    augmentations: List of str, representing augmentation function names.
    steps_per_epoch: Int, number of training batches to take in one epoch.
  """
  num_classes: int = 2
  attention_heads: int = 3
  out_node_feature_dim: int = 32
  readout_layer_size: int = 32
  constant_attention: bool = False
  dropout_rate: float = 0.1
  learning_rate: float = 0.001
  num_epochs: int = 100
  augmentations: Optional[List[str]] = None
  steps_per_epoch: Optional[int] = None


def get_tpu_strategy(master: str) -> tf.distribute.TPUStrategy:
  """Builds a TPU distribution strategy."""
  logging.info('TPU master: %s', master)
  resolver = tf.distribute.cluster_resolver.TPUClusterResolver(master)
  tf.config.experimental_connect_to_cluster(resolver)
  tf.tpu.experimental.initialize_tpu_system(resolver)
  return tf.distribute.TPUStrategy(resolver)


def write_params(params: Any, filename: str):
  """Writes a dataclass to disk."""
  tf.io.gfile.makedirs(os.path.dirname(filename))
  with tf.io.gfile.GFile(filename, 'w') as f:
    json.dump(params, f, indent=2)


def get_metric_result_value(metric):
  """Gets the value of the input metric current result."""
  result = metric.result()
  if isinstance(metric, tf.keras.metrics.Metric):
    return result.numpy()
  elif isinstance(metric, rm.metrics.Metric):
    return list(result.values())[0]
  else:
    raise ValueError(f'Metric type {type(metric)} not supported.')


def load_dataset(data_dir, split, batch_size):
  """Loads a single dataset with specific split."""
  known_splits = [
      tfds.Split.TRAIN, tfds.Split.VALIDATION, tfds.Split.TEST,
      tfds.Split('test2')
  ]
  if split in known_splits:
    is_training = split == tfds.Split.TRAIN
  else:
    raise ValueError(
        'Received ambiguous split {}, must set is_training for splits other '
        'than "train", "validation", "test".'.format(split))

  builder = DrugCardiotoxicityDataset(
      split=split, data_dir=data_dir, is_training=is_training)
  dataset = builder.load(
      batch_size=batch_size).map(lambda x: (x['features'], x['labels']))
  steps = builder.num_examples//batch_size
  if not is_training:
    steps += 1

  return dataset, steps


def load_eval_datasets(identifiers, splits, data_dir, batch_size):
  """Loads all the eval datasets with specific splits."""
  eval_datasets = {}
  steps_per_eval = {}

  for identifier, split in zip(identifiers, splits):
    dataset, steps = load_dataset(data_dir, split, batch_size)
    eval_datasets[identifier] = dataset
    steps_per_eval[identifier] = steps

  return eval_datasets, steps_per_eval

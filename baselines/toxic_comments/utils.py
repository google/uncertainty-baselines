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
"""Utility function for BERT models."""
import json
import re
from typing import Any, Dict, List, Mapping, Optional, Tuple
from absl import logging

import tensorflow as tf

from tensorflow.core.protobuf import trackable_object_graph_pb2  # pylint: disable=g-direct-tensorflow-import
from official.nlp import optimization
from official.nlp.bert import configs


# Number of positive examples for each datasets.
# For `civil_comments`, the positive examples are based on threshold = 0.7.
NUM_POS_EXAMPLES = {
    'wikipedia_toxicity_subtypes': {'train': 15294, 'test': 6090},
    'civil_comments': {'train': 45451, 'test': 2458},
    'civil_comments/CivilCommentsIdentities': {'train': 12246, 'test': 634},
    'ind': {'train': 15294, 'test': 6090},
    'ood': {'train': 45451, 'test': 2458},
    'ood_identity': {'train': 12246, 'test': 634},
}


def create_class_weight(train_dataset_builders=None,
                        test_dataset_builders=None):
  """Creates a dictionary of class weights for computing eval metrics."""

  def generate_weight(num_positive_examples, num_examples):
    pos_fraction = num_positive_examples / num_examples
    neg_fraction = 1 - pos_fraction
    return [0.5 / neg_fraction, 0.5 / pos_fraction]

  class_weight = {}
  if train_dataset_builders:
    for dataset_name, dataset_builder in train_dataset_builders.items():
      class_weight['train/{}'.format(dataset_name)] = generate_weight(
          NUM_POS_EXAMPLES[dataset_name]['train'],
          dataset_builder.info['num_train_examples'])
  if test_dataset_builders:
    for dataset_name, dataset_builder in test_dataset_builders.items():
      class_weight['test/{}'.format(dataset_name)] = generate_weight(
          NUM_POS_EXAMPLES[dataset_name]['test'],
          dataset_builder.info['num_test_examples'])

  return class_weight


def create_config(config_dir: str) -> configs.BertConfig:
  """Load a BERT config object from directory."""
  with tf.io.gfile.GFile(config_dir) as config_file:
    bert_config = json.load(config_file)
  return configs.BertConfig(**bert_config)


def create_feature_and_label(inputs: Dict[str, Any], feature_size: int):
  """Creates features and labels for a BERT model."""
  input_token_ids = inputs['features']
  labels = inputs['labels']
  num_tokens = inputs['num_tokens']

  input_mask = tf.sequence_mask(num_tokens, feature_size, dtype=tf.int32)
  type_id = tf.sequence_mask(num_tokens, feature_size, dtype=tf.int32)
  features = [input_token_ids, input_mask, type_id]

  return features, labels


def create_optimizer(
    initial_lr: float,
    steps_per_epoch: int,
    epochs: int,
    warmup_proportion: float,
    end_lr: float = 0.0,
    optimizer_type: str = 'adamw') -> tf.keras.optimizers.Optimizer:
  """Creates a BERT optimizer with learning rate schedule."""
  num_train_steps = steps_per_epoch * epochs
  num_warmup_steps = int(num_train_steps * warmup_proportion)
  return optimization.create_optimizer(
      initial_lr,
      num_train_steps,
      num_warmup_steps,
      end_lr=end_lr,
      optimizer_type=optimizer_type)


def load_bert_weight_from_ckpt(
    bert_model: tf.keras.Model,
    bert_ckpt_dir: str,
    repl_patterns: Optional[Dict[str, str]] = None
) -> Tuple[tf.keras.Model, Mapping[str, str], Mapping[str, Any]]:
  """Loads checkpoint weights and match to model weights if applicable.

  Args:
    bert_model: The BERT Encoder model whose weights to load from checkpoints.
    bert_ckpt_dir: Path to BERT pre-trained checkpoints.
    repl_patterns: A mapping of regex string patterns and their replacements. To
      be used to update checkpoint weight names so they match those in
      bert_model (e.g., via re.sub(pattern, repl, weight_name))

  Returns:
    bert_model: The BERT Encoder model with loaded weights.
    names_to_keys: A dict mapping of weight name to checkpoint keys.
    keys_to_weights:  A dict mapping of checkpoint keys to weight values.
  """
  # Load a dict mapping of weight names to their corresponding checkpoint keys.
  names_to_keys = object_graph_key_mapping(bert_ckpt_dir)
  if repl_patterns:
    # Update weight names so they match those in bert_model
    names_to_keys = {
        update_weight_name(repl_patterns, weight_name): weight_key
        for weight_name, weight_key in names_to_keys.items()
    }

  # Load a dict mapping of checkpoint keys to weight values.
  logging.info('Loading weights from checkpoint: %s', bert_ckpt_dir)
  keys_to_weights = load_ckpt_keys_to_weight_mapping(bert_ckpt_dir)

  # Arranges the pre-trained weights in the order of model weights.
  init_weight_list = match_ckpt_weights_to_model(bert_model, names_to_keys,
                                                 keys_to_weights)

  # Load weights into model.
  bert_model.set_weights(init_weight_list)

  return bert_model, names_to_keys, keys_to_weights


def load_ckpt_keys_to_weight_mapping(ckpt_path: str) -> Mapping[str, Any]:
  """Loads weight values and their checkpoint keys from BERT checkpoint."""
  init_vars = tf.train.list_variables(ckpt_path)

  keys_to_weights = {}
  for name, _ in init_vars:
    var = tf.train.load_variable(ckpt_path, name)
    keys_to_weights[name] = var

  return keys_to_weights


def match_ckpt_weights_to_model(
    model: tf.keras.Model,
    names_to_keys: Mapping[str, str],
    keys_to_weights: Mapping[str, Any]) -> List[Any]:
  """Produces a list of checkpoint weights in the order specified by model."""
  init_weight_list = []

  for weight in model.weights:
    # Look up weight name in checkpoint weight names.
    weight_name = weight.name.replace(':0', '')
    ckpt_key = names_to_keys.get(weight_name, None)

    if ckpt_key:
      init_weight = keys_to_weights[ckpt_key]
    else:
      logging.info(
          '"%s" not found in checkpoint. '
          'Using randomly initialized values.', weight_name)
      init_weight = weight.numpy()

    init_weight_list.append(init_weight)

  return init_weight_list


def update_weight_name(repl_patterns: Dict[str, str], weight_name: str) -> str:
  """Updates weight names according a dictionary of replacement patterns."""
  # Create a regular expression from all of the dictionary keys
  regex = re.compile('|'.join(map(re.escape, repl_patterns.keys())))

  # For each match, look up the corresponding value in the repl_patterns dict.
  return regex.sub(lambda match: repl_patterns[match.group(0)], weight_name)


def object_graph_key_mapping(checkpoint_path: str) -> Dict[str, str]:
  """Return name to key mappings from the checkpoint."""
  reader = tf.train.load_checkpoint(checkpoint_path)
  object_graph_string = reader.get_tensor('_CHECKPOINTABLE_OBJECT_GRAPH')
  object_graph_proto = trackable_object_graph_pb2.TrackableObjectGraph()
  object_graph_proto.ParseFromString(object_graph_string)
  names_to_keys = {}
  for node in object_graph_proto.nodes:
    for attribute in node.attributes:
      names_to_keys[attribute.full_name] = attribute.checkpoint_key
  return names_to_keys

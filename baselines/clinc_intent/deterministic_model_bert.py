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
"""The Bidirectional Encoder Representations from Transformers (BERT) model."""

import json
import tensorflow as tf

from official.nlp import optimization
from official.nlp.bert import bert_models
from official.nlp.bert import configs



def create_config(config_dir):
  """Load a BERT config object from directory."""
  with tf.io.gfile.GFile(config_dir) as config_file:
    bert_config = json.load(config_file)
  return configs.BertConfig(**bert_config)


def create_feature_and_label(inputs, feature_size):
  """Creates features and labels for a BERT model."""
  input_token_ids = inputs['features']
  labels = inputs['labels']
  num_tokens = inputs['num_tokens']

  input_mask = tf.sequence_mask(num_tokens, feature_size, dtype=tf.int32)
  type_id = tf.sequence_mask(num_tokens, feature_size, dtype=tf.int32)
  features = [input_token_ids, input_mask, type_id]

  return features, labels


def create_optimizer(initial_lr,
                     steps_per_epoch,
                     epochs,
                     warmup_proportion,
                     end_lr=0.0,
                     optimizer_type='adamw'):
  """Creates a BERT optimizer with learning rate schedule."""
  num_train_steps = steps_per_epoch * epochs
  num_warmup_steps = int(num_train_steps * warmup_proportion)
  return optimization.create_optimizer(
      initial_lr,
      num_train_steps,
      num_warmup_steps,
      end_lr=end_lr,
      optimizer_type=optimizer_type)


def create_model(num_classes, feature_size, bert_config):
  """Creates a BERT classifier model."""
  # TODO(jereliu): Point to a locally implemented BERT for v2.
  return bert_models.classifier_model(
      bert_config=bert_config,
      num_labels=num_classes,
      max_seq_length=feature_size,
  )

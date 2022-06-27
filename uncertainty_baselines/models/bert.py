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

"""Bidirectional Encoder Representations from Transformers (BERT) model.

## References
[1]: Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova.
     BERT: Pre-training of Deep Bidirectional Transformers for Language
     Understanding.
     In _Proceedings of NAACL-HLT_, 2019.
     https://www.aclweb.org/anthology/N19-1423
"""
import json
from typing import Any, Dict, Tuple, List

import tensorflow as tf
from official.nlp import optimization
from official.nlp.bert import bert_models
from official.nlp.bert import configs



def create_config(config_dir: str) -> Any:
  """Load a BERT config object from directory."""
  with tf.io.gfile.GFile(config_dir) as config_file:
    return json.load(config_file)


def create_feature_and_label(inputs: Dict[str, Any],
                             max_seq_length: int) -> Tuple[List[Any], Any]:
  """Creates features and labels for a BERT model."""
  input_token_ids = inputs['features']
  labels = inputs['labels']
  num_tokens = inputs['num_tokens']

  input_mask = tf.sequence_mask(num_tokens, max_seq_length, dtype=tf.int32)
  type_id = tf.zeros_like(input_mask)
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


def bert_model(num_classes: int,
               max_seq_length: int,
               bert_config: configs.BertConfig,
               num_heads: int = 1) -> Tuple[tf.keras.Model, tf.keras.Model]:
  """BERT classifier model in functional API style.

  Construct a Keras model for predicting `num_labels` outputs from an input with
  maximum sequence length `max_seq_length`.

  Args:
    num_classes: (int) the number of classes.
    max_seq_length: (int) the maximum input sequence length.
    bert_config: (BertConfig) Configuration for a BERT model.
    num_heads: (int) the number of additional output heads.

  Returns:
    Combined prediction model (words, mask, type) -> (one-hot labels)
    BERT sub-model (words, mask, type) -> (bert_outputs)
  """
  # Defines initializer and encoder.
  final_layer_initializer = tf.keras.initializers.TruncatedNormal(
      stddev=bert_config.initializer_range)
  bert_encoder = bert_models.get_transformer_encoder(
      bert_config, max_seq_length, output_range=1)

  # Build model.
  inputs = bert_encoder.inputs
  _, cls_output = bert_encoder(inputs)
  cls_output = tf.keras.layers.Dropout(rate=bert_config.hidden_dropout_prob)(
      cls_output)

  # Build output.
  outputs = tf.keras.layers.Dense(
      num_classes,
      activation=None,
      kernel_initializer=final_layer_initializer,
      name='predictions/transform/logits')(
          cls_output)

  # Build additional heads if num_heads > 1.
  if num_heads > 1:
    outputs = [outputs]
    for head_id in range(1, num_heads):
      additional_outputs = tf.keras.layers.Dense(
          num_classes,
          activation=None,
          kernel_initializer=final_layer_initializer,
          name=f'predictions/transform/logits_{head_id}')(
              cls_output)

      outputs.append(additional_outputs)

  # Construct model.
  bert_classifier = tf.keras.Model(inputs=inputs, outputs=outputs)

  return bert_classifier, bert_encoder

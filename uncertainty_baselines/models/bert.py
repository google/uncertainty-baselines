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
from tensorflow_models.official.nlp import optimization
from tensorflow_models.official.nlp.bert import bert_models
from tensorflow_models.official.nlp.bert import configs



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


def create_model(
    num_classes: int, max_seq_length: int, initializer_range: float,
    hidden_dropout_prob: float,
    **bert_kwargs: Dict[str, Any]) -> Tuple[tf.keras.Model, tf.keras.Model]:
  """BERT classifier model in functional API style.

  Construct a Keras model for predicting `num_labels` outputs from an input with
  maximum sequence length `max_seq_length`.

  Args:
    num_classes: (int) the number of classes.
    max_seq_length: (int) the maximum input sequence length.
    initializer_range: (float) The stdev of the truncated_normal_initializer for
      initializing all weight matrices.
    hidden_dropout_prob: (float) The dropout probability for all fully connected
      layers in the embeddings, encoder, and pooler.
    **bert_kwargs: Additional arguements to the BertConfig.

  Returns:
    Combined prediction model (words, mask, type) -> (one-hot labels)
    BERT sub-model (words, mask, type) -> (bert_outputs)
  """
  final_layer_initializer = tf.keras.initializers.TruncatedNormal(
      stddev=initializer_range)
  bert_config = configs.BertConfig(
      initializer_range=initializer_range,
      hidden_dropout_prob=hidden_dropout_prob,
      **bert_kwargs)

  bert_encoder = bert_models.get_transformer_encoder(
      bert_config, max_seq_length, output_range=1)

  # initializer
  final_layer_initializer = tf.keras.initializers.TruncatedNormal(
      stddev=initializer_range)

  # build model
  inputs = bert_encoder.inputs
  _, cls_output = bert_encoder(inputs)
  cls_output = tf.keras.layers.Dropout(rate=hidden_dropout_prob)(cls_output)

  # build output
  outputs = tf.keras.layers.Dense(
      num_classes,
      activation=None,
      kernel_initializer=final_layer_initializer,
      name='predictions/transform/logits')(
          cls_output)

  # construct model
  bert_classifier = tf.keras.Model(inputs=inputs, outputs=outputs)

  return bert_classifier, bert_encoder

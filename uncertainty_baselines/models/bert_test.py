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

"""Tests for uncertainty_baselines.models.bert."""

import tensorflow as tf
import uncertainty_baselines as ub

from official.nlp.bert import configs


class BERTTest(tf.test.TestCase):

  def testCreateModel(self):
    """Testing if a BERT model can be configured correctly from BertConfig.

    A BERT-base encoder includes 10 embedding layer ops, 12 hidden layers and
    2 output layer ops. Therefore the created model is expected to contain 12
    layers.
    """
    num_classes = 100
    max_seq_length = 512
    # A standard BERT-base config
    bert_config_dict = configs.BertConfig(
        vocab_size=30521,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act='gelu',
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=16,
        initializer_range=0.02)

    _, bert_encoder = ub.models.BertBuilder(num_classes, max_seq_length,
                                            bert_config_dict)
    self.assertLen(bert_encoder.layers, 24)


if __name__ == '__main__':
  tf.test.main()

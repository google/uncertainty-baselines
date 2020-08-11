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
from official.nlp.bert import bert_models


def create_model(num_classes, feature_size, bert_config):
  """Creates a BERT classifier model."""
  # TODO(jereliu): Point to a locally implemented BERT for v2.
  return bert_models.classifier_model(
      bert_config=bert_config,
      num_labels=num_classes,
      max_seq_length=feature_size)

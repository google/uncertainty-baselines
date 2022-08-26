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

"""Library of models to use in Introspective Active Sampling.

This file contains a library of models that have two output heads: one for the
main training task and an optional second for bias. Any of these models can
serve as the base model trained in Introspective Active Sampling.
"""

import dataclasses
from typing import List, Optional

import tensorflow as tf


MODEL_REGISTRY = {}


def register_model(name: str):
  """Provides decorator to register model classes."""
  def save(model_class):
    MODEL_REGISTRY[name] = model_class
    return model_class

  return save


def get_model(name: str):
  """Retrieves dataset based on name."""
  if name not in MODEL_REGISTRY:
    raise ValueError(
        f'Unknown model: {name}\nPossible choices: {MODEL_REGISTRY.keys()}')
  return MODEL_REGISTRY[name]


@dataclasses.dataclass
class ModelTrainingParameters:
  model_name: str
  train_bias: bool
  num_classes: int
  num_epochs: float
  learning_rate: float = 1e-4
  hidden_sizes: Optional[List[int]] = None


@register_model('mlp')
class MLP(tf.keras.Model):
  """Defines a MLP model class with two output heads.

  One output head is for the main training task, while the other is an optional
  head to train on bias labels. Inputs are feature vectors.
  """

  def __init__(self,
               model_params: ModelTrainingParameters):
    super(MLP, self).__init__(name=model_params.model_name)

    self.dense_layers = tf.keras.models.Sequential(name='hidden')
    for size in model_params.hidden_sizes:
      self.dense_layers.add(tf.keras.layers.Dense(size,
                                                  activation='relu',
                                                  kernel_regularizer='l2'))
      self.dense_layers.add(tf.keras.layers.Dropout(0.2))

    self.output_main = tf.keras.layers.Dense(
        model_params.num_classes, activation='softmax', name='main')
    self.output_bias = tf.keras.layers.Dense(
        model_params.num_classes,
        trainable=model_params.train_bias,
        activation='softmax',
        name='bias')

  def call(self, inputs):
    x = self.dense_layers(inputs)
    out_main = self.output_main(x)
    out_bias = self.output_bias(x)
    return {
        'main': out_main,
        'bias': out_bias
    }

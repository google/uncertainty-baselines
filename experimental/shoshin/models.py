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
RESNET_IMAGE_SIZE = 224


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
  """Dataclass for training parameters."""
  model_name: str
  train_bias: bool
  num_classes: int
  num_subgroups: int
  num_epochs: int
  l2_regularization_factor: float = 0.5
  optimizer: str = 'sgd'
  learning_rate: float = 1e-5
  worst_group_label: Optional[int] = 2
  hidden_sizes: Optional[List[int]] = None
  do_reweighting: Optional[bool] = False
  reweighting_signal: Optional[str] = 'bias'
  reweighting_lambda: Optional[float] = 0.5
  reweighting_error_percentile_threshold: Optional[float] = 0.2


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


@register_model('resnet')
class ResNet(tf.keras.Model):
  """Defines a MLP model class with two output heads.

  One output head is for the main training task, while the other is an optional
  head to train on bias labels. Inputs are feature vectors.
  """

  def __init__(self,
               model_params: ModelTrainingParameters):
    super(ResNet, self).__init__(name=model_params.model_name)

    self.resnet_model = tf.keras.applications.resnet50.ResNet50(
        include_top=False,
        weights='imagenet',
        input_shape=(RESNET_IMAGE_SIZE, RESNET_IMAGE_SIZE, 3),
        classes=model_params.num_classes,
        pooling='avg'
        # TODO(jihyeonlee): Consider making pooling method a flag.
    )

    regularizer = tf.keras.regularizers.L2(
        l2=model_params.l2_regularization_factor)
    for layer in self.resnet_model.layers:
      layer.trainable = True
      if hasattr(layer, 'kernel_regularizer'):
        setattr(layer, 'kernel_regularizer', regularizer)
      if isinstance(layer, tf.keras.layers.Conv2D):
        layer.use_bias = False
        layer.kernel_initializer = 'he_normal'
      if isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.momentum = 0.9

    self.output_main = tf.keras.layers.Dense(
        2,
        activation='softmax',
        name='main',
        kernel_regularizer=regularizer)

    self.output_bias = tf.keras.layers.Dense(
        model_params.num_classes,
        trainable=model_params.train_bias,
        activation='softmax',
        name='bias',
        kernel_regularizer=regularizer)

  def call(self, inputs):
    x = self.resnet_model(inputs)
    out_main = self.output_main(x)
    out_bias = self.output_bias(x)
    return {
        'main': out_main,
        'bias': out_bias
    }

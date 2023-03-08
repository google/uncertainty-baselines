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
from typing import Dict, List, Optional

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
  subgroup_sizes: Dict[int, int]
  num_epochs: int
  num_channels: int = 3
  l2_regularization_factor: float = 0.5
  optimizer: str = 'sgd'
  learning_rate: float = 1e-5
  batch_size: int = 64
  load_pretrained_weights: Optional[bool] = False
  worst_group_label: Optional[int] = 2
  hidden_sizes: Optional[List[int]] = None
  use_pytorch_style_resnet: Optional[bool] = False
  do_reweighting: Optional[bool] = False
  reweighting_signal: Optional[str] = 'bias'
  reweighting_lambda: Optional[float] = 0.5
  reweighting_error_percentile_threshold: Optional[float] = 0.2


@register_model('two_tower_resnet50v2')
class TwoTowerResnet50v2(tf.keras.Model):
  """Two tower model based on Resnet50v2."""

  def __init__(
      self, model_params: ModelTrainingParameters
  ):
    super(TwoTowerResnet50v2, self).__init__(name=model_params.model_name)
    self.backbone = tf.keras.applications.resnet50.ResNet50(
        include_top=False,
        # classes=2,
        weights='imagenet',  # Also set to None.
        input_shape=(RESNET_IMAGE_SIZE, RESNET_IMAGE_SIZE, 3),
        # input_tensor=None,
        pooling='avg',
    )

    inputs = tf.keras.Input((RESNET_IMAGE_SIZE, RESNET_IMAGE_SIZE, 6))
    before = inputs[:, :, :, :3]
    after = inputs[:, :, :, 3:6]
    after_crop = tf.image.resize(
        tf.image.central_crop(after, 64 / RESNET_IMAGE_SIZE),
        [RESNET_IMAGE_SIZE, RESNET_IMAGE_SIZE],
        method='bilinear',
    )
    before_embedding = self.backbone(before)
    after_embedding = self.backbone(after)
    after_crop_embedding = self.backbone(after_crop)
    combined = tf.concat(
        [before_embedding, after_embedding, after_crop_embedding], axis=1
    )
    outputs = tf.keras.layers.Dropout(0.5)(combined)
    outputs = tf.keras.layers.Dense(units=256, activation='relu')(outputs)
    outputs = tf.keras.layers.Dropout(0.5)(outputs)
    outputs = tf.keras.layers.Dense(units=64, activation='relu')(outputs)
    outputs = tf.keras.layers.Dropout(0.5)(outputs)
    outputs = tf.keras.layers.Dense(
        units=model_params.num_classes, activation='sigmoid'
    )(outputs)
    self.backbone.trainable = False
    self.backbone.layers[-1].trainable = True
    self.model = tf.keras.Model(inputs, outputs)
    self.output_bias = tf.keras.layers.Dense(
        model_params.num_classes,
        trainable=model_params.train_bias,
        activation='softmax',
        name='bias',
    )

  def call(self, inputs):
    x_1 = self.backbone(inputs[:,:, :, :3])
    x_2 = self.backbone(inputs[:,:, :, 3:])
    x = tf.concat([x_1, x_2], axis=-1)
    out_bias = self.output_bias(x)
    return {'main': self.model(inputs), 'bias': out_bias}


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
        weights='imagenet' if model_params.load_pretrained_weights else None,
        input_shape=(RESNET_IMAGE_SIZE, RESNET_IMAGE_SIZE,
                     model_params.num_channels),
        classes=model_params.num_classes,
        pooling='avg'
        # TODO(jihyeonlee): Consider making pooling method a flag.
    )

    regularizer = tf.keras.regularizers.L2(
        l2=model_params.l2_regularization_factor)
    for layer in self.resnet_model.layers:
      layer.trainable = True
      if model_params.use_pytorch_style_resnet:
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

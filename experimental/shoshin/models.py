# coding=utf-8
# Copyright 2023 The Uncertainty Baselines Authors.
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
  subgroup_sizes: Dict[str, int]
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

  def asdict(self):
    return dataclasses.asdict(self)

  @classmethod
  def from_dict(cls, kwargs):
    return ModelTrainingParameters(**kwargs)


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
    x = self.dense_layers(inputs['large_image'])
    out_main = self.output_main(x)
    out_bias = self.output_bias(x)
    return {
        'main': out_main,
        'bias': out_bias
    }


@register_model('resnet50v1')
@tf.keras.saving.register_keras_serializable('resnet50v1')
class ResNet50v1(tf.keras.Model):
  """Defines a ResNet50v1 model class with two output heads.

  One output head is for the main training task, while the other is an optional
  head to train on bias labels. Inputs are feature vectors.
  """

  def __init__(self,
               model_params: ModelTrainingParameters):
    super(ResNet50v1, self).__init__(name=model_params.model_name)

    self.model_params = model_params
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
          initializer = tf.keras.initializers.HeNormal()
          layer.kernel_initializer = initializer
        if isinstance(layer, tf.keras.layers.BatchNormalization):
          layer.momentum = 0.9

    self.output_main = tf.keras.layers.Dense(
        model_params.num_classes,
        activation='softmax',
        name='main',
        kernel_regularizer=regularizer)

    self.output_bias = tf.keras.layers.Dense(
        model_params.num_classes,
        trainable=model_params.train_bias,
        activation='softmax',
        name='bias',
        kernel_regularizer=regularizer)

  def get_config(self):
    config = super(ResNet50v1, self).get_config()
    config.update({'model_params': self.model_params.asdict(),
                   'resnet_model': self.resnet_model,
                   'output_main': self.output_main,
                   'output_bias': self.output_bias})
    return config

  @classmethod
  def from_config(cls, config):
    return cls(ModelTrainingParameters.from_dict(config['model_params']))

  def call(self, inputs):
    x = self.resnet_model(inputs['large_image'])
    out_main = self.output_main(x)
    out_bias = self.output_bias(x)
    return {
        'main': out_main,
        'bias': out_bias
    }


@register_model('resnet50v2')
@tf.keras.saving.register_keras_serializable('resnet50v2')
class ResNet50v2(tf.keras.Model):
  """Defines a ResNet50v2 model class with two output heads.

  One output head is for the main training task, while the other is an optional
  head to train on bias labels. Inputs are feature vectors.
  """

  def __init__(self,
               model_params: ModelTrainingParameters):
    super(ResNet50v2, self).__init__(name=model_params.model_name)

    self.model_params = model_params
    self.resnet_model = tf.keras.applications.resnet_v2.ResNet50V2(
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
          initializer = tf.keras.initializers.HeNormal()
          layer.kernel_initializer = initializer
        if isinstance(layer, tf.keras.layers.BatchNormalization):
          layer.momentum = 0.9

    self.output_main = tf.keras.layers.Dense(
        model_params.num_classes,
        activation='softmax',
        name='main',
        kernel_regularizer=regularizer)

    self.output_bias = tf.keras.layers.Dense(
        model_params.num_classes,
        trainable=model_params.train_bias,
        activation='softmax',
        name='bias',
        kernel_regularizer=regularizer)

  def get_config(self):
    config = super(ResNet50v2, self).get_config()
    config.update({'model_params': self.model_params.asdict(),
                   'resnet_model': self.resnet_model,
                   'output_main': self.output_main,
                   'output_bias': self.output_bias})
    return config

  @classmethod
  def from_config(cls, config):
    return cls(ModelTrainingParameters.from_dict(config['model_params']))

  def call(self, inputs):
    x = self.resnet_model(inputs['large_image'])
    out_main = self.output_main(x)
    out_bias = self.output_bias(x)
    return {
        'main': out_main,
        'bias': out_bias
    }


@register_model('two_tower')
@tf.keras.saving.register_keras_serializable('two_tower')
class TwoTower(tf.keras.Model):
  """Defines Two Tower class with two output heads.

  One output head is for the main training task, while the other is an optional
  head to train on bias labels. Inputs are feature vectors. 
  """

  def __init__(self,
               model_params: ModelTrainingParameters):
    super(TwoTower, self).__init__(name=model_params.model_name)

    self.model_params = model_params
    backbone = tf.keras.applications.resnet_v2.ResNet50V2(
        include_top=False,
        weights='imagenet' if model_params.load_pretrained_weights else None,
        input_shape=(RESNET_IMAGE_SIZE, RESNET_IMAGE_SIZE,
                     3),
        classes=model_params.num_classes,
        pooling='avg'
        # TODO(jihyeonlee): Consider making pooling method a flag.
    )

    if model_params.load_pretrained_weights:
      backbone.trainable = False

    dense = tf.keras.Sequential([
        # TODO(melfatih): Add a hyperparameter for dropout.
        tf.keras.layers.Dropout(0.5),
        # TODO(melfatih): Add a hyperparameter for embedding size.
        tf.keras.layers.Dense(units=256, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(units=64, activation='relu'),
        tf.keras.layers.Dropout(0.5),
    ])
    self.backbone = tf.keras.Sequential([backbone, dense])
    self.output_main = (
        tf.keras.layers.Dense(
            units=model_params.num_classes, activation='sigmoid'
        )
    )
    self.output_bias = tf.keras.layers.Dense(
        model_params.num_classes,
        trainable=model_params.train_bias,
        activation='softmax',
        name='bias',
    )

  def get_config(self):
    config = super(TwoTower, self).get_config()
    config.update({'model_params': self.model_params.asdict(),
                   'backbone': self.backbone,
                   'output_main': self.output_main,
                   'output_bias': self.output_bias})
    return config

  @classmethod
  def from_config(cls, config):
    return cls(ModelTrainingParameters.from_dict(config['model_params']))

  def call(self, inputs):
    large_image, small_image = inputs['large_image'], inputs['small_image']

    if self.model_params.num_channels == 3:
      after_embed = self.backbone(large_image)
      after_crop_embed = self.backbone(small_image)
      combined = tf.concat([after_embed, after_crop_embed], axis=-1)
    elif self.model_params.num_channels == 6:
      after_embed = self.backbone(large_image[:, :, :, 3:])
      after_crop_embed = self.backbone(small_image[:, :, :, 3:])
      before_embed = self.backbone(large_image[:, :, :, :3])
      combined = tf.concat(
          [before_embed, after_embed, after_crop_embed], axis=-1
      )

    out_main = self.output_main(combined)
    out_bias = self.output_bias(combined)
    return {'main': out_main, 'bias': out_bias}

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

"""Heteroscedastic ResNet50 model.

An heteroscedastic output layer [1] uses a multivariate Normal distributed
latent variable on the final hidden layer. The covariance matrix of this latent
variable, models the aleatoric uncertainty due to label noise.

References:
  [1]: Mark Collier, Basil Mustafa, Efi Kokiopoulou, Rodolphe Jenatton and
       Jesse Berent. Correlated Input-Dependent Label Noise in Large-Scale Image
       Classification. In Proc. of the IEEE/CVF Conference on Computer Vision
       and Pattern Recognition (CVPR), 2021, pp. 1551-1560.
       https://arxiv.org/abs/2105.10305
"""

import string
from typing import Optional

import edward2 as ed
import tensorflow as tf

# Use batch normalization defaults from Pytorch.
BATCH_NORM_DECAY = 0.9
BATCH_NORM_EPSILON = 1e-5


def bottleneck_block(inputs,
                     filters,
                     stage,
                     block,
                     strides):
  """Residual block with 1x1 -> 3x3 -> 1x1 convs in main path.

  Note that strides appear in the second conv (3x3) rather than the first (1x1).
  This is also known as "ResNet v1.5" as it differs from He et al. (2015)
  (http://torch.ch/blog/2016/02/04/resnets.html).

  Args:
    inputs: tf.Tensor.
    filters: list of integers, the filters of 3 conv layer at main path
    stage: integer, current stage label, used for generating layer names
    block: 'a','b'..., current block label, used for generating layer names
    strides: Strides for the second conv layer in the block.

  Returns:
    tf.Tensor.
  """
  filters1, filters2, filters3 = filters
  conv_name_base = 'res' + str(stage) + block + '_branch'
  bn_name_base = 'bn' + str(stage) + block + '_branch'

  x = tf.keras.layers.Conv2D(
      filters1,
      kernel_size=1,
      use_bias=False,
      kernel_initializer='he_normal',
      name=conv_name_base + '2a')(inputs)
  x = tf.keras.layers.BatchNormalization(
      momentum=BATCH_NORM_DECAY,
      epsilon=BATCH_NORM_EPSILON,
      name=bn_name_base + '2a')(x)
  x = tf.keras.layers.Activation('relu')(x)

  x = tf.keras.layers.Conv2D(
      filters2,
      kernel_size=3,
      strides=strides,
      padding='same',
      use_bias=False,
      kernel_initializer='he_normal',
      name=conv_name_base + '2b')(x)
  x = tf.keras.layers.BatchNormalization(
      momentum=BATCH_NORM_DECAY,
      epsilon=BATCH_NORM_EPSILON,
      name=bn_name_base + '2b')(x)
  x = tf.keras.layers.Activation('relu')(x)

  x = tf.keras.layers.Conv2D(
      filters3,
      kernel_size=1,
      use_bias=False,
      kernel_initializer='he_normal',
      name=conv_name_base + '2c')(x)
  x = tf.keras.layers.BatchNormalization(
      momentum=BATCH_NORM_DECAY,
      epsilon=BATCH_NORM_EPSILON,
      name=bn_name_base + '2c')(x)

  shortcut = inputs
  if not x.shape.is_compatible_with(shortcut.shape):
    shortcut = tf.keras.layers.Conv2D(
        filters3,
        kernel_size=1,
        use_bias=False,
        strides=strides,
        kernel_initializer='he_normal',
        name=conv_name_base + '1')(shortcut)
    shortcut = tf.keras.layers.BatchNormalization(
        momentum=BATCH_NORM_DECAY,
        epsilon=BATCH_NORM_EPSILON,
        name=bn_name_base + '1')(shortcut)

  x = tf.keras.layers.add([x, shortcut])
  x = tf.keras.layers.Activation('relu')(x)
  return x


def group(inputs, filters, num_blocks, stage, strides):
  blocks = string.ascii_lowercase
  x = bottleneck_block(inputs, filters, stage, block=blocks[0], strides=strides)
  for i in range(num_blocks - 1):
    x = bottleneck_block(x, filters, stage, block=blocks[i + 1], strides=1)
  return x


def resnet50_heteroscedastic(input_shape,
                             num_classes,
                             temperature,
                             num_factors,
                             num_mc_samples=10000,
                             multiclass=True,
                             eps=1e-5,
                             return_unaveraged_logits=False,
                             tune_temperature: bool = False,
                             temperature_lower_bound: Optional[float] = None,
                             temperature_upper_bound: Optional[float] = None):
  """Builds ResNet50.

  Using strided conv, pooling, four groups of residual blocks, and pooling, the
  network maps spatial features of size 224x224 -> 112x112 -> 56x56 -> 28x28 ->
  14x14 -> 7x7 (Table 1 of He et al. (2015)).

  Args:
    input_shape: Shape tuple of input excluding batch dimension.
    num_classes: Number of output classes.
    temperature: Float or scalar `Tensor` representing the softmax
      temperature.
    num_factors: Integer. Number of factors to use in approximation to full
      rank covariance matrix. If num_factors <= 0, then the diagonal covariance
      method MCSoftmaxDense is used.
    num_mc_samples: The number of Monte-Carlo samples used to estimate the
        predictive distribution.
    multiclass: Boolean. If True then return a multiclass classifier, otherwise
      a multilabel classifier.
    eps: Float. Clip probabilities into [eps, 1.0] softmax or
        [eps, 1.0 - eps] sigmoid before applying log (softmax), or inverse
        sigmoid.
    return_unaveraged_logits: Boolean. Whether to also return the logits
        before taking the MC average over samples.
    tune_temperature: Boolean. If True, the temperature is optimized during
      the training as any other parameters.
    temperature_lower_bound: Float. The lowest value the temperature can take
      when it is optimized. By default, a pre-defined lower bound is used.
    temperature_upper_bound: Float. The highest value the temperature can take
      when it is optimized. By default, a pre-defined upper bound is used.

  Returns:
    tf.keras.Model.
  """
  inputs = tf.keras.layers.Input(shape=input_shape)
  x = tf.keras.layers.ZeroPadding2D(padding=3, name='conv1_pad')(inputs)
  x = tf.keras.layers.Conv2D(
      64,
      kernel_size=7,
      strides=2,
      padding='valid',
      use_bias=False,
      kernel_initializer='he_normal',
      name='conv1')(x)
  x = tf.keras.layers.BatchNormalization(
      momentum=BATCH_NORM_DECAY,
      epsilon=BATCH_NORM_EPSILON,
      name='bn_conv1')(x)
  x = tf.keras.layers.Activation('relu')(x)
  x = tf.keras.layers.MaxPooling2D(3, strides=2, padding='same')(x)
  x = group(x, [64, 64, 256], stage=2, num_blocks=3, strides=1)
  x = group(x, [128, 128, 512], stage=3, num_blocks=4, strides=2)
  x = group(x, [256, 256, 1024], stage=4, num_blocks=6, strides=2)
  x = group(x, [512, 512, 2048], stage=5, num_blocks=3, strides=2)

  x = tf.keras.layers.GlobalAveragePooling2D(name='avg_pool')(x)

  het_layer_args = {'temperature': temperature,
                    'train_mc_samples': num_mc_samples,
                    'test_mc_samples': num_mc_samples,
                    'share_samples_across_batch': True,
                    'logits_only': True, 'eps': eps,
                    'dtype': tf.float32, 'name': 'fc1000',
                    'return_unaveraged_logits': return_unaveraged_logits,
                    'tune_temperature': tune_temperature,
                    'temperature_lower_bound': temperature_lower_bound,
                    'temperature_upper_bound': temperature_upper_bound,
                    }
  if multiclass:
    het_layer_args.update({'num_classes': num_classes})
    if num_factors <= 0:
      output_layer = ed.layers.MCSoftmaxDense(**het_layer_args)
    else:
      het_layer_args.update({'num_factors': num_factors})
      output_layer = ed.layers.MCSoftmaxDenseFA(**het_layer_args)
  else:
    het_layer_args.update({'num_outputs': num_classes,
                           'num_factors': num_factors})
    output_layer = ed.layers.MCSigmoidDenseFA(**het_layer_args)

  x = output_layer(x)

  return tf.keras.Model(inputs=inputs, outputs=x, name='resnet50')

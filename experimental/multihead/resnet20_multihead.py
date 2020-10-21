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
"""TF Keras definition for Resnet-20 for CIFAR."""

from typing import Any, Dict

import tensorflow as tf
import numpy as np

def _resnet_layer(
    inputs: tf.Tensor,
    num_filters: int = 16,
    kernel_size: int = 3,
    strides: int = 1,
    use_activation: bool = True,
    use_norm: bool = True,
    l2_weight: float = 1e-4) -> tf.Tensor:
  """2D Convolution-Batch Normalization-Activation stack builder.

  Args:
    inputs: input tensor from input image or previous layer.
    num_filters: Conv2D number of filters.
    kernel_size: Conv2D square kernel dimensions.
    strides: Conv2D square stride dimensions.
    use_activation: whether or not to use a non-linearity.
    use_norm: whether to include normalization.
    l2_weight: the L2 regularization coefficient to use for the convolution
      kernel regularizer.

  Returns:
      Tensor output of this layer.
  """
  kernel_regularizer = None
  if l2_weight:
    kernel_regularizer = tf.keras.regularizers.l2(l2_weight)
  conv_layer = tf.keras.layers.Conv2D(
      num_filters,
      kernel_size=kernel_size,
      strides=strides,
      padding='same',
      kernel_initializer='he_normal',
      kernel_regularizer=kernel_regularizer)

  x = conv_layer(inputs)
  x = tf.keras.layers.BatchNormalization()(x) if use_norm else x
  x = tf.keras.layers.ReLU()(x) if use_activation is not None else x
  return x


def create_model(
    batch_size: int,
    l2_weight: float = 0.0,
    certainty_variant: str = 'total', # total, partial or normalized
    **unused_kwargs: Dict[str, Any]) -> tf.keras.models.Model:
  """Resnet-20 v1, takes (32, 32, 3) input and returns logits of shape (10,)."""
  # TODO(znado): support NCHW data format.
  input_layer = tf.keras.layers.Input(
      shape=(32, 32, 3), batch_size=batch_size)
  depth = 20
  num_filters = 16
  num_res_blocks = int((depth - 2) / 6)

  x = _resnet_layer(
      inputs=input_layer,
      num_filters=num_filters,
      l2_weight=l2_weight)
  for stack in range(3):
    for res_block in range(num_res_blocks):
      strides = 1
      if stack > 0 and res_block == 0:
        strides = 2
      y = _resnet_layer(
          inputs=x,
          num_filters=num_filters,
          strides=strides,
          l2_weight=l2_weight)
      y = _resnet_layer(
          inputs=y,
          num_filters=num_filters,
          use_activation=False,
          l2_weight=l2_weight)
      if stack > 0 and res_block == 0:
        x = _resnet_layer(
            inputs=x,
            num_filters=num_filters,
            kernel_size=1,
            strides=strides,
            use_activation=False,
            use_norm=False,
            l2_weight=l2_weight)
      x = tf.keras.layers.add([x, y])
      x = tf.keras.layers.ReLU()(x)
    num_filters *= 2

  x = tf.keras.layers.AveragePooling2D(pool_size=8)(x)
  x = tf.keras.layers.Flatten()(x)
  logits = tf.keras.layers.Dense(10, kernel_initializer='he_normal')(x)
  #logits = tf.identity(logits, name="logits")
    
  probs = tf.math.sigmoid(logits)
  probs_comp = 1-probs
  K = probs.shape[1]
  cert_list = []
  for i in range(K):
    proj_vec = np.zeros(K)
    proj_vec[i]=1
    proj_mat = np.outer(proj_vec,proj_vec)
    proj_mat_comp = np.identity(K)-np.outer(proj_vec,proj_vec)
    tproj_mat = tf.constant(proj_mat,dtype=tf.float32)
    tproj_mat_comp = tf.constant(proj_mat_comp,dtype=tf.float32)
    out = tf.tensordot(probs,tproj_mat,axes=1) + tf.tensordot(probs_comp,tproj_mat_comp,axes=1)
    cert_list+=[tf.reduce_prod(out,axis=1)]
    
  if certainty_variant == 'partial':
    certs = tf.stack(cert_list,axis=1)
    
  elif certainty_variant == 'total':
    certs = tf.stack(cert_list,axis=1)
    certs_argmax = tf.one_hot(tf.argmax(certs,axis=1),depth=K)
    certs_reduce = tf.tile(tf.reduce_sum(certs,axis=1,keepdims=True),[1,K])
    certs = tf.math.multiply(certs_argmax,certs_reduce)
    
  elif certainty_variant == 'normalized':
    certs = tf.stack(cert_list,axis=1)
    certs_norm = tf.tile(tf.reduce_sum(certs,axis=1,keepdims=True),[1,K])
    certs = tf.math.divide(certs,certs_norm)
    
  else:
    raise ValueError('unknown certainty_variant')

  #certs = tf.identity(certs, name="certs")

  return tf.keras.models.Model(
      inputs=input_layer, 
      outputs={'logits':logits,'probs':probs,'certs':certs}, 
      name='resnet20-multihead')

# coding=utf-8
# Copyright 2021 The Uncertainty Baselines Authors.
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

"""Wide ResNet with CondConv layers."""
import functools
import warnings
import tensorflow as tf

try:
  import edward2 as ed  # pylint: disable=g-import-not-at-top
except ImportError as e:
  warnings.warn(f'Skipped due to ImportError: {e}')

BatchNormalization = functools.partial(  # pylint: disable=invalid-name
    tf.keras.layers.BatchNormalization,
    epsilon=1e-5,
    momentum=0.9)
Conv2D = functools.partial(  # pylint: disable=invalid-name
    tf.keras.layers.Conv2D,
    kernel_size=3,
    padding='same',
    use_bias=False)
CondConv2D = functools.partial(  # pylint: disable=invalid-name
    ed.layers.CondConv2D,
    kernel_size=3,
    padding='same',
    use_bias=False)


def basic_block(inputs, filters, strides, num_experts, batch_size,
                cond_placement, normalize_routing, routing_pooling, top_k,
                routing_fn, l2):
  """Basic residual block of two 3x3 convs.

  Args:
    inputs: tf.Tensor.
    filters: Number of filters for Conv2D.
    strides: Stride dimensions for Conv2D.
    num_experts: Number of experts to aggregate over.
    batch_size: batch_size for conditional layers.
    cond_placement: Enum in ['dropout', 'all', 'none'].
    normalize_routing: Whether to normalize CondConv routing weights.
    routing_pooling: Type of pooling to apply to the inputs of routing.
    top_k: Number of experts to select for a sparse MoE setting.
    routing_fn: An Enum in ['sigmoid', 'softmax', 'noisy_softmax',
      'onehot_top_k', 'noisy_onehot_top_k','softmax_top_k',
      'noisy_softmax_top_k'].
    l2: L2 regularization coefficient.

  Returns:
    tf.Tensor.
  """
  x = inputs
  y = inputs
  y = BatchNormalization(
      beta_regularizer=tf.keras.regularizers.l2(l2),
      gamma_regularizer=tf.keras.regularizers.l2(l2))(
          y)
  y = tf.keras.layers.Activation('relu')(y)
  routing_weights_list = []
  if cond_placement in ['all', 'dropout']:
    routing_weights = ed.layers.RoutingLayer(num_experts, normalize_routing,
                                             routing_pooling, top_k,
                                             routing_fn)(
                                                 y)
    routing_weights_list.append(routing_weights)
    y = CondConv2D(
        filters,
        strides=strides,
        kernel_regularizer=tf.keras.regularizers.l2(l2),
        num_experts=num_experts,
        batch_size=batch_size)(y, routing_weights)
  else:
    y = Conv2D(
        filters,
        strides=strides,
        kernel_regularizer=tf.keras.regularizers.l2(l2))(
            y)

  y = BatchNormalization(
      beta_regularizer=tf.keras.regularizers.l2(l2),
      gamma_regularizer=tf.keras.regularizers.l2(l2))(
          y)
  y = tf.keras.layers.Activation('relu')(y)
  if cond_placement == 'all':
    routing_weights = ed.layers.RoutingLayer(num_experts, normalize_routing,
                                             routing_pooling, top_k,
                                             routing_fn)(
                                                 y)
    routing_weights_list.append(routing_weights)
    y = CondConv2D(
        filters,
        strides=1,
        kernel_regularizer=tf.keras.regularizers.l2(l2),
        num_experts=num_experts,
        batch_size=batch_size)(y, routing_weights)
  else:
    y = Conv2D(
        filters, strides=1, kernel_regularizer=tf.keras.regularizers.l2(l2))(
            y)

  if not x.shape.is_compatible_with(y.shape):
    if cond_placement == 'all':
      routing_weights = ed.layers.RoutingLayer(num_experts, normalize_routing,
                                               routing_pooling, top_k,
                                               routing_fn)(
                                                   x)
      routing_weights_list.append(routing_weights)
      x = CondConv2D(
          filters,
          kernel_size=1,
          strides=strides,
          kernel_regularizer=tf.keras.regularizers.l2(l2),
          num_experts=num_experts,
          batch_size=batch_size)(x, routing_weights)
    else:
      x = Conv2D(
          filters,
          kernel_size=1,
          strides=strides,
          kernel_regularizer=tf.keras.regularizers.l2(l2))(
              x)

  x = tf.keras.layers.add([x, y])
  return x, routing_weights_list


# Sharing across the whole network, groups, weights: might be tricky
def group(inputs, filters, strides, num_blocks, num_experts, batch_size,
          cond_placement, normalize_routing, routing_pooling, top_k, routing_fn,
          l2):
  """Group of residual blocks."""
  x, routing_weights_block_list = basic_block(inputs, filters, strides,
                                              num_experts, batch_size,
                                              cond_placement, normalize_routing,
                                              routing_pooling, top_k,
                                              routing_fn, l2)
  routing_weights_group_list = routing_weights_block_list
  for _ in range(num_blocks - 1):
    x, routing_weights_block_list = basic_block(
        x,
        filters=filters,
        strides=1,
        num_experts=num_experts,
        batch_size=batch_size,
        cond_placement=cond_placement,
        normalize_routing=normalize_routing,
        routing_pooling=routing_pooling,
        top_k=top_k,
        routing_fn=routing_fn,
        l2=l2)
    routing_weights_group_list.extend(routing_weights_block_list)

  return x, routing_weights_group_list


def wide_resnet_condconv(input_shape, depth, width_multiplier, num_classes,
                         num_experts, per_core_batch_size, use_cond_dense,
                         reduce_dense_outputs, cond_placement, routing_fn,
                         normalize_routing, normalize_dense_routing,
                         routing_pooling, top_k, l2):
  """Builds Wide ResNet.

  Following Zagoruyko and Komodakis (2016), it accepts a width multiplier on the
  number of filters. Using three groups of residual blocks, the network maps
  spatial features of size 32x32 -> 16x16 -> 8x8.

  Args:
    input_shape: tf.Tensor.
    depth: Total number of convolutional layers. "n" in WRN-n-k. It differs from
      He et al. (2015)'s notation which uses the maximum depth of the network
      counting non-conv layers like dense.
    width_multiplier: Integer to multiply the number of typical filters by. "k"
      in WRN-n-k.
    num_classes: Number of output classes.
    num_experts: Number of experts to aggregate over.
    per_core_batch_size: Size of dataset ber core/device.
    use_cond_dense: Whether to use CondDense.
    reduce_dense_outputs: Whether to reduce the outputs of the CondDense or to
      return a list of weights and logits for each expert.
    cond_placement: Enum in ['dropout', 'all', 'none'].
    routing_fn: An Enum in ['sigmoid', 'softmax', 'noisy_softmax',
      'onehot_top_k', 'noisy_onehot_top_k','softmax_top_k',
      'noisy_softmax_top_k'].
    normalize_routing: Whether to normalize the routing weights of CondConv.
    normalize_dense_routing: Whether to normalize the routing weights of the
      CondDense layer.
    routing_pooling: Enum in ['global_average', 'global_max', 'average_8',
      'max_8', 'flatten']
    top_k: Number of experts to select in a sparse MoE setting.
    l2: L2 regularization coefficient.

  Returns:
    tf.keras.Model.
  """
  if (depth - 4) % 6 != 0:
    raise ValueError('depth should be 6n+4 (e.g., 16, 22, 28, 40).')
  num_blocks = (depth - 4) // 6
  inputs = tf.keras.layers.Input(shape=input_shape)
  all_routing_weights = []
  if cond_placement == 'all':
    routing_weights = ed.layers.RoutingLayer(num_experts, normalize_routing,
                                             routing_pooling, top_k,
                                             routing_fn)(
                                                 inputs)
    all_routing_weights.extend([routing_weights])
    x = CondConv2D(
        16,
        strides=1,
        kernel_regularizer=tf.keras.regularizers.l2(l2),
        num_experts=num_experts,
        batch_size=per_core_batch_size)(inputs, routing_weights)
  else:
    # For the 'dropout' and 'none' configurations, the input layer is not MoE.
    x = Conv2D(filters=16, strides=1)(inputs)

  for strides, filters in zip([1, 2, 2], [16, 32, 64]):
    x, routing_weights_group_list = group(
        x,
        filters=filters * width_multiplier,
        strides=strides,
        num_blocks=num_blocks,
        num_experts=num_experts,
        batch_size=per_core_batch_size,
        cond_placement=cond_placement,
        normalize_routing=normalize_routing,
        routing_pooling=routing_pooling,
        top_k=top_k,
        routing_fn=routing_fn,
        l2=l2)
    all_routing_weights.extend([routing_weights_group_list])

  x = BatchNormalization(
      beta_regularizer=tf.keras.regularizers.l2(l2),
      gamma_regularizer=tf.keras.regularizers.l2(l2))(
          x)
  x = tf.keras.layers.Activation('relu')(x)
  if use_cond_dense:
    routing_weights = ed.layers.RoutingLayer(num_experts,
                                             normalize_dense_routing,
                                             routing_pooling, top_k,
                                             routing_fn)(
                                                 x)
    all_routing_weights.extend([routing_weights])
    x = tf.keras.layers.AveragePooling2D(pool_size=8)(x)
    x = tf.keras.layers.Flatten()(x)
    if reduce_dense_outputs:
      # `Tensor` of shape [batch_size, num_classes]
      dense_output = ed.layers.CondDense(
          num_classes, num_experts=num_experts)(x, routing_weights)
    else:
      dense_output = tf.keras.layers.Dense(num_classes * num_experts)(x)
      # `Tensor` of shape [batch_size, num_experts, num_classes]
      dense_output = tf.reshape(dense_output, [-1, num_experts, num_classes])
  else:
    x = tf.keras.layers.AveragePooling2D(pool_size=8)(x)
    x = tf.keras.layers.Flatten()(x)
    # `Tensor` of shape [batch_size, num_classes]
    dense_output = tf.keras.layers.Dense(num_classes)(x)

  outputs = (dense_output, all_routing_weights)
  return tf.keras.Model(inputs=inputs, outputs=outputs)

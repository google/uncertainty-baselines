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

"""Wide ResNet architecture with multiple input and outputs."""
import functools
import tensorflow as tf

BatchNormalization = functools.partial(  # pylint: disable=invalid-name
    tf.keras.layers.BatchNormalization,
    epsilon=1e-5,  # using epsilon and momentum defaults from Torch
    momentum=0.9)
Conv2D = functools.partial(  # pylint: disable=invalid-name
    tf.keras.layers.Conv2D,
    kernel_size=3,
    padding='same',
    use_bias=False,
    kernel_initializer='he_normal')


def basic_block(inputs, filters, strides):
  """Basic residual block of two 3x3 convs."""

  x = inputs
  y = inputs
  y = BatchNormalization()(y)
  y = tf.keras.layers.Activation('relu')(y)
  y = Conv2D(filters, strides=strides)(y)
  y = BatchNormalization()(y)
  y = tf.keras.layers.Activation('relu')(y)
  y = Conv2D(filters, strides=1)(y)
  if not x.shape.is_compatible_with(y.shape):
    x = Conv2D(filters, kernel_size=1, strides=strides)(x)

  x = tf.keras.layers.add([x, y])
  return x


def group(inputs, filters, strides, num_blocks, **kwargs):
  """Group of residual blocks."""
  x = basic_block(inputs, filters=filters, strides=strides, **kwargs)
  for _ in range(num_blocks - 1):
    x = basic_block(x, filters=filters, strides=1, **kwargs)
  return x


class BIMO(tf.keras.Model):
  """Keras module describing a Bayesian MIMO model."""

  def __init__(self,
               in_head,
               trunk,
               out_head,
               input_shape,
               num_members,
               name='bimo'):
    super(BIMO, self).__init__(name=name)
    self.in_heads = []
    self.out_heads = []
    self.trunk = None
    self.num_members = num_members

    inputs = tf.keras.Input(shape=input_shape)
    for k in range(num_members):
      in_head_out = in_head(inputs)
      self.in_heads.append(
          tf.keras.Model(inputs, in_head_out, name='in_head_%d' % k))

      if self.trunk is None:
        trunk_out = trunk(in_head_out)
        self.trunk = tf.keras.Model(in_head_out, trunk_out, name='trunk')
      else:
        trunk_out = self.trunk(in_head_out)

      out_head_out = out_head(trunk_out)
      self.out_heads.append(
          tf.keras.Model(trunk_out, out_head_out, name='out_head_%d' % k))

  def call(self, inputs):
    outs = []
    for ih, oh in zip(self.in_heads, self.out_heads):
      outs.append(oh(self.trunk(ih(inputs))))
    return tf.stack(outs, axis=0)

  def member_vars(self, i):
    """Returns a tuple of lists of variables in a specific BIMO head.

    A 3-tuple is returned containing a list of input head variables, a list
    of trunk variables, and a list of output head variables.

    Args:
      i: int, the index of the ensemble member.
    Returns:
      A three tuple containing the requested ensemble members in_head, trunk,
      and out_head variables.
    """
    return (self.in_heads[i].trainable_variables,
            self.trunk.trainable_variables,
            self.out_heads[i].trainable_variables)

  def all_vars(self):
    """Returns all variables in this BIMO model.

    A 3-tuple is returned containing the in head variables, trunk variables,
    and out head variables. The in head variables and out head variables are
    contained in lists of lists -- the outer list is of length number of
    variables and each element in that list is a list of length num ensemble
    members containing the individual member variables. So in_vars[i][j]
    contains the ith variable of the jth ensemble member.
    """
    num_in_vars = len(self.in_heads[0].trainable_variables)
    in_vars = [[self.in_heads[i].trainable_variables[j]
                for i in range(self.num_members)]
               for j in range(num_in_vars)]
    num_out_vars = len(self.out_heads[0].trainable_variables)
    out_vars = [[self.out_heads[i].trainable_variables[j]
                 for i in range(self.num_members)]
                for j in range(num_out_vars)]

    return (in_vars, self.trunk.trainable_variables, out_vars)

  def pair_grads_and_vars(self, grads):
    flat_vs = self.flatten_vars(self.all_vars())
    flat_gs = self.flatten_vars(grads)
    return zip(flat_gs, flat_vs)

  def flatten_vars(self, vs):
    in_vs, trunk_vs, out_vs = vs
    flat_in_vs = []
    flat_out_vs = []

    for member_list in in_vs:
      flat_in_vs.extend(member_list)

    for member_list in out_vs:
      flat_out_vs.extend(member_list)

    return flat_in_vs + trunk_vs + flat_out_vs


class MIMOBIMO(tf.keras.Model):
  """A BIMO model with MIMO architecture -- all input heads feed in to trunk."""

  def __init__(self,
               in_head,
               trunk,
               out_head,
               input_shape,
               num_members,
               name='mimo_bimo'):
    super(MIMOBIMO, self).__init__(name=name)
    self.in_heads = []
    self.out_heads = []
    self.num_members = num_members

    inputs = tf.keras.Input(shape=input_shape)
    in_head_outs = []
    for k in range(num_members):
      in_head_out = in_head(inputs)
      self.in_heads.append(
          tf.keras.Model(inputs, in_head_out, name='in_head_%d' % k))
      in_head_outs.append(in_head_out)

    in_head_out_concat = tf.keras.layers.Concatenate(axis=-1)(in_head_outs)
    trunk_out = trunk(in_head_out_concat)
    self.trunk = tf.keras.Model(in_head_outs, trunk_out, name='trunk')

    for k in range(num_members):
      out_head_out = out_head(trunk_out)
      self.out_heads.append(
          tf.keras.Model(trunk_out, out_head_out, name='out_head_%d' % k))

  def call(self, inputs):
    in_head_outs = []
    for ih in self.in_heads:
      in_head_outs.append(ih(inputs))

    trunk_out = self.trunk(in_head_outs)
    out_head_outs = []
    for oh in self.out_heads:
      out_head_outs.append(oh(trunk_out))
    return tf.stack(out_head_outs, axis=0)

  def member_vars(self, i):
    """Returns a tuple of lists of variables in a specific BIMO head.

    A 3-tuple is returned containing a list of input head variables, a list
    of trunk variables, and a list of output head variables.

    Args:
      i: int, the index of the ensemble member.
    Returns:
      A three tuple containing the requested ensemble members in_head, trunk,
      and out_head variables.
    """
    return (self.in_heads[i].trainable_variables,
            self.trunk.trainable_variables,
            self.out_heads[i].trainable_variables)

  def all_vars(self):
    """Returns all variables in this BIMO model.

    A 3-tuple is returned containing the in head variables, trunk variables,
    and out head variables. The in head variables and out head variables are
    contained in lists of lists -- the outer list is of length number of
    variables and each element in that list is a list of length num ensemble
    members containing the individual member variables. So in_vars[i][j]
    contains the ith variable of the jth ensemble member.
    """
    num_in_vars = len(self.in_heads[0].trainable_variables)
    in_vars = [[self.in_heads[i].trainable_variables[j]
                for i in range(self.num_members)]
               for j in range(num_in_vars)]
    num_out_vars = len(self.out_heads[0].trainable_variables)
    out_vars = [[self.out_heads[i].trainable_variables[j]
                 for i in range(self.num_members)]
                for j in range(num_out_vars)]

    return (in_vars, self.trunk.trainable_variables, out_vars)

  def pair_grads_and_vars(self, grads):
    flat_vs = self.flatten_vars(self.all_vars())
    flat_gs = self.flatten_vars(grads)
    return zip(flat_gs, flat_vs)

  def flatten_vars(self, vs):
    in_vs, trunk_vs, out_vs = vs
    flat_in_vs = []
    flat_out_vs = []

    for member_list in in_vs:
      flat_in_vs.extend(member_list)

    for member_list in out_vs:
      flat_out_vs.extend(member_list)

    return flat_in_vs + trunk_vs + flat_out_vs


def make_model(
    input_shape,
    depth,
    width_multiplier,
    num_classes,
    ensemble_size,
    model_name):
  """Builds Wide ResNet with Sparse BatchEnsemble.

  Following Zagoruyko and Komodakis (2016), it accepts a width multiplier on the
  number of filters. Using three groups of residual blocks, the network maps
  spatial features of size 32x32 -> 16x16 -> 8x8.

  Args:
    input_shape: tf.Tensor. The input shape must be (width, height, channels).
    depth: Total number of convolutional layers. "n" in WRN-n-k. It differs from
      He et al. (2015)'s notation which uses the maximum depth of the network
      counting non-conv layers like dense.
    width_multiplier: Integer to multiply the number of typical filters by. "k"
      in WRN-n-k.
    num_classes: Number of output classes.
    ensemble_size: Number of ensemble members.
    model_name: The name of the model.

  Returns:
    tf.keras.Model.
  """
  if (depth - 4) % 6 != 0:
    raise ValueError('depth should be 6n+4 (e.g., 16, 22, 28, 40).')
  num_blocks = (depth - 4) // 6
  input_shape = list(input_shape)
  assert len(input_shape) == 3

  def input_head(x):
    x = Conv2D(16, strides=1)(x)
    x = group(
        x,
        filters=16 * width_multiplier,
        strides=1,
        num_blocks=num_blocks)
    return x

  def trunk(x):
    for strides, filters in zip([2, 2], [32, 64]):
      x = group(
          x,
          filters=filters * width_multiplier,
          strides=strides,
          num_blocks=num_blocks)

    x = BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.AveragePooling2D(pool_size=8)(x)
    x = tf.keras.layers.Flatten()(x)
    return x

  def output_head(x):
    x = tf.keras.layers.Dense(
        num_classes,
        kernel_initializer='he_normal',
        activation=None)(x)
    return x

  if model_name == 'bimo':
    model = BIMO(input_head, trunk, output_head, input_shape, ensemble_size)
  elif model_name == 'mimobimo':
    model = MIMOBIMO(input_head, trunk, output_head, input_shape, ensemble_size)
  return model

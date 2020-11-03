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

"""Hyper-BatchEnsemble for a Wide ResNet architecture."""

import functools
import warnings
import tensorflow as tf

try:
  import edward2 as ed  # pylint: disable=g-import-not-at-top
except ImportError as err:
  warnings.warn(f'Skipped due to ImportError: {err}')


ENCODING = ('proj_log', 'sigmoid')
DEFAULT_ENCODING = 'sigmoid'
DISTRIBUTION = ('log_uniform', 'uniform')
DEFAULT_DISTRIBUTION = 'log_uniform'


class LogScaler(tf.keras.layers.Layer):
  """Layer that applies log transformation of hyperparameters in lambdas."""

  def __init__(self, ranges, **kwargs):
    super(LogScaler, self).__init__(**kwargs)
    self.ranges = ranges
    tf_log = tf.math.log
    self.log_min = tf.convert_to_tensor([[tf_log(r['min']) for r in ranges]])
    self.log_max = tf.convert_to_tensor([[tf_log(r['max']) for r in ranges]])

  def build(self, input_shape):
    pass

  def call(self, inputs):
    log_inputs = tf.math.log(inputs)
    return (log_inputs - self.log_min)/(self.log_max - self.log_min)

  def get_config(self):
    config = {
        'ranges': self.ranges
    }

    new_config = super(LogScaler, self).get_config()
    new_config.update(config)
    return new_config


class LambdaConfig:
  """Description of the hyperparameters in lambdas."""

  def __init__(self, ranges, key_to_index,
               dist=DEFAULT_DISTRIBUTION, encoding=DEFAULT_ENCODING):

    super(LambdaConfig, self).__init__()

    assert encoding in ENCODING
    assert dist in DISTRIBUTION

    self.ranges = ranges
    self.key_to_index = key_to_index
    self.encoding = encoding
    self.dist = dist

    tf_log = tf.math.log

    self.log_min = tf.convert_to_tensor([[tf_log(r['min']) for r in ranges]])
    self.log_max = tf.convert_to_tensor([[tf_log(r['max']) for r in ranges]])

    self.min_ = tf.convert_to_tensor([[r['min'] for r in ranges]])
    self.max_ = tf.convert_to_tensor([[r['max'] for r in ranges]])

    self.input_shape = (len(ranges),)
    self.dim = len(ranges)

    if encoding == 'sigmoid':
      sig = tf.math.sigmoid
      self._to_lambdas = lambda z: (self.max_ - self.min_) * sig(z) + self.min_
      self._to_z = lambda l: tf.math.log((l - self.min_)/(self.max_ - l))
      self._proj_z = lambda z: z
    elif encoding == 'proj_log':
      minimum = tf.math.minimum
      maximum = tf.math.maximum
      self._to_lambdas = tf.math.exp
      self._to_z = tf.math.log
      self._proj_z = lambda z: maximum(minimum(z, self.log_max), self.log_min)

  def sample(self, batchsize):

    samples = tf.random.uniform(shape=(batchsize, len(self.ranges)))
    if self.dist == 'log_uniform':
      samples = (self.log_max - self.log_min) * samples + self.log_min
      samples = tf.math.exp(samples)
    elif self.dist == 'uniform':
      samples = (self.max_ - self.min_) * samples + self.min_

    return samples

  def to_lambdas(self, z):
    return self._to_lambdas(z)

  def to_z(self, lambdas):
    return self._to_z(lambdas)

  def proj_z(self, z):
    return self._proj_z(z)

  def get_config(self):
    """Returns the config of the lambdas_config object.

    This config is a Python dictionary (serializable)
    containing the configuration of a lambdas_config object.
    """
    config = {
        'ranges': self.ranges,
        'key_to_index': self.key_to_index,
        'encoding': self.encoding,
        'dist': self.dist
    }
    return config


def e_factory(lambdas_input_shape, e_head_dims,
              e_body_arch=tuple(), e_shared_arch=tuple(),
              activation='tanh',
              use_bias=True,
              e_head_init=-1):
  """Factory to build the mlp embedding models for lambdas."""

  reg = tf.keras.regularizers.L2(l2=0.)

  if e_head_init <= 0:
    head_kernel_initializer = 'glorot_uniform'
  else:
    head_kernel_initializer = tf.keras.initializers.RandomNormal(
        mean=0.0, stddev=e_head_init)

  prefix = 'dense_e'

  log_lambdas_input = tf.keras.layers.Input(shape=lambdas_input_shape)
  out = log_lambdas_input

  for index, units in enumerate(e_shared_arch):
    out = tf.keras.layers.Dense(
        units,
        name=prefix + '_shared_{}'.format(index),
        activation=activation,
        kernel_regularizer=reg,
        bias_regularizer=reg)(out)

  common_out = out
  e = []
  for i, e_head_dim in enumerate(e_head_dims):
    out = common_out
    for j, units in enumerate(e_body_arch):
      out = tf.keras.layers.Dense(
          units,
          name=prefix+'_{}_{}'.format(j, i),
          activation=activation,
          kernel_regularizer=reg,
          bias_regularizer=reg)(out)

    out = tf.keras.layers.Dense(
        e_head_dim,
        name=prefix+'_{}_{}'.format(len(e_body_arch), i),
        activation='linear',
        use_bias=use_bias,
        kernel_regularizer=reg,
        bias_regularizer=reg,
        kernel_initializer=head_kernel_initializer)(out)

    e.append(tf.keras.models.Model(inputs=log_lambdas_input, outputs=out))

  return e


def make_sign_initializer(random_sign_init):
  """Random sign intitializer for HyperBatchEnsemble layers."""
  if random_sign_init > 0:
    return ed.initializers.RandomSign(random_sign_init)
  else:
    return tf.keras.initializers.RandomNormal(
        mean=1.0, stddev=-random_sign_init)


def wide_resnet_hyperbatchensemble(input_shape,
                                   depth,
                                   width_multiplier,
                                   num_classes,
                                   ensemble_size,
                                   random_sign_init,
                                   config,
                                   e_models,
                                   l2_batchnorm_layer=15,
                                   regularize_fast_weights=False,
                                   fast_weights_eq_contraint=True,
                                   version=2):
  """Builds Hyper-Batch Ensemble Wide ResNet.

  Following Zagoruyko and Komodakis (2016), it accepts a width multiplier on the
  number of filters. Using three groups of residual blocks, the network maps
  spatial features of size 32x32 -> 16x16 -> 8x8.

   Uses the following hyperparameters:
    * l2 reg. parameter for input conv layer (doesn't use bias)
    * separate l2 reg. parameter for each group of blocks (don't use biases)
      * this results in 3 parameters
      * to get more flexibility, we could use separate l2 parameter for each
        block in each group
    * l2 reg. parameter for kernel and bias of final dense layer
    * Fixed l2 reg. parameter for BatchNorm layers
    --> In total we have 6 self-tuned hyperparameters

  Args:
    input_shape: tf.Tensor.
    depth: Total number of convolutional layers. "n" in WRN-n-k. It differs from
      He et al. (2015)'s notation which uses the maximum depth of the network
      counting non-conv layers like dense.
    width_multiplier: Integer to multiply the number of typical filters by. "k"
      in WRN-n-k.
    num_classes: Number of output classes.
    ensemble_size: Only used by batch_stn layers.
    random_sign_init: Initializer used by batch_stn layers.
    config: lambdas_config.
    e_models: list of e_models.
    l2_batchnorm_layer: l2 parameter used for batchnorm layers.
    regularize_fast_weights: whether to regularize fast weights.
    fast_weights_eq_contraint: If true we impose (r_k, s_k) = (u_k, v_k).
    version: 1, indicating the original ordering from He et al. (2015); or 2,
      indicating the preactivation ordering from He et al. (2016).

  Returns:
    tf.keras.Model.
  """
  # INTERNAL FUNCTION AND LAYER DEFINITIONS

  BatchNormalization = functools.partial(  # pylint: disable=invalid-name
      tf.keras.layers.BatchNormalization,
      epsilon=1e-5,  # using epsilon and momentum defaults from Torch
      momentum=0.9)
  Conv2D = functools.partial(  # pylint: disable=invalid-name
      ed.layers.Conv2DHyperBatchEnsemble,
      config.key_to_index,
      ensemble_size=ensemble_size,
      alpha_initializer=make_sign_initializer(random_sign_init),
      gamma_initializer=make_sign_initializer(random_sign_init),
      kernel_size=3,
      padding='same',
      use_bias=False,
      kernel_initializer='he_normal',
      regularize_fast_weights=regularize_fast_weights,
      fast_weights_eq_contraint=fast_weights_eq_contraint)
  Dense = functools.partial(  # pylint: disable=invalid-name
      ed.layers.DenseHyperBatchEnsemble,
      ensemble_size=ensemble_size,
      alpha_initializer=make_sign_initializer(random_sign_init),
      gamma_initializer=make_sign_initializer(random_sign_init),
      kernel_initializer='he_normal',
      regularize_fast_weights=regularize_fast_weights,
      fast_weights_eq_contraint=fast_weights_eq_contraint)

  def basic_block(inputs, filters, strides, version, block_name,
                  l2_batchnorm_layer):
    """Basic residual block of two 3x3 convs.

    Args:
      inputs: tf.Tensor.
      filters: Number of filters for Conv2D.
      strides: Stride dimensions for Conv2D.
      version: 1, indicating the original ordering from He et al. (2015); or 2,
        indicating the preactivation ordering from He et al. (2016).
      block_name: Prefix for the name of all layers in this block.
      l2_batchnorm_layer: fixed l2 parameter used for batchnorm layers.

    Returns:
      tf.Tensor.
    """

    # unpack inputs
    data, lambdas, block_e_list = inputs
    assert len(block_e_list) == 2 or len(block_e_list) == 3, \
        'Length of e_list must match the number of layers in block.'
    x = data
    y = data

    if version == 2:
      y = BatchNormalization(
          beta_regularizer=tf.keras.regularizers.l2(l2_batchnorm_layer),
          gamma_regularizer=tf.keras.regularizers.l2(l2_batchnorm_layer))(y)
      y = tf.keras.layers.Activation('relu')(y)

    layer_index = 0
    layer_name = block_name + '_conv_{}/'.format(layer_index)
    y = Conv2D(
        filters, strides=strides,
        name=layer_name)([y, lambdas, block_e_list[layer_index]])

    y = BatchNormalization(
        beta_regularizer=tf.keras.regularizers.l2(l2_batchnorm_layer),
        gamma_regularizer=tf.keras.regularizers.l2(l2_batchnorm_layer))(y)
    y = tf.keras.layers.Activation('relu')(y)

    layer_index += 1
    layer_name = block_name + '_conv_{}/'.format(layer_index)
    y = Conv2D(
        filters, strides=1,
        name=layer_name)([y, lambdas, block_e_list[layer_index]])

    if version == 1:
      y = BatchNormalization(
          beta_regularizer=tf.keras.regularizers.l2(l2_batchnorm_layer),
          gamma_regularizer=tf.keras.regularizers.l2(l2_batchnorm_layer))(y)

    if not x.shape.is_compatible_with(y.shape):
      layer_index += 1
      layer_name = block_name + '_conv_{}/'.format(layer_index)
      x = Conv2D(
          filters, kernel_size=1, strides=strides,
          name=layer_name)([x, lambdas, block_e_list[layer_index]])

    x = tf.keras.layers.add([x, y])
    if version == 1:
      x = tf.keras.layers.Activation('relu')(x)
    return x

  def group(inputs, filters, strides, num_blocks, version, name,
            l2_batchnorm_layer):
    """Group of residual blocks."""

    # unpack inputs
    x, lambdas, group_e_list = inputs
    assert len(group_e_list) == 2 * num_blocks + 1, \
        'Length of e_list has to match the number of layers in group.'

    # first block
    block_name = name + '/block_{}'.format(0)
    block_e_list = group_e_list[:3]  # First block has 3 layers
    x = basic_block(
        [x, lambdas, block_e_list],
        filters=filters,
        strides=strides,
        version=version,
        block_name=block_name,
        l2_batchnorm_layer=l2_batchnorm_layer)

    # next blocks
    for i in range(1, num_blocks):
      block_name = name + '/block_{}'.format(i)
      block_e_list = group_e_list[1+2*i: 1+2*(i+1)]  # Next blocks have 2 layers
      x = basic_block(
          [x, lambdas, block_e_list],
          filters=filters,
          strides=1,
          version=version,
          block_name=block_name,
          l2_batchnorm_layer=l2_batchnorm_layer)
    return x

  # START BUILDING RESNET

  if (depth - 4) % 6 != 0:
    raise ValueError('depth should be 6n+4 (e.g., 16, 22, 28, 40).')
  num_blocks = (depth - 4) // 6
  num_layer_per_group = 2 * num_blocks + 1

  # inputs
  input_data = tf.keras.layers.Input(shape=input_shape)
  input_lambdas = tf.keras.layers.Input(shape=config.input_shape)
  log_lambdas = LogScaler(config.ranges)(input_lambdas)

  # input conv layer
  e = e_models[0]
  x = Conv2D(
      16,
      strides=1,
      name='input_conv')([input_data, input_lambdas, e(log_lambdas)])

  if version == 1:
    x = BatchNormalization(
        beta_regularizer=tf.keras.regularizers.l2(l2_batchnorm_layer),
        gamma_regularizer=tf.keras.regularizers.l2(l2_batchnorm_layer))(x)
    x = tf.keras.layers.Activation('relu')(x)

  e_range_group = range(1, 1+num_layer_per_group)
  group_e_list = [e_models[i](log_lambdas) for i in e_range_group]
  x = group([x, input_lambdas, group_e_list],
            filters=16 * width_multiplier,
            strides=1,
            num_blocks=num_blocks,
            version=version,
            name='group',
            l2_batchnorm_layer=l2_batchnorm_layer)

  e_range_group = range(1+num_layer_per_group, 1+2*num_layer_per_group)
  group_e_list = [e_models[i](log_lambdas) for i in e_range_group]
  x = group([x, input_lambdas, group_e_list],
            filters=32 * width_multiplier,
            strides=2,
            num_blocks=num_blocks,
            version=version,
            name='group_1',
            l2_batchnorm_layer=l2_batchnorm_layer)

  e_range_group = range(1+2*num_layer_per_group, 1+3*num_layer_per_group)
  group_e_list = [e_models[i](log_lambdas) for i in e_range_group]
  x = group([x, input_lambdas, group_e_list],
            filters=64 * width_multiplier,
            strides=2,
            num_blocks=num_blocks,
            version=version,
            name='group_2',
            l2_batchnorm_layer=l2_batchnorm_layer)

  if version == 2:
    x = BatchNormalization(
        beta_regularizer=tf.keras.regularizers.l2(l2_batchnorm_layer),
        gamma_regularizer=tf.keras.regularizers.l2(l2_batchnorm_layer))(x)
    x = tf.keras.layers.Activation('relu')(x)
  x = tf.keras.layers.AveragePooling2D(pool_size=8)(x)
  x = tf.keras.layers.Flatten()(x)

  assert 1+3*num_layer_per_group == len(e_models)-1
  e = e_models[1+3*num_layer_per_group](log_lambdas)
  x = Dense(
      num_classes,
      config.key_to_index,
      name='dense',
      activation=None)([x, input_lambdas, e])

  return tf.keras.Model(inputs=[input_data, input_lambdas], outputs=x)

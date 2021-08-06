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

# Lint as: python3
"""Wide Residual Posterior Network (see https://arxiv.org/abs/2006.09239)."""

import functools
from typing import Dict, Iterable, Optional

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

HP_KEYS = ('bn_l2', 'input_conv_l2', 'group_1_conv_l2', 'group_2_conv_l2',
           'group_3_conv_l2', 'dense_kernel_l2', 'dense_bias_l2')

BatchNormalization = functools.partial(  # pylint: disable=invalid-name
    tf.keras.layers.BatchNormalization,
    epsilon=1e-5,  # using epsilon and momentum defaults from Torch
    momentum=0.9)


def Conv2D(filters, seed=None, **kwargs):  # pylint: disable=invalid-name
  """Conv2D layer that is deterministically initialized."""
  default_kwargs = {
      'kernel_size': 3,
      'padding': 'same',
      'use_bias': False,
      # Note that we need to use the class constructor for the initializer to
      # get deterministic initialization.
      'kernel_initializer': tf.keras.initializers.HeNormal(seed=seed),
  }
  # Override defaults with the passed kwargs.
  default_kwargs.update(kwargs)
  return tf.keras.layers.Conv2D(filters, **default_kwargs)


def basic_block(
    inputs: tf.Tensor,
    filters: int,
    strides: int,
    conv_l2: float,
    bn_l2: float,
    seed: int,
    version: int) -> tf.Tensor:
  """Basic residual block of two 3x3 convs.

  Args:
    inputs: tf.Tensor.
    filters: Number of filters for Conv2D.
    strides: Stride dimensions for Conv2D.
    conv_l2: L2 regularization coefficient for the conv kernels.
    bn_l2: L2 regularization coefficient for the batch norm layers.
    seed: random seed used for initialization.
    version: 1, indicating the original ordering from He et al. (2015); or 2,
      indicating the preactivation ordering from He et al. (2016).

  Returns:
    tf.Tensor.
  """
  x = inputs
  y = inputs
  if version == 2:
    y = BatchNormalization(beta_regularizer=tf.keras.regularizers.l2(bn_l2),
                           gamma_regularizer=tf.keras.regularizers.l2(bn_l2))(y)
    y = tf.keras.layers.Activation('relu')(y)
  seeds = tf.random.experimental.stateless_split([seed, seed + 1], 3)[:, 0]
  y = Conv2D(filters,
             strides=strides,
             seed=seeds[0],
             kernel_regularizer=tf.keras.regularizers.l2(conv_l2))(y)
  y = BatchNormalization(beta_regularizer=tf.keras.regularizers.l2(bn_l2),
                         gamma_regularizer=tf.keras.regularizers.l2(bn_l2))(y)
  y = tf.keras.layers.Activation('relu')(y)
  y = Conv2D(filters,
             strides=1,
             seed=seeds[1],
             kernel_regularizer=tf.keras.regularizers.l2(conv_l2))(y)
  if version == 1:
    y = BatchNormalization(beta_regularizer=tf.keras.regularizers.l2(bn_l2),
                           gamma_regularizer=tf.keras.regularizers.l2(bn_l2))(y)
  if not x.shape.is_compatible_with(y.shape):
    x = Conv2D(filters,
               kernel_size=1,
               strides=strides,
               seed=seeds[2],
               kernel_regularizer=tf.keras.regularizers.l2(conv_l2))(x)
  x = tf.keras.layers.add([x, y])
  if version == 1:
    x = tf.keras.layers.Activation('relu')(x)
  return x


def group(inputs, filters, strides, num_blocks, conv_l2, bn_l2, version, seed):
  """Group of residual blocks."""
  seeds = tf.random.experimental.stateless_split(
      [seed, seed + 1], num_blocks)[:, 0]
  x = basic_block(
      inputs,
      filters=filters,
      strides=strides,
      conv_l2=conv_l2,
      bn_l2=bn_l2,
      version=version,
      seed=seeds[0])
  for i in range(num_blocks - 1):
    x = basic_block(
        x,
        filters=filters,
        strides=1,
        conv_l2=conv_l2,
        bn_l2=bn_l2,
        version=version,
        seed=seeds[i + 1])
  return x


def _parse_hyperparameters(l2: float, hps: Dict[str, float]):
  """Extract the L2 parameters for the dense, conv and batch-norm layers."""

  assert_msg = ('Ambiguous hyperparameter specifications: either l2 or hps '
                'must be provided (received {} and {}).'.format(l2, hps))
  is_specified = lambda h: bool(h) and all(v is not None for v in h.values())
  only_l2_is_specified = l2 is not None and not is_specified(hps)
  only_hps_is_specified = l2 is None and is_specified(hps)
  assert only_l2_is_specified or only_hps_is_specified, assert_msg
  if only_hps_is_specified:
    assert_msg = 'hps must contain the keys {}!={}.'.format(HP_KEYS, hps.keys())
    assert set(hps.keys()).issuperset(HP_KEYS), assert_msg
    return hps
  else:
    return {k: l2 for k in HP_KEYS}


class ReversedRadialFlow(tfp.bijectors.Bijector):
  """A radial flow, but backwards.

  The original radial flow (Rezende & Mohamed 2016) is not invertible.
  Link to paper: https://arxiv.org/abs/1505.05770

  Since in our particular application of posterior networks, we do not
  intend to sample from the model, we do not need the _forward() function.
  Instead, we only use it as a density estimator, so we need the _log_prob()
  function, which depends on the _inverse() pass and respective log det of the
  Jacobian. Hence, these functions are implemented the way that would
  normally be the forward pass through the flow.
  """

  def __init__(self,
               dim,
               x0=None,
               alpha_prime_init=None,
               beta_prime_init=None,
               validate_args=False,
               name='radial'):
    """Builds a reversed radial flow model.

    Args:
      dim(int): Dimensionality of the input and output space.
      x0(array,Tensor): Reference point in the input space.
      alpha_prime_init(float): Alpha' parameter for the radial flow.
      beta_prime_init(float): Beta' parameter for the radial flow.
      validate_args(bool): Flag to validate the arguments.
      name(str): Name for the model.
    """
    super().__init__(
        validate_args=validate_args,
        inverse_min_event_ndims=1,
        name=name)
    with tf.name_scope(name) as name:
      self._name = name
      if x0 is None:
        x0 = tf.zeros(dim)
      else:
        x0 = tf.convert_to_tensor(x0)
        if x0.shape[-1] != tf.TensorShape(dim):
          raise ValueError(f'Variable x0={x0} needs to have shape [{dim}]. '
                           f'Found shape {x0.shape}')
      self.x0 = tf.Variable(x0, name='x0', dtype=tf.float32, trainable=True)
      # if alpha' and beta' are not defined, we sample them the same way
      # that Pyro does (https://docs.pyro.ai/en/latest/_modules/pyro/
      # distributions/transforms/radial.html)
      if alpha_prime_init is None:
        alpha_prime_init = tf.random.uniform([], -1 / np.sqrt(dim),
                                             1 / np.sqrt(dim))
      if beta_prime_init is None:
        beta_prime_init = tf.random.uniform([], -1/np.sqrt(dim),
                                            1/np.sqrt(dim))
      self.alpha_prime = tf.Variable(
          alpha_prime_init, name='alpha', dtype=tf.float32, trainable=True)
      self.beta_prime = tf.Variable(
          beta_prime_init, name='beta', dtype=tf.float32, trainable=True)
      self.dim = dim

  def _forward(self, z):
    """The normal radial flow is not invertible, so this is not defined."""
    raise NotImplementedError("Forward shouldn't be called!")

  def _get_alpha_beta(self):
    """Regularize alpha' and beta' to get alpha and beta."""
    alpha = tf.nn.softplus(self.alpha_prime)
    beta = -alpha + tf.nn.softplus(self.beta_prime)
    return alpha, beta

  def _inverse(self, x):
    """The forward pass of the original radial flow, following the paper.

    This is the first line of Eq. (14) in https://arxiv.org/abs/1505.05770.
    Args:
      x(Tensor): Input to the flow.
    Returns:
      The transformed output tensor of the flow.
    """
    alpha, beta = self._get_alpha_beta()
    diff = x - self.x0
    r = tf.linalg.norm(diff, axis=-1, keepdims=True)
    h = 1. / (alpha + r)
    beta_h = beta * h
    return x + beta_h * diff

  def _inverse_log_det_jacobian(self, x):
    """Computes the log det Jacobian, as per the paper.

    This is the second line of Eq. (14) in https://arxiv.org/abs/1505.05770.
    Args:
      x(Tensor): Input to the flow.
    Returns:
      The log determinant of the Jacobian of the flow.
    """
    alpha, beta = self._get_alpha_beta()
    diff = x - self.x0
    r = tf.linalg.norm(diff, axis=-1, keepdims=True)
    h = 1. / (alpha + r)
    h_prime = -(h ** 2)
    beta_h = beta * h
    log_det_jacobian = tf.reduce_sum(
        (self.dim - 1) * tf.math.log1p(beta_h)
        + tf.math.log1p(beta_h + beta * h_prime * r), axis=-1)
    return log_det_jacobian


def _make_deep_flow(flow_type, flow_depth, flow_width, dim):
  """Builds a deep flow of the specified type."""
  if flow_type not in ['maf', 'radial', 'affine']:
    raise ValueError(f'Flow type {flow_type} is not maf, radial, or affine.')
  if flow_type == 'maf':
    return _make_maf_flow(flow_depth, flow_width)
  elif flow_type == 'radial':
    return _make_radial_flow(dim, flow_depth)
  elif flow_type == 'affine':
    return _make_affine_flow(dim, flow_depth)


def _make_maf_flow(flow_depth, flow_width):
  """Builds a deep stack of masked autoregressive flows."""
  # If not otherwise specified, make the hidden layers of the flow twice
  # as wide as the latent dimension, to make them expressive enough to
  # parameterize a shift and scale for each dimension.
  bijectors = []
  bijectors.append(tfp.bijectors.BatchNormalization())
  # Build the deep MAF flow.
  # Each layer outputs two params per dimension, for shift and scale.
  bijectors.append(
      tfp.bijectors.MaskedAutoregressiveFlow(
          tfp.bijectors.AutoregressiveNetwork(
              params=2, hidden_units=[flow_width]*flow_depth,
              activation='relu')))
  # For numerical stability of training, we need these batch norms.
  bijectors.append(tfp.bijectors.BatchNormalization())
  return tfp.bijectors.Chain(list(reversed(bijectors)))


def _make_radial_flow(dim, flow_depth):
  """Builds a deep stack of radial flows."""
  bijectors = []
  bijectors.append(tfp.bijectors.BatchNormalization())
  for _ in range(flow_depth):
    bijectors.append(ReversedRadialFlow(dim))
  bijectors.append(tfp.bijectors.BatchNormalization())
  return tfp.bijectors.Chain(list(reversed(bijectors)))


def _make_affine_flow(dim, flow_depth):
  """Builds a deep stack of affine flows."""
  bijectors = []
  bijectors.append(tfp.bijectors.BatchNormalization())
  for _ in range(flow_depth):
    bijectors.append(
        tfp.bijectors.Shift(tf.Variable(tf.zeros(dim), trainable=True)))
    bijectors.append(
        tfp.bijectors.ScaleMatvecDiag(
            tf.Variable(tf.ones(dim), trainable=True)))
  bijectors.append(tfp.bijectors.BatchNormalization())
  return tfp.bijectors.Chain(list(reversed(bijectors)))


# TODO(fortuin): refactor this into ed.layers
class PosteriorNetworkLayer(tf.keras.layers.Layer):
  """Output layer for a Posterior Network model."""

  def __init__(self,
               num_classes,
               flow_type='maf',
               flow_depth=8,
               flow_width=None,
               class_counts=None,
               name='PosteriorNetworkLayer'):
    """Makes a Posterior Network output layer.

    Args:
      num_classes: Number of output classes.
      flow_type: Type of the normalizing flow to be used; has to be one
                      of 'maf', 'radial', or 'affine'.
      flow_depth: Number of latent flows to stack into a deep flow.
      flow_width: Width of the hidden layers inside the MAF flows.
      class_counts: List of counts of training examples per class.
      name: Name of the layer.
    """
    super().__init__(name=name)
    self.num_classes = num_classes
    self.flow_type = flow_type
    self.flow_depth = flow_depth
    self.flow_width = flow_width
    if class_counts is None:
      class_counts = tf.ones(num_classes)
    self.class_counts = class_counts

  def build(self, input_shape):
    """Builds the layer based on the passed input shape."""
    with tf.name_scope(self.name):
      # Using the PyTorch default hyperparameters.
      self.batch_norm = tf.keras.layers.BatchNormalization(epsilon=1e-5,
                                                           momentum=0.9)
      self.latent_dim = input_shape[-1]
      if self.flow_width is None:
        self.flow_width = 2 * self.latent_dim
      self.flows = []
      for _ in range(self.num_classes):
        flow = tfp.distributions.TransformedDistribution(
            distribution=tfp.distributions.MultivariateNormalDiag(
                loc=tf.zeros(self.latent_dim, dtype=tf.float32),
                scale_diag=tf.ones(self.latent_dim, dtype=tf.float32)),
            bijector=_make_deep_flow(self.flow_type,
                                     self.flow_depth,
                                     self.flow_width,
                                     self.latent_dim))
        self.flows.append(flow)

  def call(self, inputs, training=True, return_probs=False):
    latents = self.batch_norm(inputs)
    log_ps = [self.flows[i].log_prob(latents) for i in range(self.num_classes)]
    log_ps_stacked = tf.stack(log_ps, axis=-1)
    alphas = 1. + (self.class_counts * tf.exp(log_ps_stacked))
    probs, _ = tf.linalg.normalize(alphas, ord=1, axis=-1)
    if return_probs:
      return probs
    else:
      return alphas

  def get_config(self):
    config = super().get_config()
    config.update({
        'num_classes': self.num_classes,
        'flow_type': self.flow_type,
        'flow_depth': self.flow_depth,
        'flow_width': self.flow_width,
        'class_counts': self.class_counts
    })
    return config


# TODO(fortuin): refactor this into ed.losses
def uce_loss(entropy_reg=1e-5, sparse=True,
             return_all_loss_terms=False, num_classes=None):
  """Computes the UCE loss, either from sparse or dense labels."""
  if sparse and num_classes is None:
    raise ValueError('Number of classes must be defined for the sparse loss.')
  if sparse:
    return functools.partial(_sparse_uce_loss,
                             entropy_reg=entropy_reg,
                             return_all_loss_terms=return_all_loss_terms,
                             num_classes=num_classes)
  else:
    return functools.partial(_uce_loss,
                             entropy_reg=entropy_reg,
                             return_all_loss_terms=return_all_loss_terms)


def _uce_loss(labels, alpha, entropy_reg=1e-5, return_all_loss_terms=False):
  """UCE loss, as in the Posterior Network paper.

  Args:
    labels: Numpy array or Tensor of true labels for the data points.
    alpha: Predicted Dirichlet parameters for the data points.
    entropy_reg: Entropy regularizer to weigh the entropy term in the loss.
    return_all_loss_terms: Flag to return the separate loss terms.

  Returns:
    Either the scalar total loss or a tuple of the different loss terms.
  """
  # This computes the normalizer of the Dirichlet distribution.
  alpha_0 = tf.reduce_sum(alpha, axis=-1, keepdims=True) * tf.ones_like(alpha)
  # This computes the Dichlet entropy.
  entropy = _dirichlet_entropy(alpha)
  # This computes the expected cross-entropy under the Dirichlet.
  # (Eq. 10 in the paper)
  ce_loss = tf.reduce_sum(
      (tf.math.digamma(alpha_0) - tf.math.digamma(alpha)) * labels, axis=-1)
  # The UCE loss is E_q[CE(p,y)] - lambda * H(q).
  loss = ce_loss - entropy_reg * entropy
  if return_all_loss_terms:
    return tf.reduce_mean(loss), tf.reduce_mean(ce_loss), tf.reduce_mean(
        entropy)
  else:
    return tf.reduce_mean(loss)


def _sparse_uce_loss(labels,
                     alpha,
                     num_classes,
                     entropy_reg=1e-5,
                     return_all_loss_terms=False):
  """UCE loss with sparse labels. Same args as self.uce_loss."""
  labels_one_hot = tf.one_hot(tf.cast(labels, tf.int32), num_classes)
  return _uce_loss(labels_one_hot, alpha, entropy_reg, return_all_loss_terms)


def _dirichlet_entropy(alpha):
  """Computes the entropy of a Dirichlet distribution (Eq. 11 in paper)."""
  k = tf.cast(tf.shape(alpha)[-1], alpha.dtype)
  total_concentration = tf.reduce_sum(alpha, axis=-1)
  entropy = (
      tf.math.lbeta(alpha) +
      ((total_concentration - k) * tf.math.digamma(total_concentration)) -
      tf.reduce_sum((alpha - 1.) * tf.math.digamma(alpha), axis=-1))
  return entropy


def wide_resnet_posterior_network(
    input_shape: Iterable[int],
    depth: int,
    width_multiplier: int,
    num_classes: int,
    l2: float,
    version: int = 2,
    seed: int = 42,
    class_counts: Optional[Iterable[int]] = None,
    latent_dim: int = 16,
    flow_depth: int = 6,
    flow_width: Optional[int] = None,
    flow_type: str = 'maf',
    hps: Optional[Dict[str, float]] = None) -> tf.keras.models.Model:
  """Builds Wide ResNet Posterior Network.

  Following Zagoruyko and Komodakis (2016), it accepts a width multiplier on the
  number of filters. Using three groups of residual blocks, the network maps
  spatial features of size 32x32 -> 16x16 -> 8x8.

  Args:
    input_shape: tf.Tensor storing the shape of the inputs.
    depth: Total number of convolutional layers. "n" in WRN-n-k. It differs from
      He et al. (2015)'s notation which uses the maximum depth of the network
      counting non-conv layers like dense.
    width_multiplier: Integer to multiply the number of typical filters by. "k"
      in WRN-n-k.
    num_classes: Number of output classes.
    l2: L2 regularization coefficient.
    version: 1, indicating the original ordering from He et al. (2015); or 2,
      indicating the preactivation ordering from He et al. (2016).
    seed: random seed used for initialization.
    class_counts: List of counts of training examples per class.
    latent_dim: Dimensionality of the latent space.
    flow_depth: Number of latent flows to stack into a deep flow.
    flow_width: Width of the hidden layers inside the MAF flows.
    flow_type: Type of the normalizing flow to be used; has to be one
                    of 'maf', 'radial', or 'affine'.
    hps: Fine-grained specs of the hyperparameters, as a Dict[str, float].

  Returns:
    tf.keras.Model.
  """
  l2_reg = tf.keras.regularizers.l2
  hps = _parse_hyperparameters(l2, hps)

  seeds = tf.random.experimental.stateless_split([seed, seed + 1], 5)[:, 0]
  if (depth - 4) % 6 != 0:
    raise ValueError('depth should be 6n+4 (e.g., 16, 22, 28, 40).')
  num_blocks = (depth - 4) // 6

  if class_counts is None:
    class_counts = tf.ones(num_classes)
  class_counts = tf.convert_to_tensor(class_counts, dtype=tf.float32)

  inputs = tf.keras.layers.Input(shape=input_shape)
  x = Conv2D(16,
             strides=1,
             seed=seeds[0],
             kernel_regularizer=l2_reg(hps['input_conv_l2']))(inputs)
  if version == 1:
    x = BatchNormalization(beta_regularizer=l2_reg(hps['bn_l2']),
                           gamma_regularizer=l2_reg(hps['bn_l2']))(x)
    x = tf.keras.layers.Activation('relu')(x)
  x = group(x,
            filters=16 * width_multiplier,
            strides=1,
            num_blocks=num_blocks,
            conv_l2=hps['group_1_conv_l2'],
            bn_l2=hps['bn_l2'],
            version=version,
            seed=seeds[1])
  x = group(x,
            filters=32 * width_multiplier,
            strides=2,
            num_blocks=num_blocks,
            conv_l2=hps['group_2_conv_l2'],
            bn_l2=hps['bn_l2'],
            version=version,
            seed=seeds[2])
  x = group(x,
            filters=64 * width_multiplier,
            strides=2,
            num_blocks=num_blocks,
            conv_l2=hps['group_3_conv_l2'],
            bn_l2=hps['bn_l2'],
            version=version,
            seed=seeds[3])
  if version == 2:
    x = BatchNormalization(beta_regularizer=l2_reg(hps['bn_l2']),
                           gamma_regularizer=l2_reg(hps['bn_l2']))(x)
    x = tf.keras.layers.Activation('relu')(x)
  x = tf.keras.layers.AveragePooling2D(pool_size=8)(x)
  x = tf.keras.layers.Flatten()(x)

  latents = tf.keras.layers.Dense(latent_dim)(x)
  postnet_layer = PosteriorNetworkLayer(num_classes=num_classes,
                                        flow_type=flow_type,
                                        flow_depth=flow_depth,
                                        flow_width=flow_width,
                                        class_counts=class_counts)
  alphas = postnet_layer(latents)

  return tf.keras.Model(
      inputs=inputs,
      outputs=alphas,
      name='wide_resnet-posterior_network-{}-{}'.format(depth,
                                                        width_multiplier))

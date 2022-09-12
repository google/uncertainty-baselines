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

"""Implementation of ResNetV1 with group norm and weight standardization.

Ported from:
https://github.com/google-research/scenic/blob/main/scenic/projects/baselines/bit_resnet.py.
"""

from typing import Any, Dict, Optional, Sequence, Tuple, Union

import flax.linen as nn
import jax
import jax.numpy as jnp


Array = Any
PRNGKey = Any
Shape = Tuple[int]
Dtype = Any

################################################################################
################################################################################

MIN_SCALE_MONTE_CARLO = 1e-3


# TODO(markcollier): migrate to het library once API is stabilized.
class MCSigmoidDenseFA(nn.Module):
  """Sigmoid and factor analysis approx to heteroscedastic predictions.

  if we assume:
  u ~ N(mu(x), sigma(x))
  and
  y = sigmoid(u / temperature)

  we can do a low rank approximation of sigma(x) the full rank matrix as:
  eps_R ~ N(0, I_R), eps_K ~ N(0, I_K)
  u = mu(x) + matmul(V(x), e) + d(x) * e_d
  where V(x) is a matrix of dimension [num_outputs, R=num_factors]
  and d(x) is a vector of dimension [num_outputs, 1]
  num_factors << num_outputs => approx to sampling ~ N(mu(x), sigma(x)).
  """

  num_outputs: int
  num_factors: int  # set num_factors = 0 for diagonal method
  temperature: float = 1.0
  temp_lower: float = 0.05
  temp_upper: float = 5.0
  parameter_efficient: bool = False
  train_mc_samples: int = 1000
  test_mc_samples: int = 1000
  share_samples_across_batch: bool = False
  logits_only: bool = False
  return_locs: bool = False
  eps: float = 1e-7

  def setup(self):
    if self.parameter_efficient:
      self._scale_layer_homoscedastic = nn.Dense(
          self.num_outputs, name='scale_layer_homoscedastic')
      self._scale_layer_heteroscedastic = nn.Dense(
          self.num_outputs, name='scale_layer_heteroscedastic')
    elif self.num_factors > 0:
      self._scale_layer = nn.Dense(
          self.num_outputs * self.num_factors, name='scale_layer')

    self._loc_layer = nn.Dense(self.num_outputs, name='loc_layer')
    self._diag_layer = nn.Dense(self.num_outputs, name='diag_layer')

    if self.temperature > 0:
      self._temperature_pre_sigmoid = None
    else:
      self._temperature_pre_sigmoid = self.param('temperature_pre_sigmoid',
                                                 nn.initializers.zeros, (1,))

  def _compute_loc_param(self, inputs):
    """Computes location parameter of the "logits distribution".

    Args:
      inputs: Tensor. The input to the heteroscedastic output layer.

    Returns:
      Tensor of shape [batch_size, num_classes].
    """
    return self._loc_layer(inputs)

  def _compute_scale_param(self, inputs):
    """Computes scale parameter of the "logits distribution".

    Args:
      inputs: Tensor. The input to the heteroscedastic output layer.

    Returns:
      Tuple of tensors of shape
      ([batch_size, num_classes * max(num_factors, 1)],
      [batch_size, num_classes]).
    """
    if self.parameter_efficient or self.num_factors <= 0:
      return (inputs,
              jax.nn.softplus(self._diag_layer(inputs)) + MIN_SCALE_MONTE_CARLO)
    else:
      return (self._scale_layer(inputs),
              jax.nn.softplus(self._diag_layer(inputs)) + MIN_SCALE_MONTE_CARLO)

  def _compute_diagonal_noise_samples(self, diag_scale, num_samples):
    """Compute samples of the diagonal elements logit noise.

    Args:
      diag_scale: `Tensor` of shape [batch_size, num_classes]. Diagonal
        elements of scale parameters of the distribution to be sampled.
      num_samples: Integer. Number of Monte-Carlo samples to take.

    Returns:
      `Tensor`. Logit noise samples of shape: [batch_size, num_samples,
        1 if num_classes == 2 else num_classes].
    """
    if self.share_samples_across_batch:
      samples_per_batch = 1
    else:
      samples_per_batch = diag_scale.shape[0]

    key = self.make_rng('diag_noise_samples')
    return jnp.expand_dims(diag_scale, 1) * jax.random.normal(
        key, shape=(samples_per_batch, num_samples, 1))

  def _compute_standard_normal_samples(self, factor_loadings, num_samples):
    """Utility function to compute samples from a standard normal distribution.

    Args:
      factor_loadings: `Tensor` of shape
        [batch_size, num_classes * num_factors]. Factor loadings for scale
        parameters of the distribution to be sampled.
      num_samples: Integer. Number of Monte-Carlo samples to take.

    Returns:
      `Tensor`. Samples of shape: [batch_size, num_samples, num_factors].
    """
    if self.share_samples_across_batch:
      samples_per_batch = 1
    else:
      samples_per_batch = factor_loadings.shape[0]

    key = self.make_rng('standard_norm_noise_samples')
    standard_normal_samples = jax.random.normal(
        key, shape=(samples_per_batch, num_samples, self.num_factors))

    if self.share_samples_across_batch:
      standard_normal_samples = jnp.tile(standard_normal_samples,
                                         [factor_loadings.shape[0], 1, 1])

    return standard_normal_samples

  def _compute_noise_samples(self, scale, num_samples):
    """Utility function to compute additive noise samples.

    Args:
      scale: Tuple of tensors of shape (
        [batch_size, num_classes * num_factors],
        [batch_size, num_classes]). Factor loadings and diagonal elements
        for scale parameters of the distribution to be sampled.
      num_samples: Integer. Number of Monte-Carlo samples to take.

    Returns:
      `Tensor`. Logit noise samples of shape: [batch_size, num_samples,
        1 if num_classes == 2 else num_classes].
    """
    factor_loadings, diag_scale = scale

    # Compute the diagonal noise.
    diag_noise_samples = self._compute_diagonal_noise_samples(diag_scale,
                                                              num_samples)

    if self.num_factors > 0:
      # Now compute the factors.
      standard_normal_samples = self._compute_standard_normal_samples(
          factor_loadings, num_samples)

      if self.parameter_efficient:
        res = self._scale_layer_homoscedastic(standard_normal_samples)
        res *= jnp.expand_dims(
            self._scale_layer_heteroscedastic(factor_loadings), 1)
      else:
        # Reshape scale vector into factor loadings matrix.
        factor_loadings = jnp.reshape(factor_loadings,
                                      [-1, self.num_outputs, self.num_factors])

        # Transform standard normal into ~full rank covariance Gaussian samples.
        res = jnp.einsum('ijk,iak->iaj',
                         factor_loadings, standard_normal_samples)
      return res + diag_noise_samples
    return diag_noise_samples

  def get_temperature(self):
    if self.temperature > 0:
      return self.temperature
    else:
      t = jax.nn.sigmoid(self._temperature_pre_sigmoid)
      return (self.temp_upper - self.temp_lower) * t + self.temp_lower

  def _compute_mc_samples(self, locs, scale, num_samples):
    """Utility function to compute Monte-Carlo samples (using softmax).

    Args:
      locs: Tensor of shape [batch_size, total_mc_samples,
        1 if num_classes == 2 else num_classes]. Location parameters of the
        distributions to be sampled.
      scale: Tensor of shape [batch_size, total_mc_samples,
        1 if num_classes == 2 else num_classes]. Scale parameters of the
        distributions to be sampled.
      num_samples: Integer. Number of Monte-Carlo samples to take.

    Returns:
      Tensor of shape [batch_size, num_samples,
        1 if num_classes == 2 else num_classes]. All of the MC samples.
    """
    locs = jnp.expand_dims(locs, axis=1)

    noise_samples = self._compute_noise_samples(scale, num_samples)

    latents = locs + noise_samples
    samples = jax.nn.sigmoid(latents / self.get_temperature())

    return jnp.mean(samples, axis=1)

  @nn.compact
  def __call__(self, inputs, training=True):
    """Computes predictive and log predictive distributions.

    Uses Monte Carlo estimate of softmax approximation to heteroscedastic model
    to compute predictive distribution. O(mc_samples * num_classes).

    Args:
      inputs: Tensor. The input to the heteroscedastic output layer.
      training: Boolean. Whether we are training or not.

    Returns:
      Tensor logits if logits_only = True. Otherwise,
      tuple of (logits, log_probs, probs, predictive_variance). Logits
      represents the argument to a sigmoid function that would yield probs
      (logits = inverse_sigmoid(probs)), so logits can be used with the
      sigmoid cross-entropy loss function.
    """
    locs = self._compute_loc_param(inputs)  # pylint: disable=assignment-from-none
    scale = self._compute_scale_param(inputs)  # pylint: disable=assignment-from-none

    if training:
      total_mc_samples = self.train_mc_samples
    else:
      total_mc_samples = self.test_mc_samples

    probs_mean = self._compute_mc_samples(locs, scale, total_mc_samples)

    probs_mean = jnp.clip(probs_mean, a_min=self.eps)
    log_probs = jnp.log(probs_mean)

    # Inverse sigmoid.
    probs_mean = jnp.clip(probs_mean, a_min=self.eps, a_max=1.0 - self.eps)
    logits = log_probs - jnp.log(1.0 - probs_mean)

    if self.return_locs:
      logits = locs

    if self.logits_only:
      return logits

    return logits, log_probs, probs_mean


# TODO(markcollier): migrate to het library once API is stabilized.
class LatentMCSigmoidDenseFA(nn.Module):
  """Sigmoid and factor analysis approx to heteroscedastic predictions.

  if we assume:
  u ~ N(mu(x), sigma(x))
  and
  y = sigmoid(Wu / temperature)

  we can do a low rank approximation of sigma(x) the full rank matrix as:
  eps_R ~ N(0, I_R), eps_K ~ N(0, I_K)
  u = mu(x) + matmul(V(x), e) + d(x) * e_d
  where V(x) is a matrix of dimension [num_outputs, R=num_factors]
  and d(x) is a vector of dimension [num_outputs, 1]
  num_factors << num_outputs => approx to sampling ~ N(mu(x), sigma(x)).

  The key difference to the non-latent MCSigmoidDenseFA layer is that the latent
  u variable is of the dimension of the pre-logits layer in the network as
  opposed to the number of outputs/classes in the MCSigmoidDenseFA. As a result
  parameter count scales in the pre-logits layer dimensionality rather than the
  number of classes, significantly reducing the parameter count for problems
  with large output spaces.
  """

  num_outputs: int
  latent_dim: int
  num_factors: int  # Set num_factors = 0 for diagonal method.
  temperature: float = 1.0
  temp_lower: float = 0.05
  temp_upper: float = 5.0
  parameter_efficient: bool = False
  train_mc_samples: int = 1000
  test_mc_samples: int = 1000
  share_samples_across_batch: bool = False
  logits_only: bool = False
  return_locs: bool = False
  eps: float = 1e-7

  def setup(self):
    if self.parameter_efficient:
      self._scale_layer_homoscedastic = nn.Dense(
          self.latent_dim, name='scale_layer_homoscedastic',
          kernel_init=nn.initializers.glorot_uniform())
      self._scale_layer_heteroscedastic = nn.Dense(
          self.latent_dim, name='scale_layer_heteroscedastic',
          kernel_init=nn.initializers.glorot_uniform())
    elif self.num_factors > 0:
      self._scale_layer = nn.Dense(
          self.latent_dim * self.num_factors, name='scale_layer',
          kernel_init=nn.initializers.glorot_uniform())

    self._loc_layer = nn.Dense(self.num_outputs, name='loc_layer',
                               kernel_init=nn.initializers.glorot_uniform())
    self._diag_layer = nn.Dense(self.latent_dim, name='diag_layer',
                                kernel_init=nn.initializers.glorot_uniform())

    if self.temperature > 0:
      self._temperature_pre_sigmoid = None
    else:
      self._temperature_pre_sigmoid = self.param('temperature_pre_sigmoid',
                                                 nn.initializers.zeros, (1,))

  def _compute_loc_param(self, inputs):
    """Computes location parameter of the "logits distribution".

    Args:
      inputs: Tensor. The input to the heteroscedastic output layer.

    Returns:
      Tensor of shape [batch_size, num_classes].
    """
    return self._loc_layer(inputs)

  def _compute_scale_param(self, inputs):
    """Computes scale parameter of the "logits distribution".

    Args:
      inputs: Tensor. The input to the heteroscedastic output layer.

    Returns:
      Tuple of tensors of shape
      ([batch_size, num_classes * max(num_factors, 1)],
      [batch_size, num_classes]).
    """
    if self.parameter_efficient or self.num_factors <= 0:
      return (inputs,
              jax.nn.softplus(self._diag_layer(inputs)) + MIN_SCALE_MONTE_CARLO)
    else:
      return (self._scale_layer(inputs),
              jax.nn.softplus(self._diag_layer(inputs)) + MIN_SCALE_MONTE_CARLO)

  def _compute_diagonal_noise_samples(self, diag_scale, num_samples):
    """Compute samples of the diagonal elements logit noise.

    Args:
      diag_scale: `Tensor` of shape [batch_size, num_classes]. Diagonal
        elements of scale parameters of the distribution to be sampled.
      num_samples: Integer. Number of Monte-Carlo samples to take.

    Returns:
      `Tensor`. Logit noise samples of shape: [batch_size, num_samples,
        1 if num_classes == 2 else num_classes].
    """
    if self.share_samples_across_batch:
      samples_per_batch = 1
    else:
      samples_per_batch = diag_scale.shape[0]

    key = self.make_rng('diag_noise_samples')
    return jnp.expand_dims(diag_scale, 1) * jax.random.normal(
        key, shape=(samples_per_batch, num_samples, 1))

  def _compute_standard_normal_samples(self, factor_loadings, num_samples):
    """Utility function to compute samples from a standard normal distribution.

    Args:
      factor_loadings: `Tensor` of shape
        [batch_size, num_classes * num_factors]. Factor loadings for scale
        parameters of the distribution to be sampled.
      num_samples: Integer. Number of Monte-Carlo samples to take.

    Returns:
      `Tensor`. Samples of shape: [batch_size, num_samples, num_factors].
    """
    if self.share_samples_across_batch:
      samples_per_batch = 1
    else:
      samples_per_batch = factor_loadings.shape[0]

    key = self.make_rng('standard_norm_noise_samples')
    standard_normal_samples = jax.random.normal(
        key, shape=(samples_per_batch, num_samples, self.num_factors))

    if self.share_samples_across_batch:
      standard_normal_samples = jnp.tile(standard_normal_samples,
                                         [factor_loadings.shape[0], 1, 1])

    return standard_normal_samples

  def _compute_noise_samples(self, scale, num_samples):
    """Utility function to compute additive noise samples.

    Args:
      scale: Tuple of tensors of shape (
        [batch_size, num_classes * num_factors],
        [batch_size, num_classes]). Factor loadings and diagonal elements
        for scale parameters of the distribution to be sampled.
      num_samples: Integer. Number of Monte-Carlo samples to take.

    Returns:
      `Tensor`. Logit noise samples of shape: [batch_size, num_samples,
        1 if num_classes == 2 else num_classes].
    """
    factor_loadings, diag_scale = scale

    # Compute the diagonal noise.
    diag_noise_samples = self._compute_diagonal_noise_samples(diag_scale,
                                                              num_samples)

    if self.num_factors > 0:
      # Now compute the factors.
      standard_normal_samples = self._compute_standard_normal_samples(
          factor_loadings, num_samples)

      if self.parameter_efficient:
        res = self._scale_layer_homoscedastic(standard_normal_samples)
        res *= jnp.expand_dims(
            self._scale_layer_heteroscedastic(factor_loadings), 1)
      else:
        # Reshape scale vector into factor loadings matrix.
        factor_loadings = jnp.reshape(factor_loadings,
                                      [-1, self.latent_dim, self.num_factors])

        # Transform standard normal into ~full rank covariance Gaussian samples.
        res = jnp.einsum('ijk,iak->iaj',
                         factor_loadings, standard_normal_samples)
      return res + diag_noise_samples
    return diag_noise_samples

  def get_temperature(self):
    if self.temperature > 0:
      return self.temperature
    else:
      t = jax.nn.sigmoid(self._temperature_pre_sigmoid)
      return (self.temp_upper - self.temp_lower) * t + self.temp_lower

  def _compute_mc_samples(self, inputs, scale, num_samples):
    """Utility function to compute Monte-Carlo samples (using softmax).

    Args:
      inputs: Tensor of shape [batch_size, total_mc_samples, latent_dim].
        Location parameters of the distributions to be sampled.
      scale: Tensor of shape [batch_size, total_mc_samples,
        1 if num_classes == 2 else num_classes]. Scale parameters of the
        distributions to be sampled.
      num_samples: Integer. Number of Monte-Carlo samples to take.

    Returns:
      Tensor of shape [batch_size, num_samples,
        1 if num_classes == 2 else num_classes]. All of the MC samples.
    """
    inputs = jnp.expand_dims(inputs, axis=1)

    noise_samples = self._compute_noise_samples(scale, num_samples)

    latents = inputs + noise_samples
    latents = self._compute_loc_param(latents)
    samples = jax.nn.sigmoid(latents / self.get_temperature())

    return jnp.mean(samples, axis=1)

  @nn.compact
  def __call__(self, inputs, training=True):
    """Computes predictive and log predictive distributions.

    Uses Monte Carlo estimate of softmax approximation to heteroscedastic model
    to compute predictive distribution. O(mc_samples * num_classes).

    Args:
      inputs: Tensor. The input to the heteroscedastic output layer.
      training: Boolean. Whether we are training or not.

    Returns:
      Tensor logits if logits_only = True. Otherwise,
      tuple of (logits, log_probs, probs). Logits represents the argument to a
      sigmoid function that would yield probs (logits = inverse_sigmoid(probs)),
      so logits can be used with the sigmoid cross-entropy loss function.
    """
    scale = self._compute_scale_param(inputs)  # pylint: disable=assignment-from-none

    if training:
      total_mc_samples = self.train_mc_samples
    else:
      total_mc_samples = self.test_mc_samples

    probs_mean = self._compute_mc_samples(inputs, scale, total_mc_samples)

    probs_mean = jnp.clip(probs_mean, a_min=self.eps)
    log_probs = jnp.log(probs_mean)

    # Inverse sigmoid.
    probs_mean = jnp.clip(probs_mean, a_min=self.eps, a_max=1.0 - self.eps)
    logits = log_probs - jnp.log(1.0 - probs_mean)

    if self.logits_only:
      return logits

    return logits, log_probs, probs_mean


def weight_standardize(w: jnp.ndarray,
                       axis: Union[Sequence[int], int],
                       eps: float):
  """Standardize (mean=0, var=1) a weight."""
  w = w - jnp.mean(w, axis=axis, keepdims=True)
  w = w / jnp.sqrt(jnp.mean(jnp.square(w), axis=axis, keepdims=True) + eps)
  return w


class IdentityLayer(nn.Module):
  """Identity layer, convenient for giving a name to an array."""

  @nn.compact
  def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
    return x


class StdConv(nn.Conv):

  def param(self, name, *a, **kw):
    param = super().param(name, *a, **kw)
    if name == 'kernel':
      param = weight_standardize(param, axis=[0, 1, 2], eps=1e-5)
    return param


class ResidualUnit(nn.Module):
  """Bottleneck ResNet block.

  Attributes:
    nout: Number of output features.
    strides: Downsampling stride.
    dilation: Kernel dilation.
    bottleneck: If True, the block is a bottleneck block.
    gn_num_groups: Number of groups in GroupNorm layer.
  """
  nout: int
  strides: Tuple[int, ...] = (1, 1)
  dilation: Tuple[int, ...] = (1, 1)
  bottleneck: bool = True
  gn_num_groups: int = 32

  @nn.compact
  def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
    features = self.nout
    nout = self.nout * 4 if self.bottleneck else self.nout
    needs_projection = x.shape[-1] != nout or self.strides != (1, 1)
    residual = x
    if needs_projection:
      residual = StdConv(nout,
                         (1, 1),
                         self.strides,
                         use_bias=False,
                         name='conv_proj')(residual)
      residual = nn.GroupNorm(num_groups=self.gn_num_groups, epsilon=1e-4,
                              name='gn_proj')(residual)

    if self.bottleneck:
      x = StdConv(features, (1, 1), use_bias=False, name='conv1')(x)
      x = nn.GroupNorm(num_groups=self.gn_num_groups, epsilon=1e-4,
                       name='gn1')(x)
      x = nn.relu(x)

    x = StdConv(features, (3, 3), self.strides, kernel_dilation=self.dilation,
                use_bias=False, name='conv2')(x)
    x = nn.GroupNorm(num_groups=self.gn_num_groups, epsilon=1e-4, name='gn2')(x)
    x = nn.relu(x)

    last_kernel = (1, 1) if self.bottleneck else (3, 3)
    x = StdConv(nout, last_kernel, use_bias=False, name='conv3')(x)
    x = nn.GroupNorm(num_groups=self.gn_num_groups,
                     epsilon=1e-4,
                     name='gn3',
                     scale_init=nn.initializers.zeros)(x)
    x = nn.relu(residual + x)

    return x


class ResNetStage(nn.Module):
  """ResNet Stage: one or more stacked ResNet blocks.

  Attributes:
    block_size: Number of ResNet blocks to stack.
    nout: Number of features.
    first_stride: Downsampling stride.
    first_dilation: Kernel dilation.
    bottleneck: If True, the bottleneck block is used.
    gn_num_groups: Number of groups in group norm layer.
  """

  block_size: int
  nout: int
  first_stride: Tuple[int, ...]
  first_dilation: Tuple[int, ...] = (1, 1)
  bottleneck: bool = True
  gn_num_groups: int = 32

  @nn.compact
  def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
    x = ResidualUnit(self.nout,
                     strides=self.first_stride,
                     dilation=self.first_dilation,
                     bottleneck=self.bottleneck,
                     gn_num_groups=self.gn_num_groups,
                     name='unit1')(x)
    for i in range(1, self.block_size):
      x = ResidualUnit(self.nout,
                       strides=(1, 1),
                       bottleneck=self.bottleneck,
                       gn_num_groups=self.gn_num_groups,
                       name=f'unit{i + 1}')(x)
    return x


class BitResNetHeteroscedastic(nn.Module):
  """Bit ResNetV1 Heteroscedastic.

  Attributes:
    num_outputs: Num output classes. If None, a dict of intermediate feature
      maps is returned
    gn_num_groups: Number groups in the group norm layer..
    width_factor: Width multiplier for each of the ResNet stages.
    num_layers: Number of layers (see `_BLOCK_SIZE_OPTIONS` for stage
      configurations).
    max_output_stride: Defines the maximum output stride of the resnet.
      Typically, resnets output feature maps have sride 32. We can, however,
      lower that number by swapping strides with dilation in later stages. This
      is common in cases where stride 32 is too large, e.g., in dense prediciton
      tasks.
  """

  num_outputs: Optional[int] = 1000
  gn_num_groups: int = 32
  width_factor: int = 1
  num_layers: int = 50
  max_output_stride: int = 32
  # heteroscedastic args
  multiclass: bool = False
  temperature: float = 1.0
  temp_lower: float = 0.05
  temp_upper: float = 5.0
  mc_samples: int = 1000
  num_factors: int = 0
  param_efficient: bool = True
  return_locs: bool = False
  latent_het: bool = False
  fix_base_model: bool = False

  @nn.compact
  def __call__(self,
               x: jnp.ndarray,
               train: bool = True,
               debug: bool = False
               ) -> Union[jnp.ndarray, Dict[str, jnp.ndarray]]:
    """Applies the Bit ResNet model to the inputs.

    Args:
      x: Inputs to the model.
      train: Unused.
      debug: Unused.

    Returns:
       Un-normalized logits if `num_outputs` is provided, a dictionary with
       representations otherwise.
    """
    del train
    del debug
    if self.max_output_stride not in [4, 8, 16, 32]:
      raise ValueError('Only supports output strides of [4, 8, 16, 32]')

    blocks, bottleneck = _BLOCK_SIZE_OPTIONS[self.num_layers]

    width = int(64 * self.width_factor)

    # Root block.
    x = StdConv(width, (7, 7), (2, 2), use_bias=False, name='conv_root')(x)
    x = nn.GroupNorm(num_groups=self.gn_num_groups, epsilon=1e-4,
                     name='gn_root')(x)
    x = nn.relu(x)
    x = nn.max_pool(x, (3, 3), strides=(2, 2), padding='SAME')
    representations = {'stem': x}

    # Stages.
    x = ResNetStage(
        blocks[0],
        width,
        first_stride=(1, 1),
        bottleneck=bottleneck,
        gn_num_groups=self.gn_num_groups,
        name='block1')(x)
    stride = 4
    for i, block_size in enumerate(blocks[1:], 1):
      max_stride_reached = self.max_output_stride <= stride
      x = ResNetStage(
          block_size,
          width * 2**i,
          first_stride=(2, 2) if not max_stride_reached else (1, 1),
          first_dilation=(2, 2) if max_stride_reached else (1, 1),
          bottleneck=bottleneck,
          gn_num_groups=self.gn_num_groups,
          name=f'block{i + 1}')(x)
      if not max_stride_reached:
        stride *= 2
      representations[f'stage_{i + 1}'] = x

    # Head.
    x = jnp.mean(x, axis=(1, 2))
    x = IdentityLayer(name='pre_logits')(x)
    representations['pre_logits'] = x

    if self.multiclass:
      # TODO(markcollier): Add support for multiclass case
      pass
    else:
      if self.latent_het:
        output_layer = LatentMCSigmoidDenseFA(
            self.num_outputs, int(2048 * self.width_factor), self.num_factors,
            self.temperature, self.temp_lower, self.temp_upper,
            self.param_efficient, self.mc_samples, self.mc_samples,
            logits_only=True, return_locs=self.return_locs,
            name='multilabel_head')
      else:
        output_layer = MCSigmoidDenseFA(
            self.num_outputs, self.num_factors, self.temperature,
            self.temp_lower, self.temp_upper, self.param_efficient,
            self.mc_samples, self.mc_samples, logits_only=True,
            return_locs=self.return_locs, name='multilabel_head')

    # TODO(markcollier): Fix base model without using stop_gradient.
    if self.fix_base_model:
      x = jax.lax.stop_gradient(x)

    x = output_layer(x)

    representations['temperature'] = output_layer.get_temperature()
    representations['logits'] = x
    return x, representations


# A dictionary mapping the number of layers in a resnet to the number of
# blocks in each stage of the model. The second argument indicates whether we
# use bottleneck layers or not.
_BLOCK_SIZE_OPTIONS = {
    5: ([1], True),  # Only strided blocks. Total stride 4.
    8: ([1, 1], True),  # Only strided blocks. Total stride 8.
    11: ([1, 1, 1], True),  # Only strided blocks. Total stride 16.
    14: ([1, 1, 1, 1], True),  # Only strided blocks. Total stride 32.
    9: ([1, 1, 1, 1], False),  # Only strided blocks. Total stride 32.
    18: ([2, 2, 2, 2], False),
    26: ([2, 2, 2, 2], True),
    34: ([3, 4, 6, 3], False),
    50: ([3, 4, 6, 3], True),
    101: ([3, 4, 23, 3], True),
    152: ([3, 8, 36, 3], True),
    200: ([3, 24, 36, 3], True)
}


def bit_resnet_heteroscedastic(  # pylint: disable=missing-function-docstring
    num_classes: int,
    num_layers: int,
    width_factor: int,
    gn_num_groups: int = 32,
    multiclass: bool = False,
    temperature: float = 1.0,
    temperature_lower_bound: float = 0.05,
    temperature_upper_bound: float = 5.0,
    mc_samples: int = 1000,
    num_factors: int = 0,
    param_efficient: bool = True,
    return_locs: bool = False,
    latent_het: bool = False,
    fix_base_model: bool = False) -> nn.Module:
  return BitResNetHeteroscedastic(
      num_outputs=num_classes,
      gn_num_groups=gn_num_groups,
      width_factor=width_factor,
      num_layers=num_layers,
      multiclass=multiclass,
      temperature=temperature,
      temp_lower=temperature_lower_bound,
      temp_upper=temperature_upper_bound,
      mc_samples=mc_samples,
      num_factors=num_factors,
      param_efficient=param_efficient,
      return_locs=return_locs,
      latent_het=latent_het,
      fix_base_model=fix_base_model)

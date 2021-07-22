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

"""Utils for Radial BNNs."""
import math
from absl import logging

from edward2.tensorflow import constraints
from edward2.tensorflow import generated_random_variables
from edward2.tensorflow import regularizers
from edward2.tensorflow.initializers import get
from edward2.tensorflow.initializers import serialize
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import samplers
from tensorflow_probability.python.internal import tensor_util
from tensorflow_probability.python.internal import tensorshape_util

__all__ = ['Radial', 'TrainableHeRadial']


class Radial(tfp.distributions.Distribution):
  r"""Creates a Radial distribution as described in Radial Bayesian Neural Networks (Farquhar et al., AISTATS 2020).

  Parameterized by a location tensor :math:`\mu` and scaling tensor
  :math:`\rho' such that :math:`\sigma = \mathrm{softplus}(\rho)`.

  The random variable is defined:
    :math:`\mathbf{w} = \mu + \mathrm{softplus}(\rho) \cdot \epsilon`,
  where epsilon is distributed uniformly in angle and normally in radius
  from the origin:
    :math:`\epsilon = \frac{\mathbf{z}}{||\mathbf{z}||_2} \times z`,
  where
    :math:`\mathbf{z} \sim \mathcal{N}(\mathbf{0}, \mathbf{1})`
  with the same shape as :math:`\rho`.
  """

  def __init__(self,
               loc,
               presoftplus_scale,
               batch_shape=tuple(),
               dtype=tf.float32,
               validate_args=False,
               allow_nan_stats=True,
               name='Radial'):
    r"""Constructor.

    Args:
      loc: `Tensor` representing the mean of the distribution.
      presoftplus_scale: `Tensor` representing the pre-softplus scale, `\rho`.
      batch_shape: Positive `int`-like vector-shaped `Tensor` representing
        the new shape of the batch dimensions. Default value: [].
      dtype: the data type of the distribution.
      validate_args: Python `bool`, default `False`. When `True`, distribution
      parameters are checked for validity despite possibly degrading runtime
      performance. When `False` invalid inputs may silently render incorrect
      outputs.
      allow_nan_stats: Python `bool`, default `True`. When `True`, statistics
      (e.g., mean, mode, variance) use the value "`NaN`" to indicate the result
      is undefined. When `False`, an exception is raised if one or more of the
      statistic's batch members are undefined.
      name: Python `str` name prefixed to Ops created by this class.

    Raises:
      ValueError: For known-bad arguments, i.e. unsupported event
      dimension.
    """
    parameters = dict(locals())
    with tf.name_scope(name) as name:
      shape_dtype = dtype_util.common_dtype([batch_shape], dtype_hint=tf.int32)
      self._loc = loc
      self._presoftplus_scale = presoftplus_scale
      self._batch_shape_parameter = tensor_util.convert_nonref_to_tensor(
          batch_shape, dtype=shape_dtype, name='batch_shape')
      self._batch_shape_static = (
          tensorshape_util.constant_value_as_shape(self._batch_shape_parameter))

      super().__init__(
          dtype=dtype,
          validate_args=validate_args,
          allow_nan_stats=allow_nan_stats,
          reparameterization_type=(tfp.distributions.FULLY_REPARAMETERIZED),
          parameters=parameters,
          name=name)

  @classmethod
  def _params_event_ndims(cls):
    return dict()

  @property
  def loc(self):
    """The `loc` `Tensor` in `Y = scale @ X + loc`."""
    return self._loc

  @property
  def scale(self):
    """The `scale` `Tensor` in `Y = scale @ X + loc`."""
    return tf.math.softplus(self._presoftplus_scale)

  def _batch_shape_tensor(self):
    return self._batch_shape_parameter

  def _batch_shape(self):
    return self._batch_shape_static

  def _event_shape_tensor(self):
    return tf.constant(self._loc.shape, dtype=tf.int32)

  def _event_shape(self):
    return self._loc.shape

  def log_prob(self, x):
    # pylint: disable=line-too-long
    r"""Returns the log of the probability density/mass function evaluated at `x`.

    From equations 19 and 21 in Farquhar et al. 2020 for an input in
      Cartesian space w^x:

    :math:`q(w^x) = \frac{q(\epsilon^r)}{\prod_k
    \sigma_k^x|\frac{\partial\epsilon_i^x}{\partial\epsilon_j^r}|^{-1}`

    and we have:
    :math:`|\frac{\partial\epsilon_i^x}{\partial\epsilon_j^r}| =
    (\epsilon_0^r)^{D-1}\prod_{i=2}^{D}(\sin(\epsilon_i^r))^{i-1}`

    and equation 26:
    :math:`q(\epsilon^r)=\frac{1}{\sqrt{2\pi}}e^{-\frac{\epsilon_0^2}{2}}\prod_{i=1}^{D-1}\sin(\epsilon_i^r)^{D-i}`

    Args:
        x (Tensor): must be of shape: sample_shape + batch_shape + event_shape
    """
    # pylint: enable=line-too-long

    # For simplicity center on the noise
    epsilon_cartesian = (x - self._loc) / self.scale

    # We view the event shape dimensions as a vector
    epsilon_cartesian_shape = epsilon_cartesian.shape
    epsilon_cartesian_nonevent_shape = epsilon_cartesian_shape[:-len(
        self.event_shape)]
    epsilon_cartesian = tf.reshape(
        epsilon_cartesian,
        tuple(epsilon_cartesian_nonevent_shape) + (-1,))
    epsilon_radial = self._cartesian_to_hyperspherical(epsilon_cartesian)

    # Jacobian
    dimension = epsilon_radial.shape[-1]
    jacobian = epsilon_radial[:, 0]**(dimension - 1) * tf.math.prod(
        tf.math.pow(
            tf.math.sin(epsilon_radial[..., 1:]), tf.math.arange(1, dimension)),
        axis=-1)

    # q(epsilon^r)
    noise_density = ((1 / tf.math.sqrt(2 * tf.constant(math.pi))) *
                     math.exp(-0.5 * epsilon_radial[..., 0]**2) * tf.math.prod(
                         tf.math.pow(
                             tf.math.sin(epsilon_radial[..., 1:]),
                             dimension - tf.arange(1, dimension)),
                         axis=-1))

    sigma_prod = tf.math.prod(self.scale)
    log_probability = noise_density / jacobian / sigma_prod
    return log_probability.reshape(epsilon_cartesian_shape)

  def _cartesian_to_hyperspherical(self, x_c):
    """Helper for log_prob.

    As a convention here, the first element is the radius and the rest are
    angles.

    Assumes only one dimension representing a vector in cartesian space.

    Args:
      x_c: `Tensor` input in cartesian coordinates (weight space). Has shape
        sample_shape + batch_shape + flattened_event_shape.

    Returns:
      x_r: same dimensions converted to hyperspherical coordinates.
    """
    # x_i = Sum of x_j ** 2 for j = N - i
    base = tf.reverse(x_c, [-1])
    base = tf.math.cumsum(base**2, -1)

    # Now we flip back which gives us
    # [x_n**2 + x_{n-1}**2 + ... + x_1**2, ..., x_n**2]
    base = tf.reverse(base, [-1])
    base = tf.math.sqrt(base)
    theta = tf.math.atan(base[..., 1:] / x_c[..., 0:-1]) + (
        tf.constant(math.pi) / 2)  # The first element is x_0 / base_1

    # Correcting the last element
    theta[..., -1] = tf.math.atan(
        x_c[..., -1] /
        (x_c[..., -2] + tf.math.sqrt((x_c[..., -1]**2) +
                                     (x_c[..., -2]**2)))) + (
                                         tf.constant(math.pi) / 2)

    # Now we add the radius to the start,
    # which is just the first term of the base vector that we haven't used
    x_r = tf.cat((tf.expand_dims(base[..., 0], -1), theta), 1)

    return x_r

  def _sample_n(self, n, seed=None):
    raw = samplers.normal(
        shape=tf.concat([[n], self.batch_shape, self.event_shape], axis=0),
        seed=seed,
        dtype=self.dtype)
    direction = raw / tf.norm(raw, ord=2, axis=-1)[..., tf.newaxis]
    distance = samplers.normal(
        shape=tf.concat([[n], self.batch_shape, [1] * len(self.event_shape)],
                        axis=0),
        seed=seed,
        dtype=self.dtype)
    return self._loc + self.scale * direction * distance

  def entropy(self):
    logging.warning(
        'Entropy is correct only up to a constant, for optimization only.')
    return tf.math.log(tf.math.reduce_sum(self.scale))

  def mean(self):
    shape = tensorshape_util.concatenate(self.batch_shape, self.event_shape)
    has_static_shape = tensorshape_util.is_fully_defined(shape)
    if not has_static_shape:
      shape = tf.concat([
          self.batch_shape_tensor(),
          self.event_shape_tensor(),
      ], 0)

    if self.loc is None:
      return tf.zeros(shape, self.dtype)

    if has_static_shape and shape == self.loc.shape:
      return tf.identity(self.loc)

    # Add dummy tensor of zeros to broadcast. This is only necessary if
    # shape != self.loc.shape,
    # but we could not determine if this is the case.
    return tf.identity(self.loc) + tf.zeros(shape, self.dtype)

  def kl_divergence(self, other, name='kl_divergence'):
    if (isinstance(other, tfp.distributions.independent.Independent) and
        isinstance(other.distribution, tfp.distributions.Normal)):
      return kl_radial_normal(p=self, q=other)
    elif isinstance(other, Radial):
      raise NotImplementedError(
          'Have not yet added Radial-Radial KL divergence.')
    else:
      raise NotImplementedError(f'Other distribution has type {type(other)}.')


@tfp.distributions.RegisterKL(
    Radial, tfp.distributions.MultivariateNormalLinearOperator)
def kl_radial_normal(p, q, n_samples=10, name=None):
  """Computes the KL between a Radial and MVN."""
  with tf.name_scope(name or 'kl_radial_normal'):
    if p.event_shape != q.event_shape:
      raise ValueError(
          'KL-divergence between Radial and Multivariate Normals with'
          'different event shapes cannot be computed.')

    logging.warning(
        'KL is correct only up to a constant, for optimization only.')

    # KL = Cross-Entropy - Entropy
    # We find cross-entropy by MC estimation
    cross_entropy = -tf.math.reduce_sum(q.log_prob(p.sample(
        (n_samples,)))) / n_samples
    return cross_entropy - p.entropy()


class TrainableRadial(tf.keras.layers.Layer):
  """Random Radial op as an initializer with trainable mean and stddev."""

  def __init__(
      self,
      mean_initializer=tf.keras.initializers.TruncatedNormal(stddev=1e-5),
      stddev_initializer=tf.keras.initializers.TruncatedNormal(
          mean=-3., stddev=0.1),
      mean_regularizer=None,
      stddev_regularizer=None,
      mean_constraint=None,
      stddev_constraint='softplus',
      seed=None,
      **kwargs):
    """Constructs the initializer."""
    super().__init__(**kwargs)
    self.mean_initializer = get(mean_initializer)
    self.stddev_initializer = get(stddev_initializer)
    self.mean_regularizer = regularizers.get(mean_regularizer)
    self.stddev_regularizer = regularizers.get(stddev_regularizer)
    self.mean_constraint = constraints.get(mean_constraint)
    self.stddev_constraint = constraints.get(stddev_constraint)
    self.seed = seed

  def build(self, shape, dtype=None):
    if dtype is None:
      dtype = self.dtype

    self.mean = self.add_weight(
        'mean',
        shape=shape,
        initializer=self.mean_initializer,
        regularizer=self.mean_regularizer,
        constraint=None,
        dtype=dtype,
        trainable=True)
    self.stddev = self.add_weight(
        'stddev',
        shape=shape,
        initializer=self.stddev_initializer,
        regularizer=self.stddev_regularizer,
        constraint=None,
        dtype=dtype,
        trainable=True)
    self.built = True

  def __call__(self, shape, dtype=None):
    if not self.built:
      self.build(shape, dtype)
    mean = self.mean
    if self.mean_constraint:
      mean = self.mean_constraint(mean)
    stddev = self.stddev
    if self.stddev_constraint:
      stddev = self.stddev_constraint(stddev)
    return generated_random_variables.make_random_variable(Radial)(
        loc=mean, presoftplus_scale=tf.math.log(tf.math.expm1(stddev)))

  def get_config(self):
    return {
        'mean_initializer': serialize(self.mean_initializer),
        'stddev_initializer': serialize(self.stddev_initializer),
        'mean_regularizer': regularizers.serialize(self.mean_regularizer),
        'stddev_regularizer': regularizers.serialize(self.stddev_regularizer),
        'mean_constraint': constraints.serialize(self.mean_constraint),
        'stddev_constraint': constraints.serialize(self.stddev_constraint),
        'seed': self.seed,
    }


class TrainableHeRadial(TrainableRadial):
  """Trainable radial initialized per He et al.

  2015, given a ReLU nonlinearity.

  The distribution is initialized to a Radial scaled by `sqrt(2 / fan_in)`,
  where `fan_in` is the number of input units. A ReLU nonlinearity is assumed
  for this initialization scheme.

  References:
    He K, Zhang X, Ren S, Sun J. Delving deep into rectifiers: Surpassing
    human-level performance on imagenet classification. In Proceedings of the
    IEEE international conference on computer vision 2015 (pp. 1026-1034).
    https://arxiv.org/abs/1502.01852
  """

  def __init__(self, seed=None, **kwargs):
    super().__init__(
        mean_initializer=tf.keras.initializers.he_normal(seed),
        seed=seed,
        **kwargs)

  def get_config(self):
    return {
        'seed': self.seed,
    }

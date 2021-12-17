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

"""Run a small BIMO MLP agent.
"""
# Disabling invalid-name because it doesn't like the variable name W_init,
# however that is a descriptive name in this code.
# pylint: disable=invalid-name

import functools
import math
from typing import Any, Callable, Sequence, Tuple, Union

from absl import app
from absl import flags
import chex
from enn import base as enn_base
import jax
from jax import jit
from jax import value_and_grad
from jax import vmap
import jax.example_libraries.stax as stax
import jax.numpy as jnp
import jax.tree_util
from neural_testbed import base as testbed_base
from neural_testbed import leaderboard
from neural_testbed.experiments import experiment
import optax
import tensorflow.data as tf_data
import tensorflow_datasets as tfds
from tensorflow_probability.substrates import jax as tfp

tfd = tfp.distributions


flags.DEFINE_integer(
    'input_dim', 1,
    'Input dimension.')
flags.DEFINE_float(
    'temperature', 1,
    'Testbed softmax temperature.')
flags.DEFINE_integer(
    'num_classes', 2,
    'Number of classes.')
flags.DEFINE_float(
    'data_ratio', 1,
    'Ratio of training points.')
flags.DEFINE_integer(
    'tau', 1,
    'Test distribution order.')
flags.DEFINE_integer(
    'seed', 1,
    'Random seed')

#  Model flags
flags.DEFINE_enum(
    'method', 'wsvgd', ['wsvgd', 'fsvgd'],
    'SVGD method to use.')
flags.DEFINE_float(
    'base_lr', 1e-3,
    'Learning rate')
flags.DEFINE_float(
    'svgd_temp', 1.,
    'Temperature for SVGD update.')
flags.DEFINE_float(
    'kernel_bandwidth', 1.,
    'Bandwidth for RBF kernel.')
flags.DEFINE_float(
    'fn_kernel_bandwidth', 1.,
    'Bandwidth for functional RBF kernel.')
flags.DEFINE_float(
    'prior_var', 1.,
    'Variance of the Gaussian prior on the weights.')
flags.DEFINE_boolean(
    'use_prior', True,
    'Whether to use a prior on the weights.')
flags.DEFINE_integer(
    'batch_size', 32,
    'Batch size')
flags.DEFINE_integer(
    'num_heads', 4,
    'Number of input and output heads.')
flags.DEFINE_list(
    'hidden_sizes', '32,32',
    'Hidden layer sizes in the trunk of the network.')
flags.DEFINE_boolean(
    'adaptive_prior_var', True,
    'Whether to adapt the prior variance based on temperature and data ratio.')
flags.DEFINE_integer(
    'num_train_steps', 20000,
    'Number of steps to train for.')
flags.DEFINE_integer(
    'summarize_every', 1000,
    'Number of steps between summaries.')

FLAGS = flags.FLAGS

# Types
PyTree = Any
BIMOParams = Tuple[PyTree, PyTree, PyTree]
PRNGKey = jnp.ndarray
Val = Union[float, jnp.ndarray]
LossFn = Callable[[PyTree, PRNGKey, int, jnp.ndarray, jnp.ndarray], float]
LogProbFn = Callable[[BIMOParams], float]
DataLogProbFn = Callable[[BIMOParams, Tuple[jnp.ndarray, jnp.ndarray]], float]
KernelFn = Callable[[PyTree, PyTree], float]
KernelFnWithData = Callable[[PyTree, PyTree, jnp.ndarray], float]
Batch = Tuple[jnp.ndarray, jnp.ndarray]
ModelApplyFn = Callable[[PyTree, jnp.ndarray], jnp.ndarray]


def lr_sched_fn(
    cur_step: int,
    base_lr: float = 1e-4,
    decay_rate: float = 0.2,
    steps=None) -> Val:
  """Creates a learning rate schedule function.

  This is meant to be used with functools.partial. After supplying the
  keyword arguments, a call to the function with only the cur_step arg
  will return the learning rate for the current step.

  Args:
    cur_step: The current step to compute the learning rate for.
    base_lr: The starting learning rate which will be decayed over time.
    decay_rate: The multiplicative rate to decay the learning rate by.
    steps: The steps on which to decay the learning rate.
  Returns:
    The learning rate for cur_step.
  """
  if steps is None:
    steps = []
  steps = jnp.array([0] + steps)
  lrs = jnp.array(
      [base_lr * math.pow(decay_rate, i) for i in range(len(steps))])
  ind = jnp.sum(steps < cur_step)
  return lrs[ind]


def lr_perc_sched_fn(
    cur_step: int,
    max_num_steps: int = 10000,
    base_lr: float = 1e-4,
    decay_rate: float = 0.2,
    perc_steps=None) -> Val:
  """Creates a learning rate schedule function.

  This is meant to be used with functools.partial. After supplying the
  keyword arguments, a call to the function with only the cur_step arg
  will return the learning rate for the current step.

  Args:
    cur_step: The current step to compute the learning rate for.
    max_num_steps: The total number of steps in the schedule.
    base_lr: The starting learning rate which will be decayed over time.
    decay_rate: The multiplicative rate to decay the learning rate by.
    perc_steps: The percents of max_num_steps when the learning rate should
      be decayed.
  Returns:
    The learning rate for cur_step.
  """
  if perc_steps is None:
    perc_steps = []
  steps = [int(max_num_steps*p) for p in perc_steps]
  return lr_sched_fn(
      cur_step, base_lr=base_lr, decay_rate=decay_rate, steps=steps)


def vectorize_pytree(tree: PyTree) -> jnp.ndarray:
  """Flattens a PyTree to a single array."""
  values, _ = jax.tree_util.tree_flatten(tree)
  flat_values = [jnp.reshape(x, [-1]) for x in values]
  concat_values = jnp.concatenate(flat_values)
  return concat_values


def treemap_kernel(kernel_fn: KernelFn) -> KernelFn:
  """Maps a kernel function over all elements of a PyTree."""

  def new_kernel(x_tree, y_tree):
    flat_x = vectorize_pytree(x_tree)
    flat_y = vectorize_pytree(y_tree)
    return kernel_fn(flat_x, flat_y)

  return new_kernel


def rbf_kernel(
    x: jnp.ndarray,
    y: jnp.ndarray,
    kernel_bandwidth: float = 1.) -> Val:
  """Computes the radial basis function kernel: exp(- 0.5 * (x - y)^2)."""
  return jnp.exp(-(1. / (2. * kernel_bandwidth)) * jnp.sum(jnp.square(x - y)))


def functional_kernel(
    params_a: BIMOParams,
    params_b: BIMOParams,
    xs: jnp.ndarray,
    model_apply: Union[ModelApplyFn, None] = None,
    inner_kernel: Union[KernelFn, None] = None):
  batch_model_apply = vmap(model_apply, in_axes=(None, 0))
  a_vals = batch_model_apply(params_a, xs)
  b_vals = batch_model_apply(params_b, xs)
  return inner_kernel(a_vals, b_vals)


def svgd_grad(
    log_p_fn: LogProbFn,
    kernel_fn: KernelFn,
    num_particles: int,
    particle_params: PyTree,
    temp: float = 1.) -> Tuple[Tuple[Val, jnp.ndarray], PyTree]:
  """Computes the Stein Variational Gradient Descent gradient.

  Args:
    log_p_fn: A function which accepts parameters and returns the log
      probability under the model.
    kernel_fn: A function which computes kernel values for pairs of parameters.
    num_particles: The number of particles being evolved.
    particle_params: The current parameter settings of all the particles.
    temp: The 'temperature' of the SVGD update.
  Returns:
    A PyTree of the same structure and shapes as particle_params containing
    the SVGD update.
  """
  # Compute the gradients of the target distribution.
  # [num_particles, param_dim]
  log_ps, log_p_grads = vmap(value_and_grad(log_p_fn))(particle_params)

  # Compute the gradients of the kernel.
  kernel_val_and_grad_fn = value_and_grad(kernel_fn, argnums=0)
  kernel_pwise_val_and_grad_fn = vmap(
      vmap(kernel_val_and_grad_fn, in_axes=(None, 0)),
      in_axes=(0, None))
  # [num_particles, num_particles], [num_particles, num_particles, param_dim]
  k_vals, k_grads = kernel_pwise_val_and_grad_fn(
      particle_params, particle_params)

  def sum_fn(log_p_grads: jnp.ndarray, k_grads: jnp.ndarray) -> jnp.ndarray:
    weighted_grads = jnp.einsum('ij,j...->i...', k_vals, log_p_grads)
    avg_grads = weighted_grads * (1 / (temp * num_particles))
    sum_k_grads = jnp.mean(k_grads, axis=0)
    return avg_grads + sum_k_grads

  grads = jax.tree_util.tree_multimap(sum_fn, log_p_grads, k_grads)
  loss_val = jnp.mean(log_ps)
  return (loss_val, k_vals), grads


def bimo_p_svgd_grad(
    log_p_fn: LogProbFn,
    kernel_fn: KernelFn,
    num_particles: int,
    params: BIMOParams,
    temp: float = 1.) -> Tuple[PyTree, BIMOParams]:
  """Computes the BIMO projected SVGD update.

  This is a thin wrapper around svgd_grad that handles BIMO's trunk parameters.
  Because there is only one set of trunk parameters, it is tiled before calling
  svgd_grad, and the returned trunk updates are averaged (aka 'projected').

  Args:
    log_p_fn: A function which accepts parameters and returns the log
      probability under the model.
    kernel_fn: A function which computes kernel values for pairs of parameters.
    num_particles: The number of particles being evolved.
    params: The current parameters of the BIMO model.
    temp: The 'temperature' of the SVGD update.
  Returns:
    A PyTree of the same structure and shapes as params containing the
    project SVGD update.
  """

  def tile_fn(leaf: jnp.ndarray) -> jnp.ndarray:
    tile_reps = [num_particles] + [1] * len(leaf.shape)
    return jnp.tile(leaf[jnp.newaxis, ...], tile_reps)

  # Replicate the trunk params for each ensemble member.
  tiled_betas = jax.tree_util.tree_map(tile_fn, params[1])
  params = (params[0], tiled_betas, params[2])

  # Compute SVGD grads.
  vals, grads = svgd_grad(
      log_p_fn, kernel_fn, num_particles, params, temp=temp)
  alpha_grads, beta_grads, gamma_grads = grads

  # Project the SVGD grads back onto the constraint set by taking the mean.
  beta_grads = jax.tree_util.tree_map(lambda l: jnp.mean(l, axis=0), beta_grads)
  return vals, (alpha_grads, beta_grads, gamma_grads)


def NEW_svgd_grad(
    log_p_fn: LogProbFn,
    kernel_fn: KernelFn,
    num_particles: int,
    particle_params: PyTree,
    avg_kernel_fn: Union[KernelFn, None] = None,
    temp: float = 1.) -> Tuple[Tuple[Val, jnp.ndarray], PyTree]:
  """Computes the Stein Variational Gradient Descent gradient.

  Args:
    log_p_fn: A function which accepts parameters and returns the log
      probability under the model.
    kernel_fn: A function which computes kernel values for pairs of parameters.
    num_particles: The number of particles being evolved.
    particle_params: The current parameter settings of all the particles.
    avg_kernel_fn: Kernel used to compute weighted gradient averages.
    temp: The 'temperature' of the SVGD update.
  Returns:
    A PyTree of the same structure and shapes as particle_params containing
    the SVGD update.
  """
  # Compute the gradients of the target distribution.
  # [num_particles, param_dim]
  log_ps, log_p_grads = vmap(value_and_grad(log_p_fn))(particle_params)

  # Compute the values and gradients of the repulsor kernel.
  kernel_val_and_grad_fn = value_and_grad(kernel_fn, argnums=0)
  kernel_pwise_val_and_grad_fn = vmap(
      vmap(kernel_val_and_grad_fn, in_axes=(None, 0)),
      in_axes=(0, None))
  # [num_particles, num_particles], [num_particles, num_particles, param_dim]
  kernel_vals, kernel_grads = kernel_pwise_val_and_grad_fn(
      particle_params, particle_params)

  if avg_kernel_fn is not None:
    # Compute the gradient averaging kernel
    pwise_avg_kernel_fn = vmap(vmap(
        avg_kernel_fn, in_axes=(None, 0)), in_axes=(0, None))
    kernel_vals = pwise_avg_kernel_fn(particle_params, particle_params)

  def sum_fn(log_p_grads: jnp.ndarray, k_grads: jnp.ndarray) -> jnp.ndarray:
    weighted_grads = jnp.einsum('ij,j...->i...', kernel_vals, log_p_grads)
    avg_grads = weighted_grads * (1 / (temp * num_particles))
    sum_k_grads = jnp.mean(k_grads, axis=0)
    return avg_grads + sum_k_grads

  grads = jax.tree_util.tree_multimap(sum_fn, log_p_grads, kernel_grads)
  loss_val = jnp.mean(log_ps)
  return (loss_val, kernel_vals), grads


def NEW_bimo_p_svgd_grad(
    log_p_fn: LogProbFn,
    kernel_fn: KernelFn,
    num_particles: int,
    params: BIMOParams,
    avg_kernel_fn: Union[KernelFn, None],
    temp: float = 1.) -> Tuple[PyTree, BIMOParams]:
  """Computes the BIMO projected SVGD update.

  This is a thin wrapper around svgd_grad that handles BIMO's trunk parameters.
  Because there is only one set of trunk parameters, it is tiled before calling
  svgd_grad, and the returned trunk updates are averaged (aka 'projected').

  Args:
    log_p_fn: A function which accepts parameters and returns the log
      probability under the model.
    kernel_fn: A function which computes kernel values for pairs of parameters.
    num_particles: The number of particles being evolved.
    params: The current parameters of the BIMO model.
    avg_kernel_fn: Kernel used to compute weighted gradient averages.
    temp: The 'temperature' of the SVGD update.
  Returns:
    A PyTree of the same structure and shapes as params containing the
    project SVGD update.
  """

  def tile_fn(leaf: jnp.ndarray) -> jnp.ndarray:
    tile_reps = [num_particles] + [1] * len(leaf.shape)
    return jnp.tile(leaf[jnp.newaxis, ...], tile_reps)

  # Replicate the trunk params for each ensemble member.
  tiled_betas = jax.tree_util.tree_map(tile_fn, params[1])
  params = (params[0], tiled_betas, params[2])

  # Compute SVGD grads.
  vals, grads = NEW_svgd_grad(
      log_p_fn, kernel_fn, num_particles, params,
      avg_kernel_fn=avg_kernel_fn, temp=temp)
  alpha_grads, beta_grads, gamma_grads = grads

  # Project the SVGD grads back onto the constraint set by taking the mean.
  beta_grads = jax.tree_util.tree_map(lambda l: jnp.mean(l, axis=0), beta_grads)
  return vals, (alpha_grads, beta_grads, gamma_grads)


def bimo_train_p_svgd(
    log_p_fn: DataLogProbFn,
    kernel_fn: KernelFnWithData,
    init_params: BIMOParams,
    num_members: int,
    dataset: tf_data.Dataset,
    opt: optax.GradientTransformation,
    avg_kernel_fn: Union[KernelFnWithData, None],
    temp: float = 1.,
    num_steps: int = 10000,
    summarize_every: int = 1000) -> BIMOParams:
  """Trains a BIMO model with projected SVGD.

  Args:
    log_p_fn: A function which accepts parameters and returns the log
      probability under the model.
    kernel_fn: A function which computes kernel values for pairs of parameters.
    init_params: The initial particle parameters.
    num_members: The number of heads in the BIMO model.
    dataset: The dataset to train on.
    opt: The optimizer to use to update the parameters.
    avg_kernel_fn: Kernel used to compute weighted gradient averages.
    temp: The SVGD temperature.
    num_steps: The number of steps to train for.
    summarize_every: The number of steps between summaries.
  Returns:
    The trained parameters.
  """

  def grad_fn(params: BIMOParams, batch: Batch) -> PyTree:

    def avg_log_p_fn(params: BIMOParams) -> Val:
      log_ps = vmap(log_p_fn, in_axes=(None, 0))(params, batch)
      return jnp.mean(log_ps, axis=0)

    xs, _ = batch
    kernel_fn_with_data = lambda a, b: kernel_fn(a, b, xs)
    if avg_kernel_fn is not None:
      avg_kernel_fn_with_data = lambda a, b: avg_kernel_fn(a, b, xs)
    else:
      avg_kernel_fn_with_data = None

    return NEW_bimo_p_svgd_grad(
        avg_log_p_fn, kernel_fn_with_data, num_members, params,
        avg_kernel_fn=avg_kernel_fn_with_data, temp=temp)

  @jit
  def p_svgd_update(
      opt_state: optax.OptState,
      params: BIMOParams,
      batch: Batch) -> Tuple[optax.OptState, PyTree, PyTree]:
    vals, grads = grad_fn(params, batch)
    # Negate the gradient so we maximize the function.
    neg_grads = jax.tree_util.tree_map(lambda x: -x, grads)
    updates, opt_state = opt.update(neg_grads, opt_state)
    params = optax.apply_updates(params, updates)
    return opt_state, params, vals

  # Initialize the optimizer, and dataset iterator.
  opt_state = opt.init(init_params)
  data_itr = iter(tfds.as_numpy(dataset))
  params = init_params

  for step in range(num_steps):
    batch = next(data_itr)
    opt_state, params, (log_p, _) = p_svgd_update(
        opt_state, params, (batch.x, batch.y))
    if step % summarize_every == 0:
      print('Step %d loss: %0.6f' % (step, log_p))

  return params


def make_bimo_model(
    hidden_sizes: Sequence[int],
    num_classes: int,
    activation_fn=stax.Relu,
    W_init=stax.glorot_normal(),
    b_init=stax.zeros):
  """Constructs the BIMO model."""

  assert len(hidden_sizes) >= 2

  layers = []
  for size in hidden_sizes:
    layers.append(stax.Dense(size, W_init=W_init, b_init=b_init))
    layers.append(activation_fn)
  layers.append(stax.Dense(num_classes, W_init=W_init, b_init=b_init))

  out_head_init, out_head_apply = layers[-1]
  trunk_init, trunk_apply = stax.serial(*layers[1:-1])
  in_head_init, in_head_apply = layers[0]

  def model_apply(params: BIMOParams, x: jnp.ndarray) -> jnp.ndarray:
    out_head_params, trunk_params, in_head_params = params
    return out_head_apply(
        out_head_params,
        trunk_apply(
            trunk_params,
            in_head_apply(in_head_params, x)))

  def model_apply_individual(
      params: BIMOParams, x: jnp.ndarray, i: int) -> jnp.ndarray:
    out_head_params, trunk_params, in_head_params = params
    out_head_params = jax.tree_util.tree_map(lambda e: e[i], out_head_params)
    in_head_params = jax.tree_util.tree_map(lambda e: e[i], in_head_params)
    return model_apply((out_head_params, trunk_params, in_head_params), x)

  def model_init(key: PRNGKey, input_size: int, num_members: int) -> BIMOParams:
    k1, k2, k3 = jax.random.split(key, num=3)
    keys = jax.random.split(k1, num=num_members)
    h_out_sizes, h_params = vmap(in_head_init, in_axes=(0, None))(
        keys, (input_size,))
    h_out_size = jax.tree_util.tree_map(lambda x: x[0], h_out_sizes)
    g_out_size, g_params = trunk_init(k2, h_out_size)
    keys = jax.random.split(k3, num=num_members)
    _, f_params = vmap(out_head_init, in_axes=(0, None))(keys, g_out_size)
    params = (f_params, g_params, h_params)
    return params

  return model_init, model_apply, model_apply_individual


def log_gaussian_prior(params: PyTree, prior_variance: float = 1.):
  param_vec = vectorize_pytree(params)
  return (-0.5 / prior_variance) * jnp.sum(jnp.square(param_vec))


def categorical_log_prob(
    params: BIMOParams,
    data: Tuple[jnp.ndarray, jnp.ndarray],
    model_apply_fn=None) -> Val:
  """Computes the classification loss function."""
  assert model_apply_fn is not None
  x, y = data
  logits = model_apply_fn(params, x)
  log_prob = tfd.Categorical(logits=logits).log_prob(y)
  return log_prob.reshape([])


def make_dataset(data: testbed_base.Data, batch_size):
  """Constructs a Tensorflow dataset from the testbed data struct."""
  ds = tf_data.Dataset.from_tensor_slices(
      enn_base.Batch(data.x, data.y)).cache()
  n_data = data.x.shape[0]
  ds = ds.shuffle(
      min(n_data, 50 * batch_size),
      reshuffle_each_iteration=True, seed=0)
  ds = ds.repeat().batch(batch_size)
  return ds


def bimo_agent(
    data: testbed_base.Data,
    prior: testbed_base.PriorKnowledge,
) -> testbed_base.EpistemicSampler:
  """Creates a BIMO ENN agent."""

  FLAGS.hidden_sizes = [int(x) for x in FLAGS.hidden_sizes]
  batch_size = min(FLAGS.batch_size, prior.num_train)

  model_init, model_apply, model_apply_ind = make_bimo_model(
      FLAGS.hidden_sizes, prior.num_classes)

  key = jax.random.PRNGKey(0)

  init_params = model_init(key, prior.input_dim, FLAGS.num_heads)

  prior_var = FLAGS.prior_var
  if FLAGS.adaptive_prior_var:
    # The more data, the less we regularize.
    prior_var *= prior.num_train
    # The more noise, the more we regularize.
    prior_var /= (prior.temperature * 2)
    # The more input dimensions, the more we regularize
    prior_var /= prior.input_dim

  def log_prob_fn(params: BIMOParams, data: Tuple[jnp.ndarray, jnp.ndarray]):
    lprob = categorical_log_prob(params, data, model_apply_fn=model_apply)
    lprior = log_gaussian_prior(params, prior_variance=prior_var)
    if FLAGS.use_prior:
      return lprob + lprior
    else:
      return lprob

  kernel_fn_without_data = treemap_kernel(
      functools.partial(rbf_kernel, kernel_bandwidth=FLAGS.kernel_bandwidth))
  kernel_fn = lambda a, b, unused_x: kernel_fn_without_data(a, b)
  avg_kernel_fn = None

  if FLAGS.method == 'fsvgd':
    avg_kernel_fn = kernel_fn

    def inner_kernel(a, b, unused_xs):
      return rbf_kernel(
          jax.nn.log_softmax(a, axis=-1),
          jax.nn.log_softmax(b, axis=-1),
          kernel_bandwidth=FLAGS.fn_kernel_bandwidth)

    kernel_fn = functools.partial(
        functional_kernel, model_apply=model_apply, inner_kernel=inner_kernel)

  lr_fn = functools.partial(
      lr_perc_sched_fn,
      max_num_steps=FLAGS.num_train_steps,
      base_lr=FLAGS.base_lr,
      decay_rate=0.2,
      perc_steps=[.4, .6, .8])

  optimizer = optax.adam(lr_fn)

  dataset = make_dataset(data, batch_size)

  trained_params = bimo_train_p_svgd(
      log_prob_fn,
      kernel_fn,
      init_params,
      FLAGS.num_heads,
      dataset,
      opt=optimizer,
      avg_kernel_fn=avg_kernel_fn,
      temp=FLAGS.svgd_temp,
      num_steps=FLAGS.num_train_steps,
      summarize_every=FLAGS.summarize_every)

  def enn_sampler(x: enn_base.Array, key: chex.PRNGKey) -> enn_base.Array:
    num_inputs = x.shape[0]
    inds = jax.random.categorical(
        key, jnp.zeros([FLAGS.num_heads]), shape=[num_inputs])
    # [batch_size, num_classes]
    logits = vmap(model_apply_ind, in_axes=(None, 0, 0))(
        trained_params, x, inds)
    return logits

  return enn_sampler


def main(_):
  prior_kowledge = testbed_base.PriorKnowledge(
      input_dim=FLAGS.input_dim,
      num_classes=FLAGS.num_classes,
      num_train=int(FLAGS.input_dim * FLAGS.data_ratio),
      temperature=FLAGS.temperature,
      tau=FLAGS.tau,
  )
  problem = leaderboard.problem_from_config(
      leaderboard.ProblemConfig(prior_kowledge, seed=FLAGS.seed))

  experiment.run(bimo_agent, problem)


if __name__ == '__main__':
  app.run(main)

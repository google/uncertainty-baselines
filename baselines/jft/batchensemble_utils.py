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

"""JAX BatchEnsemble training related functions."""

import functools
import re
from typing import Any, Callable, Mapping, Optional, Sequence, Tuple

from clu import metric_writers
import flax.optim
import flax.struct
import flax.traverse_util
import jax
import jax.numpy as jnp

from dune.experts.jax.core import jax as core


EvaluationOutput = Tuple[jnp.ndarray, ...]
Module = type(functools)  # Python module.
Params = Mapping[str, Any]
MetricWriter = metric_writers.MetricWriter
PmapEvaluationFn = Callable[
    [jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray],
    EvaluationOutput]


def tree_count_infs_nans(tree, psum_axis_name=None):
  leaves = jax.tree_leaves(tree)
  num_infs = sum(jnp.sum(jnp.isinf(x)) for x in leaves)
  num_nans = sum(jnp.sum(jnp.isnan(x)) for x in leaves)
  if psum_axis_name:
    num_infs, num_nans = jax.lax.psum((num_infs, num_nans),
                                      axis_name=psum_axis_name)
  return num_infs, num_nans


def update_fn_be(
    opt: flax.optim.Optimizer,
    rngs: Mapping[str, jnp.ndarray],
    lr: jnp.ndarray,
    images: jnp.ndarray,
    labels: jnp.ndarray,
    batch_loss_fn: Callable[..., jnp.ndarray],
    weight_decay_fn: Optional[Callable[[Any, float], Any]],
    plot_grad_norm_name_fn: Callable[[str], bool],
    plot_grads_nan_inf: bool,
    max_grad_norm_global: Optional[float],
    frozen_vars_patterns: Optional[Sequence[str]],
    fast_weight_lr_multiplier: float):
  """Updates a model on the given inputs for one step.

  Args:
    opt: Flax optimizer used during training.
    rngs: A random number generator to be passed by stochastic operations.
    lr: The learning rate to use in each device.
    images: Array containing the images in a batch.
    labels: Array containing the labels in a batch.
    batch_loss_fn: Loss function that takes (params, images, labels, rng) as
      inputs and produces the loss value for an entire batch.
    weight_decay_fn: Function that takes a parameter and returns a new parameter
      with weight decay applied. Use None to avoid any weight decay.
    plot_grad_norm_name_fn: Function that takes a string and returns True/False
      denoting whether to plot the gradient of the params with matching name.
      Use None to not produce any plot.
    plot_grads_nan_inf: Boolean denoting whether to plot the number of NaNs or
      Infs in the gradients.
    max_grad_norm_global: Float (or None) denoting the maximum norm of the
      gradients allowed for before clipping. If the norm is larger than this,
      the gradients are scaled to have this norm. Use None to avoid any norm
      clipping.
    frozen_vars_patterns: List of regex patterns corresponding to variables
      which should be removed from gradients.
    fast_weight_lr_multiplier: the ratio of the fast weights LR to the slow
      weights one.

  Returns:
    The optimizer with the updated parameters and state.
    The split rng.
    The loss value of the batch before the update.
    A dictionary containing auxiliary information such as plots.
  """

  # If rng is provided: split rng, and return next_rng for the following step.
  rngs, next_rngs = core.tree_rngs_split(rngs, num_splits=2)
  (loss, aux), grads = jax.value_and_grad(
      batch_loss_fn, has_aux=True)(opt.target, images, labels, rngs=rngs)

  # Average gradients.
  grads = jax.lax.pmean(grads, axis_name="batch")

  # TODO(basilm, jpuigcerver): Find better ways to freeze/clip gradients.
  if frozen_vars_patterns:
    regexes = [re.compile(ptn) for ptn in frozen_vars_patterns]
    match_fn = lambda name: any([regex.search(name) for regex in regexes])
    grads = core.tree_map_with_names(jnp.zeros_like, grads, match_fn)

  loss = jax.lax.pmean(loss, axis_name="batch")

  if plot_grads_nan_inf:
    # If you think that this is heavily affecting your training speed, use
    # `config.plot_grads_nan_inf = False` in the config file.
    num_infs, num_nans = tree_count_infs_nans(
        grads, psum_axis_name="batch")
    aux["debug/num_infs"] = num_infs
    aux["debug/num_nans"] = num_nans

  if plot_grad_norm_name_fn:
    # Compute norm of selected parameters and add them as auxiliary metrics.
    aux.update({
        f"grads_norm/{name}": jnp.sqrt(jnp.vdot(grad, grad))
        for name, grad in core.tree_flatten_with_names(grads)[0]
        if plot_grad_norm_name_fn(name)
    })

  if max_grad_norm_global and max_grad_norm_global > 0.0:
    # Normalize by "global" norm (i.e. flatten all parameters).
    grads, global_norm = core.tree_clip_norm_global_pmax(
        grads, max_grad_norm_global, axis_name="batch")
    aux["grad_norm_global"] = global_norm

  if fast_weight_lr_multiplier and fast_weight_lr_multiplier != 1.0:
    fast_weights_lr_fn = lambda x: x * fast_weight_lr_multiplier
    match_fn = lambda name: ("fast_weight_alpha" in name or "fast_weight_gamma"  # pylint: disable=g-long-lambda
                             in name)
    grads = core.tree_map_with_names(fast_weights_lr_fn, grads, match_fn)

  opt = opt.apply_gradient(grads, learning_rate=lr)

  if weight_decay_fn:
    opt = opt.replace(target=weight_decay_fn(opt.target, lr))
  return opt, next_rngs, loss, aux

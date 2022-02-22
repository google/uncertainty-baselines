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

"""JAX BatchEnsemble training related functions."""

import dataclasses
import functools
from typing import Any, Callable, Mapping, Optional, Tuple

from clu import metric_writers
import flax.optim
import flax.struct
import flax.traverse_util
import jax
import jax.numpy as jnp
import numpy as np


EvaluationOutput = Tuple[jnp.ndarray, ...]
Module = type(functools)  # Python module.
Params = Mapping[str, Any]
MetricWriter = metric_writers.MetricWriter
PmapEvaluationFn = Callable[
    [jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray],
    EvaluationOutput]


# TODO(dusenberrymw,zmariet): Clean up and generalize these log marginal probs.
def log_average_softmax_probs(logits: jnp.ndarray) -> jnp.ndarray:
  # TODO(zmariet): dedicated eval loss function.
  ens_size, _, _ = logits.shape
  log_p = jax.nn.log_softmax(logits)  # (ensemble_size, batch_size, num_classes)
  log_p = jax.nn.logsumexp(log_p, axis=0) - jnp.log(ens_size)
  return log_p


def log_average_sigmoid_probs(logits: jnp.ndarray) -> jnp.ndarray:
  ens_size, _, _ = logits.shape
  log_p = jax.nn.log_sigmoid(logits)  # (ensemble_size, batch_size, num_classes)
  log_p = jax.nn.logsumexp(log_p, axis=0) - jnp.log(ens_size)
  log_not_p = jax.nn.log_sigmoid(-logits)
  log_not_p = jax.nn.logsumexp(log_not_p, axis=0) - jnp.log(ens_size)
  log_p = log_p - log_not_p
  return log_p


def tree_clip_norm_global_pmax(tree, max_norm, axis_name):
  """Global norm clipping, with pmax of global norm before clipping."""
  global_norm = jnp.sqrt(sum(jnp.vdot(x, x) for x in jax.tree_leaves(tree)))
  global_norm = jax.lax.pmax(global_norm, axis_name=axis_name)
  factor = jnp.minimum(1.0, max_norm / global_norm)
  return jax.tree_map(lambda x: factor * x, tree), global_norm


def _traverse_with_names(tree):
  """Traverses nested dicts/dataclasses and emits (leaf_name, leaf_val)."""
  if dataclasses.is_dataclass(tree):
    tree = flax.serialization.to_state_dict(tree)
  if isinstance(tree, (dict, flax.core.FrozenDict)):
    keys = sorted(tree.keys())
    for key in keys:
      for path, v in _traverse_with_names(tree[key]):
        yield (key + '/' + path).rstrip('/'), v
  else:
    yield '', tree


def tree_flatten_with_names(tree):
  """Populates tree_flatten with leaf names.

  This function populates output of tree_flatten with leaf names, using a
  custom traversal that produces names if provided. The custom traversal does
  NOT have to traverse tree in the same order as jax, as we take care of
  automatically aligning jax' and custom traversals.

  Args:
    tree: python tree.

  Returns:
    A list of values with names: [(name, value), ...].
    A PyTreeDef tree definition object.
  """
  vals, tree_def = jax.tree_flatten(tree)

  # "Fake" token tree that is use to track jax internal tree traversal and
  # adjust our custom tree traversal to be compatible with it.
  tokens = range(len(vals))
  token_tree = tree_def.unflatten(tokens)
  val_names, perm = zip(*_traverse_with_names(token_tree))
  inv_perm = np.argsort(perm)

  # Custom traverasal should visit the same number of leaves.
  if len(val_names) != len(vals):
    raise ValueError(f'Pytree traversal detected {len(val_names)} names, '
                     f'but {len(vals)} leafs.\nTreeDef is:\n{tree_def}')

  return [(val_names[i], v) for i, v in zip(inv_perm, vals)], tree_def


def tree_map_with_names(f, param_tree, match_name_fn=lambda name: True):
  """Like jax.tree_map but with a filter on the leaf path name.

  Args:
    f: The function to be applied to each parameter in `param_tree`.
    param_tree: The tree of parameters `f` should be applied to.
    match_name_fn: This function is called with each tree leave's path name,
      which has a path-like format ("a/b/c"), and decides whether `f` should
      be applied to that leaf or the leaf should be kept as-is.

  Returns:
    A tree identical in structure to `param_tree` but with the leaves the
    result of calling `f` on them in the cases where `match_name_fn` returns
    True for that leaf's path name.
  """
  names_and_vals, tree_def = tree_flatten_with_names(param_tree)
  vals = [f(v) if match_name_fn(name) else v for name, v in names_and_vals]
  return tree_def.unflatten(vals)


def tree_rngs_split(rngs, num_splits=2):
  """Splits a PyTree of PRNGKeys into num_splits PyTrees."""
  rngs = jax.tree_map(lambda rng: jax.random.split(rng, num_splits), rngs)
  slice_rngs = lambda rngs, i: jax.tree_map(lambda rng: rng[i], rngs)
  return tuple(slice_rngs(rngs, i) for i in range(num_splits))


def update_fn_be(
    opt: flax.optim.Optimizer,
    rngs: Mapping[str, jnp.ndarray],
    lr: jnp.ndarray,
    images: jnp.ndarray,
    labels: jnp.ndarray,
    batch_loss_fn: Callable[..., jnp.ndarray],
    weight_decay_fn: Optional[Callable[[Any, float], Any]],
    max_grad_norm_global: Optional[float],
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
    max_grad_norm_global: Float (or None) denoting the maximum norm of the
      gradients allowed for before clipping. If the norm is larger than this,
      the gradients are scaled to have this norm. Use None to avoid any norm
      clipping.
    fast_weight_lr_multiplier: the ratio of the fast weights LR to the slow
      weights one.

  Returns:
    The optimizer with the updated parameters and state.
    The split rng.
    The loss value of the batch before the update.
    A dictionary containing auxiliary information such as plots.
  """

  # If rng is provided: split rng, and return next_rng for the following step.
  rngs, next_rngs = tree_rngs_split(rngs, num_splits=2)
  (loss, aux), grads = jax.value_and_grad(
      batch_loss_fn, has_aux=True)(opt.target, images, labels, rngs=rngs)

  # Average gradients.
  grads = jax.lax.pmean(grads, axis_name='batch')
  loss = jax.lax.pmean(loss, axis_name='batch')
  aux['training_loss'] = loss

  if max_grad_norm_global and max_grad_norm_global > 0.0:
    # Normalize by 'global' norm (i.e. flatten all parameters).
    grads, global_norm = tree_clip_norm_global_pmax(
        grads, max_grad_norm_global, axis_name='batch')
    aux['grad_norm_global'] = global_norm

  if fast_weight_lr_multiplier and fast_weight_lr_multiplier != 1.0:
    fast_weights_lr_fn = lambda x: x * fast_weight_lr_multiplier
    match_fn = lambda name: ('fast_weight_alpha' in name or 'fast_weight_gamma'  # pylint: disable=g-long-lambda
                             in name)
    grads = tree_map_with_names(fast_weights_lr_fn, grads, match_fn)

  opt = opt.apply_gradient(grads, learning_rate=lr)

  if weight_decay_fn:
    params = weight_decay_fn(opt.target, lr)
    opt = opt.replace(target=params)

  aux['learning_rate'] = lr
  return opt, next_rngs, aux


def broadcast_batchensemble_biases(params, be_layers, ensemble_size):
  for layer in be_layers:
    for block in [0, 1]:
      be_block = params['Transformer'][f'encoderblock_{layer}']['MlpBlock_3']
      be_block[f'Dense_{block}']['bias'] = jnp.tile(
          be_block[f'Dense_{block}']['bias'], (ensemble_size, 1))
  return params

# coding=utf-8
# Copyright 2023 The Uncertainty Baselines Authors.
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
import logging
from typing import Any, Callable, Mapping, Optional, Tuple

from clu import metric_writers
import flax.optim
import flax.struct
import flax.traverse_util
import jax
import jax.numpy as jnp
import numpy as np
import train_utils  # local file import from baselines.jft

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
      which has a path-like format ("a/b/c"), and decides whether `f` should be
      applied to that leaf or the leaf should be kept as-is.

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


def update_fn_be(opt: flax.optim.Optimizer, rng: jnp.ndarray, lr: jnp.ndarray,
                 images: jnp.ndarray, labels: jnp.ndarray,
                 batch_loss_fn: Callable[..., jnp.ndarray],
                 weight_decay_fn: Optional[Callable[[Any, float], Any]],
                 max_grad_norm_global: Optional[float],
                 fast_weight_lr_multiplier: float):
  """Updates a model on the given inputs for one step.

  Args:
    opt: Flax optimizer used during training.
    rng: A random number generator to be passed by stochastic operations.
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
  rng, next_rng = jax.random.split(rng)
  (loss, aux), grads = jax.value_and_grad(
      batch_loss_fn, has_aux=True)(
          opt.target, images, labels, rng=rng)

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
    params = weight_decay_fn(opt.target, lr)  # pytype: disable=wrong-arg-types  # jax-ndarray
    opt = opt.replace(target=params)

  aux['learning_rate'] = lr
  return opt, next_rng, aux


def maybe_broadcast_batchensemble_biases(params, be_layers, ensemble_size):
  """Tiles BE biases when seeding downstream weights from a deterministic model."""
  for layer in be_layers:
    for block in [0, 1]:
      be_block = params['Transformer'][f'encoderblock_{layer}']['MlpBlock_3']

      # The biases already have the right shape if we are restarting from a
      # checkpoint (e.g., after a job got preempted).
      if be_block[f'Dense_{block}']['bias'].ndim != 2:
        be_block[f'Dense_{block}']['bias'] = jnp.tile(
            be_block[f'Dense_{block}']['bias'], (ensemble_size, 1))
  return params


def create_init(model, config, train_ds):
  """Create the initialization function for model parameters.

  Args:
    model: The model to be used in updates.
    config: The config of the experiment.
    train_ds: tf.data.Dataset.

  Returns:
    Function that returns initialized model parameters.
  """
  local_batch_size = config.batch_size // jax.process_count()
  # We want all parameters to be created in host RAM, not on any device, they'll
  # be sent there later as needed, otherwise we already encountered two
  # situations where we allocate them twice.
  @functools.partial(jax.jit, backend='cpu')
  def init(rng):
    image_size = tuple(train_ds.element_spec['image'].shape[2:])
    logging.info('image_size = %s', image_size)
    dummy_input = jnp.zeros((local_batch_size,) + image_size, jnp.float32)
    params = flax.core.unfreeze(model.init(rng, dummy_input,
                                           train=False))['params']

    # Set bias in the head to a low value, such that loss is small initially.
    params['batchensemble_head']['bias'] = jnp.full_like(
        params['batchensemble_head']['bias'], config.get('init_head_bias', 0))

    # init head kernel to all zeros for fine-tuning
    if config.get('model_init'):
      params['batchensemble_head']['kernel'] = jnp.full_like(
          params['batchensemble_head']['kernel'], 0)

    return params

  return init


def create_batch_loss_fn(model, config):
  """Create the update function from model and config.

  Args:
    model: The model to be used in updates.
    config: The config of the experiment.

  Returns:
    The function that updates the model for one step.
  """

  def batch_loss_fn(params, images, labels, rng):
    logits, _ = model.apply({'params': flax.core.freeze(params)},
                            images,
                            train=True,
                            rngs={'dropout': rng})
    labels = jnp.tile(labels, (config.model.transformer.ens_size, 1))
    loss_fn = getattr(train_utils, config.get('loss', 'sigmoid_xent'))
    loss = jnp.mean(loss_fn(logits=logits, labels=labels))
    return loss, dict()

  return batch_loss_fn


def create_update_fn(model, config):
  """Create the update function from model and config.

  Args:
    model: The model to be used in updates.
    config: The config of the experiment.

  Returns:
    The function that updates the model for one step.
  """

  batch_loss_fn = create_batch_loss_fn(model, config)

  @functools.partial(jax.pmap, axis_name='batch', donate_argnums=(0))
  def update_fn(opt, lr, images, labels, rngs):
    return update_fn_be(
        opt=opt,
        rng=rngs,
        lr=lr,
        images=images,
        labels=labels,
        batch_loss_fn=batch_loss_fn,
        weight_decay_fn=train_utils.get_weight_decay_fn(
            weight_decay_rules=config.get('weight_decay', []) or [],
            rescale_value=1.),
        max_grad_norm_global=config.get('grad_clip_norm', None),
        fast_weight_lr_multiplier=config.get('fast_weight_lr_multiplier', None))

  return update_fn


# TODO(trandustin, zmariet): Unify all evaluation functions and other utility
# functions used in different models.
def create_evaluation_fn(model, config):
  """Create the evaluation function from model and config.

  Args:
    model: The model to be used in updates.
    config: The config of the experiment.

  Returns:
    The function that evaluates the model for one step.
  """
  @functools.partial(jax.pmap, axis_name='batch')
  def evaluation_fn(params, images, labels, mask):
    # Ignore the entries with all zero labels for evaluation.
    mask *= labels.max(axis=1)
    tiled_logits, out = model.apply({'params': flax.core.freeze(params)},
                                    images,
                                    train=False)

    loss_name = config.get('loss', 'sigmoid_xent')
    # TODO(dusenberrymw,zmariet): Clean up and generalize this.
    ens_size = config.model.transformer.ens_size
    if loss_name == 'sigmoid_xent':
      ens_logits = log_average_sigmoid_probs(
          jnp.asarray(jnp.split(tiled_logits, ens_size)))
    else:  # softmax
      ens_logits = log_average_softmax_probs(
          jnp.asarray(jnp.split(tiled_logits, ens_size)))
    pre_logits = jnp.concatenate(
        jnp.split(out['pre_logits'], ens_size), axis=-1)

    losses = getattr(train_utils, loss_name)(
        logits=ens_logits,
        labels=labels[:, :config.num_classes],
        reduction=False)
    loss = jax.lax.psum(losses * mask, axis_name='batch')

    top1_idx = jnp.argmax(ens_logits, axis=1)
    top1_correct = jnp.take_along_axis(labels, top1_idx[:, None], axis=1)[:, 0]
    ncorrect = jax.lax.psum(top1_correct * mask, axis_name='batch')
    n = jax.lax.psum(mask, axis_name='batch')

    metric_args = jax.lax.all_gather(
        [ens_logits, labels, pre_logits, mask],
        axis_name='batch')
    return ncorrect, loss, n, metric_args

  return evaluation_fn

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

"""Rank-1 BNN Vision Transformer."""

import functools
import multiprocessing
import os
import re
from typing import Any, Callable, Iterable, Mapping

from absl import app
from absl import flags
from absl import logging
from clu import metric_writers
from clu import parameter_overview
from clu import periodic_actions
from clu import preprocess_spec
import flax
import jax
import jax.numpy as jnp
import ml_collections.config_flags
import numpy as np
import robustness_metrics as rm
import tensorflow as tf
import tensorflow_probability.substrates.jax as tfp
import uncertainty_baselines as ub
import batchensemble_utils  # local file import from baselines.jft
import checkpoint_utils  # local file import from baselines.jft
import data_uncertainty_utils  # local file import from baselines.jft
import input_utils  # local file import from baselines.jft
import ood_utils  # local file import from baselines.jft
import preprocess_utils  # local file import from baselines.jft
import subpopl_utils  # local file import from baselines.jft
import train_utils  # local file import from baselines.jft

# TODO(dusenberrymw): Open-source remaining imports.
fewshot = None

ml_collections.config_flags.DEFINE_config_file(
    'config', None, 'Training configuration.', lock_config=True)
flags.DEFINE_string('output_dir', default=None, help='Work unit directory.')
flags.DEFINE_integer(
    'num_cores', default=None, help='Unused. How many devices being used.')
flags.DEFINE_boolean(
    'use_gpu', default=None, help='Unused. Whether or not running on GPU.')
flags.DEFINE_string('tpu', None,
                    'Unused. Name of the TPU. Only used if use_gpu is False.')

FLAGS = flags.FLAGS

DType = type(jnp.float32)
InitializeFn = Callable[[jnp.ndarray, Iterable[int], DType], jnp.ndarray]
Params = Mapping[str, Any]


def _tree_map_with_pattern(f, tree, regex_patterns, *rest, is_leaf=None):
  """Performs a JAX-style tree_map with filtering based on regex rules.

  Args:
    f: A function that takes in `1 + len(rest)` arguments, to be applied at the
      corresponding leaves of the pytrees that match the pattern.
    tree: A pytree to be mapped over, with each leaf providing the first
      positional argument to ``f``. The top level should be a dict.
    regex_patterns: A list of regex pattern used for variable name matching
      based on the flattened path. For variables with a matching name, the value
      is replaced by the output of the `f` applied to the value of the variable.
    *rest: A tuple of pytrees, each of which has the same structure as tree or
      has tree as a prefix.
    is_leaf: an optional function that takes the next nested dictionary and
      nested keys and returns True if the nested dictionary is a leaf (i.e.,
      should not be flattened further).

  Returns:
    A tree, transformed by `f` according to the given rules.
  """

  def _f(path_tuple, v, *r):
    vname = '/'.join(path_tuple)
    for pattern in regex_patterns:
      if re.match(pattern, vname):
        if jax.process_index() == 0:
          logging.info('Updating %s due to `%s`', vname, pattern)
        return f(v, *r)
    return v

  flat_tree = flax.traverse_util.flatten_dict(tree, is_leaf=is_leaf)
  keys = flat_tree.keys()
  values = flat_tree.values()
  rest_values = [flax.traverse_util.flatten_dict(r).values() for r in rest]
  updated_flat_tree = {
      k: _f(k, v, *r) for k, v, *r in zip(keys, values, *rest_values)
  }
  updated_tree = flax.traverse_util.unflatten_dict(updated_flat_tree)
  return updated_tree


def _create_key_tree(tree, key, is_leaf=None):
  """Create a new tree of random keys with the same structure as `tree`."""
  flat_tree = flax.traverse_util.flatten_dict(tree, is_leaf=is_leaf)
  rng_keys = jax.random.split(key, len(flat_tree.keys()))
  flat_key_tree = {k: r for k, r in zip(flat_tree.keys(), rng_keys)}
  key_tree = flax.traverse_util.unflatten_dict(flat_key_tree)
  return key_tree


def _pointwise_to_loc_scale(x, key):
  """Convert a point value x into the parameters for a location-scale dist."""
  loc = x
  # Initialize the unconstrained scale params such that after a softplus the
  # value would be close to zero.
  mean = -3.
  std = 0.1
  unconstrained_scale = mean + std * jax.random.truncated_normal(
      key, lower=-2, upper=2, shape=x.shape)
  params = {'loc': loc, 'unconstrained_scale': unconstrained_scale}
  return params


def _sample_mean_field_gaussian(dist_params, key, eps=1e-8):
  assert isinstance(dist_params, dict), dist_params
  loc = dist_params['loc']
  unconstrained_scale = dist_params['unconstrained_scale']
  scale = jax.nn.softplus(unconstrained_scale) + eps
  return loc + scale * jax.random.normal(key=key, shape=loc.shape)


def _compute_gaussian_kl_divergence(dist_params,
                                    prior_mean,
                                    prior_std,
                                    eps=1e-8):
  """Computes the KL divergence between Gaussian posterior and prior dists."""
  assert isinstance(dist_params, dict), dist_params
  loc = dist_params['loc']
  unconstrained_scale = dist_params['unconstrained_scale']
  scale = jax.nn.softplus(unconstrained_scale) + eps
  posterior_dist = tfp.distributions.Normal(loc, scale)
  prior_dist = tfp.distributions.Normal(prior_mean, prior_std)
  kl = jnp.sum(tfp.distributions.kl_divergence(posterior_dist, prior_dist))
  return kl


def _get_rank1_params(params, rank1_regex_patterns=None):
  """Returns a list containing only the rank-1 parameter subtrees."""
  if not rank1_regex_patterns:
    rank1_regex_patterns = ['.*fast_weight.*']

  def is_rank1(prefix):
    path = '/'.join(prefix)
    return any(re.match(pattern, path) for pattern in rank1_regex_patterns)

  def is_leaf(prefix, xs):
    del xs
    return is_rank1(prefix)

  flat_params = flax.traverse_util.flatten_dict(params, is_leaf=is_leaf)
  filtered_flat_params = [v for k, v, in flat_params.items() if is_rank1(k)]
  return filtered_flat_params


def init_gaussian_rank1(params, key, rank1_regex_patterns=None):
  """Initializes the rank-1 vectors as parameters for mean-field Gaussians."""
  if not rank1_regex_patterns:
    rank1_regex_patterns = ['.*fast_weight.*']
  key_tree = _create_key_tree(params, key)
  return _tree_map_with_pattern(_pointwise_to_loc_scale, params,
                                rank1_regex_patterns, key_tree)


def sample_gaussian_rank1(params, key, rank1_regex_patterns=None):
  """Samples the rank-1 mean-field Gaussians to yield sampled parameters."""
  if not rank1_regex_patterns:
    rank1_regex_patterns = ['.*fast_weight.*']

  def is_leaf(prefix, xs):
    del xs
    path = '/'.join(prefix)
    return any(re.match(pattern, path) for pattern in rank1_regex_patterns)

  key_tree = _create_key_tree(params, key, is_leaf=is_leaf)
  return _tree_map_with_pattern(
      _sample_mean_field_gaussian,
      params,
      rank1_regex_patterns,
      key_tree,
      is_leaf=is_leaf)


def gaussian_rank1_kl_divergence(params,
                                 prior_mean,
                                 prior_std,
                                 rank1_regex_patterns=None):
  """Computes the KL(q||p) between rank-1 Gaussian posterior & prior dists."""
  if not rank1_regex_patterns:
    rank1_regex_patterns = ['.*fast_weight.*']
  rank1_params = _get_rank1_params(params, rank1_regex_patterns)

  kls = []
  for dist_params in rank1_params:
    kls.append(
        _compute_gaussian_kl_divergence(
            dist_params, prior_mean=prior_mean, prior_std=prior_std))
  kl = jnp.sum(jnp.asarray(kls))
  return kl


def main(config, output_dir):

  seed = config.get('seed', 0)
  rng = jax.random.PRNGKey(seed)
  tf.random.set_seed(seed)

  if config.get('data_dir'):
    logging.info('data_dir=%s', config.data_dir)
  logging.info('Output dir: %s', output_dir)
  tf.io.gfile.makedirs(output_dir)

  save_checkpoint_path = None
  if config.get('checkpoint_steps'):
    save_checkpoint_path = os.path.join(output_dir, 'checkpoint.npz')

  # Create an asynchronous multi-metric writer.
  writer = metric_writers.create_default_writer(
      output_dir, just_logging=jax.process_index() > 0)

  # The pool is used to perform misc operations such as logging in async way.
  pool = multiprocessing.pool.ThreadPool()

  def write_note(note):
    if jax.process_index() == 0:
      logging.info('NOTE: %s', note)

  write_note('Initializing...')

  # Verify settings to make sure no checkpoints are accidentally missed.
  if config.get('keep_checkpoint_steps'):
    assert config.get('checkpoint_steps'), 'Specify `checkpoint_steps`.'
    assert config.keep_checkpoint_steps % config.checkpoint_steps == 0, (
        f'`keep_checkpoint_steps` ({config.checkpoint_steps}) should be'
        f'divisible by `checkpoint_steps ({config.checkpoint_steps}).`')

  batch_size = config.batch_size
  batch_size_eval = config.get('batch_size_eval', batch_size)
  if (batch_size % jax.device_count() != 0 or
      batch_size_eval % jax.device_count() != 0):
    raise ValueError(f'Batch sizes ({batch_size} and {batch_size_eval}) must '
                     f'be divisible by device number ({jax.device_count()})')

  local_batch_size = batch_size // jax.process_count()
  local_batch_size_eval = batch_size_eval // jax.process_count()
  logging.info(
      'Global batch size %d on %d hosts results in %d local batch size. '
      'With %d devices per host (%d devices total), that\'s a %d per-device '
      'batch size.', batch_size, jax.process_count(), local_batch_size,
      jax.local_device_count(), jax.device_count(),
      local_batch_size // jax.local_device_count())

  write_note('Initializing train dataset...')
  rng, train_ds_rng = jax.random.split(rng)
  train_ds_rng = jax.random.fold_in(train_ds_rng, jax.process_index())

  train_ds = input_utils.get_data(
      dataset=config.dataset,
      split=config.train_split,
      rng=train_ds_rng,
      process_batch_size=local_batch_size,
      preprocess_fn=preprocess_spec.parse(
          spec=config.pp_train, available_ops=preprocess_utils.all_ops()),
      shuffle_buffer_size=config.shuffle_buffer_size,
      prefetch_size=config.get('prefetch_to_host', 2),
      data_dir=config.get('data_dir'))

  write_note('Initializing val dataset(s)...')

  def _get_val_split(dataset, split, pp_eval, data_dir=None):
    # We do ceil rounding such that we include the last incomplete batch.
    nval_img = input_utils.get_num_examples(
        dataset,
        split=split,
        process_batch_size=local_batch_size_eval,
        drop_remainder=False,
        data_dir=data_dir)
    val_steps = int(np.ceil(nval_img / batch_size_eval))
    logging.info('Running validation for %d steps for %s, %s', val_steps,
                 dataset, split)

    if isinstance(pp_eval, str):
      pp_eval = preprocess_spec.parse(
          spec=pp_eval, available_ops=preprocess_utils.all_ops())

    val_ds = input_utils.get_data(
        dataset=dataset,
        split=split,
        rng=None,
        process_batch_size=local_batch_size_eval,
        preprocess_fn=pp_eval,
        cache=config.get('val_cache', 'batched'),
        num_epochs=1,
        repeat_after_batching=True,
        shuffle=False,
        prefetch_size=config.get('prefetch_to_host', 2),
        drop_remainder=False,
        data_dir=data_dir)

    return val_ds

  val_ds_splits = {
      'val':
          _get_val_split(
              config.dataset,
              split=config.val_split,
              pp_eval=config.pp_eval,
              data_dir=config.get('data_dir'))
  }

  if config.get('test_split'):
    val_ds_splits.update({
        'test':
            _get_val_split(
                config.dataset,
                split=config.test_split,
                pp_eval=config.pp_eval,
                data_dir=config.get('data_dir'))
    })

  if config.get('subpopl_cifar_data_file'):
    dataset_builder = input_utils.cifar_from_sql(
        sql_database=config.subpopl_cifar_data_file,
        num_classes=config.num_classes)

    subpopl_val_ds_splits = {  # pylint: disable=g-complex-comprehension
        client_id: _get_val_split(
            dataset_builder,
            split=client_id,
            pp_eval=config.pp_eval_subpopl_cifar,
            data_dir=config.subpopl_cifar_data_file)
        for client_id in dataset_builder.client_ids
    }

  if config.get('eval_on_cifar_10h'):
    cifar10_to_cifar10h_fn = data_uncertainty_utils.create_cifar10_to_cifar10h_fn(
        config.get('data_dir', None))
    preprocess_fn = preprocess_spec.parse(
        spec=config.pp_eval_cifar_10h, available_ops=preprocess_utils.all_ops())
    pp_eval = lambda ex: preprocess_fn(cifar10_to_cifar10h_fn(ex))
    val_ds_splits['cifar_10h'] = _get_val_split(
        'cifar10',
        split=config.get('cifar_10h_split') or 'test',
        pp_eval=pp_eval,
        data_dir=config.get('data_dir'))

  elif config.get('eval_on_imagenet_real'):
    imagenet_to_real_fn = data_uncertainty_utils.create_imagenet_to_real_fn()
    preprocess_fn = preprocess_spec.parse(
        spec=config.pp_eval_imagenet_real,
        available_ops=preprocess_utils.all_ops())
    pp_eval = lambda ex: preprocess_fn(imagenet_to_real_fn(ex))
    val_ds_splits['imagenet_real'] = _get_val_split(
        'imagenet2012_real',
        split=config.get('imagenet_real_split') or 'validation',
        pp_eval=pp_eval,
        data_dir=config.get('data_dir'))

  ood_ds = {}
  if config.get('ood_datasets') and config.get('ood_methods'):
    if config.get('ood_methods'):  #  config.ood_methods is not a empty list
      logging.info('loading OOD dataset = %s', config.get('ood_datasets'))
      ood_ds, ood_ds_names = ood_utils.load_ood_datasets(
          config.dataset,
          config.ood_datasets,
          config.ood_split,
          config.pp_eval,
          config.pp_eval_ood,
          config.ood_methods,
          config.train_split,
          config.get('data_dir'),
          _get_val_split,
      )

  ntrain_img = input_utils.get_num_examples(
      config.dataset,
      split=config.train_split,
      process_batch_size=local_batch_size,
      data_dir=config.get('data_dir'))
  steps_per_epoch = ntrain_img // batch_size

  if config.get('num_epochs'):
    total_steps = int(config.num_epochs * steps_per_epoch)
    assert not config.get('total_steps'), 'Set either num_epochs or total_steps'
  else:
    total_steps = config.total_steps

  logging.info('Total train data points: %d', ntrain_img)
  logging.info(
      'Running for %d steps, that means %f epochs and %d steps per epoch',
      total_steps, total_steps * batch_size / ntrain_img, steps_per_epoch)

  write_note('Initializing model...')
  logging.info('config.model = %s', config.model)
  model = ub.models.vision_transformer_be(
      num_classes=config.num_classes, **config.model)
  ens_size = config.model.transformer.ens_size
  rank1_regex_patterns = ['.*fast_weight.*']

  # We want all parameters to be created in host RAM, not on any device, they'll
  # be sent there later as needed, otherwise we already encountered two
  # situations where we allocate them twice.
  @functools.partial(jax.jit, backend='cpu')
  def init(rng):
    image_size = tuple(train_ds.element_spec['image'].shape[2:])
    logging.info('image_size = %s', image_size)
    dummy_input = jnp.zeros((local_batch_size,) + image_size, jnp.float32)
    init_rng, rank1_init_rng = jax.random.split(rng)
    params = flax.core.unfreeze(model.init(init_rng, dummy_input,
                                           train=False))['params']

    # Set bias in the head to a low value, such that loss is small initially.
    params['batchensemble_head']['bias'] = jnp.full_like(
        params['batchensemble_head']['bias'], config.get('init_head_bias', 0))

    # init head kernel to all zeros for fine-tuning
    if config.get('model_init'):
      params['batchensemble_head']['kernel'] = jnp.full_like(
          params['batchensemble_head']['kernel'], 0)

    # Initialize the rank-1 weights as parameters for a mean-field Gaussian
    # variational posterior.
    params = init_gaussian_rank1(
        params, rank1_init_rng, rank1_regex_patterns=rank1_regex_patterns)

    return params

  rng, rng_init = jax.random.split(rng)
  params_cpu = init(rng_init)

  if jax.process_index() == 0:
    num_params = sum(p.size for p in jax.tree_flatten(params_cpu)[0])
    parameter_overview.log_parameter_overview(params_cpu)
    writer.write_scalars(step=0, scalars={'num_params': num_params})

  # TODO(dusenberrymw): Update this to not require replication for params or
  # rngs.
  @functools.partial(jax.pmap, axis_name='batch')
  def evaluation_fn(params, images, labels, mask, rng):
    # Ignore the entries with all zero labels for evaluation.
    mask *= labels.max(axis=1)
    label_indices = config.get('label_indices')
    logging.info('mask %s, label_indices %s', mask, label_indices)

    def apply_model(apply_rng):
      # Sample rank-1 parameter values from the variational posteriors.
      sampled_params = sample_gaussian_rank1(
          params, apply_rng, rank1_regex_patterns=rank1_regex_patterns)

      tiled_logits, out = model.apply(
          {'params': flax.core.freeze(sampled_params)}, images, train=False)

      if label_indices:
        tiled_logits = tiled_logits[:, label_indices]

      # Both are [ens_size, batch_size, hidden_size].
      ens_logits = jnp.asarray(jnp.split(tiled_logits, ens_size))
      ens_pre_logits = jnp.asarray(jnp.split(out['pre_logits'], ens_size))
      return ens_logits, ens_pre_logits

    # Vmap over a number of samples from the rank-1 variational distributions.
    # Outputs are [eval_samples, ens_size, ...]. Collapse first two dimensions.
    apply_rngs = jax.random.split(rng, num=config.eval_samples)
    (sampled_ens_logits, sampled_ens_pre_logits) = jax.vmap(apply_model)(
        apply_rngs)
    all_logits = jnp.reshape(sampled_ens_logits,
                             [-1] + list(sampled_ens_logits.shape[2:]))
    all_pre_logits = jnp.reshape(sampled_ens_pre_logits,
                                 [-1] + list(sampled_ens_pre_logits.shape[2:]))
    # Transpose to [batch_size, hidden_size, ens_size*eval_samples]
    all_pre_logits = jnp.transpose(all_pre_logits, axes=[1, 2, 0])

    # TODO(dusenberrymw,zmariet): Clean up and generalize this.
    loss_name = config.get('loss', 'sigmoid_xent')
    if loss_name == 'sigmoid_xent':
      logits = batchensemble_utils.log_average_sigmoid_probs(all_logits)
    else:  # softmax
      logits = batchensemble_utils.log_average_softmax_probs(all_logits)

    losses = getattr(train_utils, loss_name)(
        logits=logits,
        labels=labels[:, :(
            len(label_indices) if label_indices else config.num_classes)],
        reduction=False)
    loss = jax.lax.psum(losses * mask, axis_name='batch')

    top1_idx = jnp.argmax(logits, axis=1)
    top1_correct = jnp.take_along_axis(labels, top1_idx[:, None], axis=1)[:, 0]
    ncorrect = jax.lax.psum(top1_correct * mask, axis_name='batch')
    n = jax.lax.psum(mask, axis_name='batch')

    metric_args = jax.lax.all_gather([logits, labels, all_pre_logits, mask],
                                     axis_name='batch')
    return ncorrect, loss, n, metric_args

  # TODO(dusenberrymw): Update this to not require replication for params or
  # rngs.
  @functools.partial(jax.pmap, axis_name='batch')
  def cifar_10h_evaluation_fn(params, images, labels, mask, rng):
    label_indices = config.get('label_indices')

    def apply_model(apply_rng):
      # Sample rank-1 parameter values from the variational posteriors.
      sampled_params = sample_gaussian_rank1(
          params, apply_rng, rank1_regex_patterns=rank1_regex_patterns)

      tiled_logits, out = model.apply(
          {'params': flax.core.freeze(sampled_params)}, images, train=False)

      if label_indices:
        tiled_logits = tiled_logits[:, label_indices]

      # Both are [ens_size, batch_size, hidden_size].
      ens_logits = jnp.asarray(jnp.split(tiled_logits, ens_size))
      ens_pre_logits = jnp.asarray(jnp.split(out['pre_logits'], ens_size))
      return ens_logits, ens_pre_logits

    # Vmap over a number of samples from the rank-1 variational distributions.
    # Outputs are [eval_samples, ens_size, ...]. Collapse first two dimensions.
    apply_rngs = jax.random.split(rng, num=config.eval_samples)
    (sampled_ens_logits, sampled_ens_pre_logits) = jax.vmap(apply_model)(
        apply_rngs)
    all_logits = jnp.reshape(sampled_ens_logits,
                             [-1] + list(sampled_ens_logits.shape[2:]))
    all_pre_logits = jnp.reshape(sampled_ens_pre_logits,
                                 [-1] + list(sampled_ens_pre_logits.shape[2:]))
    # Transpose to [batch_size, hidden_size, ens_size*eval_samples]
    all_pre_logits = jnp.transpose(all_pre_logits, axes=[1, 2, 0])

    # TODO(dusenberrymw,zmariet): Clean up and generalize this.
    loss_name = config.get('loss', 'sigmoid_xent')
    if loss_name == 'sigmoid_xent':
      logits = batchensemble_utils.log_average_sigmoid_probs(all_logits)
    else:  # softmax
      logits = batchensemble_utils.log_average_softmax_probs(all_logits)

    losses = getattr(train_utils, loss_name)(
        logits=logits, labels=labels, reduction=False)
    loss = jax.lax.psum(losses, axis_name='batch')

    top1_idx = jnp.argmax(logits, axis=1)
    # Extracts the label at the highest logit index for each image.
    one_hot_labels = jnp.eye(10)[jnp.argmax(labels, axis=1)]

    top1_correct = jnp.take_along_axis(
        one_hot_labels, top1_idx[:, None], axis=1)[:, 0]
    ncorrect = jax.lax.psum(top1_correct, axis_name='batch')
    n = jax.lax.psum(one_hot_labels, axis_name='batch')

    metric_args = jax.lax.all_gather([logits, labels, all_pre_logits, mask],
                                     axis_name='batch')
    return ncorrect, loss, n, metric_args

  # Setup function for computing representation.
  # TODO(dusenberrymw): Update this to not require replication for params or
  # rngs.
  @functools.partial(jax.pmap, axis_name='batch')
  def representation_fn(params, images, labels, mask, rng):

    def apply_model(apply_rng):
      # Sample rank-1 parameter values from the variational posteriors.
      sampled_params = sample_gaussian_rank1(
          params, apply_rng, rank1_regex_patterns=rank1_regex_patterns)

      _, out = model.apply({'params': flax.core.freeze(sampled_params)},
                           images,
                           train=False)

      # Shape: [ens_size, batch_size, hidden_size].
      representation = out[config.fewshot.representation_layer]
      representation = jnp.asarray(jnp.split(representation, ens_size))
      return representation

    # Vmap over a number of samples from the rank-1 variational distributions.
    apply_rngs = jax.random.split(rng, num=config.eval_samples)
    # Output is [eval_samples, ens_size, batch_size, hidden_size]. Convert to
    # [batch_size, hidden_size*eval_samples*ens_size].
    sampled_representations = jax.vmap(apply_model)(apply_rngs)
    all_representations = jnp.transpose(sampled_representations, [2, 3, 0, 1])
    all_representations = jnp.reshape(all_representations,
                                      [all_representations.shape[0], -1])
    all_representations = jax.lax.all_gather(all_representations, 'batch')
    labels = jax.lax.all_gather(labels, 'batch')
    mask = jax.lax.all_gather(mask, 'batch')
    return all_representations, labels, mask

  opt_name = config.get('optim_name')
  write_note(f'Initializing {opt_name} optimizer...')
  opt_def = getattr(flax.optim, opt_name)(**config.get('optim', {}))

  # We jit this, such that the arrays that are created are created on the same
  # device as the input is, in this case the CPU. Else they'd be on device[0].
  opt_cpu = jax.jit(opt_def.create)(params_cpu)

  weight_decay_rules = config.get('weight_decay', []) or []
  rescale_value = 1.
  weight_decay_fn = train_utils.get_weight_decay_fn(
      weight_decay_rules=weight_decay_rules, rescale_value=rescale_value)

  def batch_loss_fn(params, images, labels, rng):
    sample_rng, train_rng = jax.random.split(rng)
    # Sample rank-1 parameter values from the variational posteriors.
    sampled_params = sample_gaussian_rank1(
        params, sample_rng, rank1_regex_patterns=rank1_regex_patterns)
    logits, _ = model.apply({'params': flax.core.freeze(sampled_params)},
                            images,
                            train=True,
                            rngs={'dropout': train_rng})
    labels = jnp.tile(labels, (ens_size, 1))
    loss_fn = getattr(train_utils, config.get('loss', 'sigmoid_xent'))
    nll = jnp.mean(loss_fn(logits=logits, labels=labels))
    kl = gaussian_rank1_kl_divergence(
        params,
        prior_mean=config.prior_mean,
        prior_std=config.prior_std,
        rank1_regex_patterns=rank1_regex_patterns)
    loss = nll + kl
    return loss, dict()

  # TODO(dusenberrymw): Include num_samples for the posterior sampling.
  # TODO(dusenberrymw): Update this to not require replication for params or
  # rngs.
  @functools.partial(jax.pmap, axis_name='batch', donate_argnums=(0, 1))
  def update_fn(opt, rng, lr, images, labels):
    return batchensemble_utils.update_fn_be(
        opt=opt,
        rng=rng,
        lr=lr,
        images=images,
        labels=labels,
        batch_loss_fn=batch_loss_fn,
        weight_decay_fn=weight_decay_fn,
        max_grad_norm_global=config.get('grad_clip_norm', None),
        fast_weight_lr_multiplier=config.get('fast_weight_lr_multiplier', None))

  reinit_params = ('batchensemble_head/bias', 'batchensemble_head/kernel',
                   'batchensemble_head/fast_weight_alpha',
                   'batchensemble_head/fast_weight_gamma')
  if config.get('only_eval', False) or not config.get('reint_head', True):
    reinit_params = []

  rng, train_rng = jax.random.split(rng)
  # TODO(dusenberrymw): Fix the existing handling of keys.
  checkpoint_data = checkpoint_utils.maybe_load_checkpoint(
      train_loop_rngs=train_rng,
      save_checkpoint_path=save_checkpoint_path,
      init_optimizer=opt_cpu,
      init_params=params_cpu,
      init_fixed_model_states=None,
      default_reinit_params=reinit_params,
      config=config)

  # TODO(dusenberrymw): Remove manual replication of keys.
  if isinstance(checkpoint_data.train_loop_rngs, dict):
    assert list(checkpoint_data.train_loop_rngs.keys()) == ['dropout']
    checkpoint_data.train_loop_rngs = checkpoint_data.train_loop_rngs['dropout']
  train_loop_rngs = checkpoint_data.train_loop_rngs
  opt_cpu = checkpoint_data.optimizer

  accumulated_train_time = checkpoint_data.accumulated_train_time

  write_note('Kicking off misc stuff...')
  first_step = int(opt_cpu.state.step)  # Might be a DeviceArray type.
  logging.info('first_step = %s', first_step)
  if first_step == 0 and jax.process_index() == 0:
    writer.write_hparams(dict(config))

  chrono = train_utils.Chrono(first_step, total_steps, batch_size,
                              accumulated_train_time)

  # Note: switch to ProfileAllHosts() if you need to profile all hosts.
  # (Xprof data become much larger and take longer to load for analysis)
  profiler = periodic_actions.Profile(
      # Create profile after every restart to analyze pre-emption related
      # problems and assure we get similar performance in every run.
      logdir=output_dir,
      first_profile=first_step + 10)

  # TODO(dusenberrymw): Remove manual replication by updating pmap axes.
  write_note(f'Replicating...\n{chrono.note}')
  opt_repl = flax.jax_utils.replicate(opt_cpu)

  write_note(f'Initializing few-shotters...\n{chrono.note}')
  fewshotter = None
  if 'fewshot' in config and fewshot is not None:
    fewshotter = fewshot.FewShotEvaluator(
        representation_fn, config.fewshot,
        config.fewshot.get('batch_size') or batch_size_eval)

  checkpoint_writer = None

  lr_fn = train_utils.create_learning_rate_schedule(total_steps,
                                                    **config.get('lr', {}))

  # Prefetch all iterators, starting at the current first step.
  if first_step > 0:
    write_note('Advancing the dataset after resuming from a checkpoint...')
    if not config.get('disable_preemption_reproducibility', False):
      # TODO(dusenberrymw): Look into checkpointing dataset state instead.
      train_ds = train_ds.skip(first_step)

  # TODO(dusenberrymw): According to flax docs, prefetching shouldn't be
  # necessary for TPUs.
  train_iter = input_utils.start_input_pipeline(
      train_ds, config.get('prefetch_to_device', 1))
  lr_iter = train_utils.prefetch_scalar(
      map(lr_fn, range(first_step, total_steps)),
      config.get('prefetch_to_device', 1))

  # Note: we return the train loss, val loss, and fewshot best l2s for use in
  # reproducibility unit tests.
  train_loss = -jnp.inf
  val_loss = {val_name: -jnp.inf for val_name, _ in val_ds_splits.items()}
  fewshot_results = {'dummy': {(0, 1): -jnp.inf}}

  write_note(f'First step compilations...\n{chrono.note}')
  for step in range(first_step + 1, total_steps + 1):
    with jax.profiler.StepTraceAnnotation('train_step', step_num=step):
      train_batch = next(train_iter)
      lr_repl = next(lr_iter)
      if not config.get('only_eval', False):
        opt_repl, train_loop_rngs, extra_measurements = update_fn(
            opt_repl,
            train_loop_rngs,
            lr_repl,
            train_batch['image'],
            train_batch['labels'])

    if jax.process_index() == 0:
      profiler(step)

    # Checkpoint saving
    if not config.get('only_eval', False) and train_utils.itstime(
        step, config.get('checkpoint_steps'), total_steps, process=0):
      write_note('Checkpointing...')
      chrono.pause()
      train_utils.checkpointing_timeout(checkpoint_writer,
                                        config.get('checkpoint_timeout', 1))
      accumulated_train_time = chrono.accum_train_time
      # We need to transfer the weights over now or else we risk keeping them
      # alive while they'll be updated in a future step, creating hard to debug
      # memory errors (see b/160593526). Also, takes device 0's params only.
      opt_cpu = jax.tree_util.tree_map(lambda x: np.array(x[0]), opt_repl)

      # Check whether we want to keep a copy of the current checkpoint.
      copy_step = None
      if train_utils.itstime(step, config.get('keep_checkpoint_steps'),
                             total_steps):
        write_note('Keeping a checkpoint copy...')
        copy_step = step

      # Checkpoint should be a nested dictionary or FLAX datataclasses from
      # `flax.struct`. Both can be present in a checkpoint.
      checkpoint_data = checkpoint_utils.CheckpointData(
          train_loop_rngs=train_loop_rngs,
          optimizer=opt_cpu,
          accumulated_train_time=accumulated_train_time)
      checkpoint_writer = pool.apply_async(
          checkpoint_utils.checkpoint_trained_model,
          (checkpoint_data, save_checkpoint_path, copy_step))
      chrono.resume()

    # Report training progress
    if not config.get('only_eval', False) and train_utils.itstime(
        step, config.log_training_steps, total_steps, process=0):
      write_note('Reporting training progress...')
      timing_measurements, note = chrono.tick(step)
      write_note(note)
      train_measurements = {}
      train_measurements.update(flax.jax_utils.unreplicate(extra_measurements))
      train_measurements.update(timing_measurements)
      writer.write_scalars(step, train_measurements)
      # Keep to return for reproducibility tests.
      train_loss = train_measurements['training_loss']

    # Report validation performance
    if config.get('only_eval', False) or train_utils.itstime(
        step, config.log_eval_steps, total_steps):
      write_note('Evaluating on the validation set...')
      chrono.pause()
      for val_name, val_ds in val_ds_splits.items():
        # Sets up evaluation metrics.
        ece_num_bins = config.get('ece_num_bins', 15)
        auc_num_bins = config.get('auc_num_bins', 1000)
        ece = rm.metrics.ExpectedCalibrationError(num_bins=ece_num_bins)
        calib_auc = rm.metrics.CalibrationAUC(correct_pred_as_pos_label=False)
        oc_auc_0_5 = rm.metrics.OracleCollaborativeAUC(
            oracle_fraction=0.005, num_bins=auc_num_bins)
        oc_auc_1 = rm.metrics.OracleCollaborativeAUC(
            oracle_fraction=0.01, num_bins=auc_num_bins)
        oc_auc_2 = rm.metrics.OracleCollaborativeAUC(
            oracle_fraction=0.02, num_bins=auc_num_bins)
        oc_auc_5 = rm.metrics.OracleCollaborativeAUC(
            oracle_fraction=0.05, num_bins=auc_num_bins)
        label_diversity = tf.keras.metrics.Mean()
        sample_diversity = tf.keras.metrics.Mean()
        ged = tf.keras.metrics.Mean()

        # Runs evaluation loop.
        val_iter = input_utils.start_input_pipeline(
            val_ds, config.get('prefetch_to_device', 1))
        ncorrect, loss, nseen = 0, 0, 0
        for batch in val_iter:
          if val_name == 'cifar_10h':
            batch_ncorrect, batch_losses, batch_n, batch_metric_args = (
                cifar_10h_evaluation_fn(
                    opt_repl.target,
                    batch['image'],
                    batch['labels'],
                    batch['mask'],
                    rng=train_loop_rngs))
          else:
            batch_ncorrect, batch_losses, batch_n, batch_metric_args = (
                evaluation_fn(
                    opt_repl.target,
                    batch['image'],
                    batch['labels'],
                    batch['mask'],
                    rng=train_loop_rngs))
          # All results are a replicated array shaped as follows:
          # (local_devices, per_device_batch_size, elem_shape...)
          # with each local device's entry being identical as they got psum'd.
          # So let's just take the first one to the host as numpy.
          ncorrect += np.sum(np.array(batch_ncorrect[0]))
          loss += np.sum(np.array(batch_losses[0]))
          nseen += np.sum(np.array(batch_n[0]))
          if config.get('loss', 'sigmoid_xent') != 'sigmoid_xent':
            # Here we parse batch_metric_args to compute uncertainty metrics.
            # (e.g., ECE or Calibration AUC).
            logits, labels, _, masks = batch_metric_args
            masks = np.array(masks[0], dtype=np.bool)
            logits = np.array(logits[0])
            probs = jax.nn.softmax(logits)
            # From one-hot to integer labels, as required by ECE.
            int_labels = np.argmax(np.array(labels[0]), axis=-1)
            int_preds = np.argmax(logits, axis=-1)
            confidence = np.max(probs, axis=-1)
            for p, c, l, d, m, label in zip(probs, confidence, int_labels,
                                            int_preds, masks, labels[0]):
              ece.add_batch(p[m, :], label=l[m])
              calib_auc.add_batch(d[m], label=l[m], confidence=c[m])
              # TODO(jereliu): Extend to support soft multi-class probabilities.
              oc_auc_0_5.add_batch(d[m], label=l[m], custom_binning_score=c[m])
              oc_auc_1.add_batch(d[m], label=l[m], custom_binning_score=c[m])
              oc_auc_2.add_batch(d[m], label=l[m], custom_binning_score=c[m])
              oc_auc_5.add_batch(d[m], label=l[m], custom_binning_score=c[m])

              if val_name == 'cifar_10h' or val_name == 'imagenet_real':
                num_classes = config.num_classes
                if config.get('label_indices'):
                  num_classes = len(config.get('label_indices'))
                batch_label_diversity, batch_sample_diversity, batch_ged = data_uncertainty_utils.generalized_energy_distance(
                    label[m], p[m, :], num_classes)
                label_diversity.update_state(batch_label_diversity)
                sample_diversity.update_state(batch_sample_diversity)
                ged.update_state(batch_ged)

        val_loss[val_name] = loss / nseen  # Keep for reproducibility tests.
        val_measurements = {
            f'{val_name}_prec@1': ncorrect / nseen,
            f'{val_name}_loss': val_loss[val_name],
        }
        if config.get('loss', 'sigmoid_xent') != 'sigmoid_xent':
          val_measurements[f'{val_name}_ece'] = ece.result()['ece']
          val_measurements[f'{val_name}_calib_auc'] = calib_auc.result(
          )['calibration_auc']
          val_measurements[f'{val_name}_oc_auc_0.5%'] = oc_auc_0_5.result(
          )['collaborative_auc']
          val_measurements[f'{val_name}_oc_auc_1%'] = oc_auc_1.result(
          )['collaborative_auc']
          val_measurements[f'{val_name}_oc_auc_2%'] = oc_auc_2.result(
          )['collaborative_auc']
          val_measurements[f'{val_name}_oc_auc_5%'] = oc_auc_5.result(
          )['collaborative_auc']
        writer.write_scalars(step, val_measurements)

        if val_name == 'cifar_10h' or val_name == 'imagenet_real':
          cifar_10h_measurements = {
              f'{val_name}_label_diversity': label_diversity.result(),
              f'{val_name}_sample_diversity': sample_diversity.result(),
              f'{val_name}_ged': ged.result(),
          }
          writer.write_scalars(step, cifar_10h_measurements)

      # OOD eval
      # Entries in the ood_ds dict include:
      # (ind_dataset, ood_dataset1, ood_dataset2, ...).
      # OOD metrics are computed using ind_dataset paired with each of the
      # ood_dataset. When Mahalanobis distance method is applied, train_ind_ds
      # is also included in the ood_ds.
      if ood_ds and config.ood_methods:
        ood_measurements = ood_utils.eval_ood_metrics(
            ood_ds,
            ood_ds_names,
            config.ood_methods,
            functools.partial(evaluation_fn, rng=train_loop_rngs),
            opt_repl.target,
            n_prefetch=config.get('prefetch_to_device', 1))
        writer.write_scalars(step, ood_measurements)
      chrono.resume()

      # Perform subpopulation shift evaluation only if flag is provided.
      if config.get('subpopl_cifar_data_file'):
        subpopl_measurements = subpopl_utils.eval_subpopl_metrics(
            subpopl_val_ds_splits,
            functools.partial(evaluation_fn, rng=train_loop_rngs),
            opt_repl.target,
            n_prefetch=config.get('prefetch_to_device', 1))
        writer.write_scalars(step, scalars=subpopl_measurements)

    if 'fewshot' in config and fewshotter is not None:
      # Compute few-shot on-the-fly evaluation.
      if config.get('only_eval', False) or train_utils.itstime(
          step, config.fewshot.log_steps, total_steps):
        chrono.pause()
        write_note(f'Few-shot evaluation...\n{chrono.note}')
        # Keep `results` to return for reproducibility tests.
        fewshot_results, best_l2 = fewshotter.run_all(
            opt_repl.target, config.fewshot.datasets, rng=train_loop_rngs)

        # TODO(dusenberrymw): Remove this once fewshot.py is updated.
        def make_writer_measure_fn(step):

          def writer_measure(name, value):
            writer.write_scalars(step, {name: value})

          return writer_measure

        fewshotter.walk_results(
            make_writer_measure_fn(step), fewshot_results, best_l2)
        chrono.resume()

    if config.get('only_eval', False):
      break
    elif config.get('testing_failure_step'):
      # Break early to simulate infra failures in test cases.
      if config.testing_failure_step == step:
        break

  write_note(f'Done!\n{chrono.note}')
  pool.close()
  pool.join()
  writer.close()

  # Return final training loss, validation loss, and fewshot results for
  # reproducibility test cases.
  return train_loss, val_loss, fewshot_results


if __name__ == '__main__':
  # Adds jax flags to the program.
  jax.config.parse_flags_with_absl()

  def _main(argv):
    del argv
    config = FLAGS.config
    output_dir = FLAGS.output_dir
    main(config, output_dir)

  app.run(_main)  # Ignore the returned values from `main`.

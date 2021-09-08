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

"""ViT-SNGP on JFT-300M."""

from functools import partial  # pylint: disable=g-importing-member so standard
import itertools
import multiprocessing
import numbers
import os

from absl import app
from absl import flags
from absl import logging
from clu import parameter_overview
from clu import periodic_actions
import flax
import flax.jax_utils as flax_utils
import flax.traverse_util as trav_utils

import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
import robustness_metrics as rm

import tensorflow as tf
from tensorflow.io import gfile
import uncertainty_baselines as ub
import checkpoint_utils  # local file import
import cifar10h_utils  # local file import
import ood_utils  # local file import

# TODO(dusenberrymw): Open-source remaining imports.
fewshot = None
input_pipeline = None
u = None
pp_builder = None
xm = None
xm_api = None


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

# Adds jax flags to the program.
jax.config.parse_flags_with_absl()


# Utility functions.
def accumulate_gradient_with_states(
    loss_and_grad_fn,
    params,
    states,  # Allows for states.
    images,
    labels,
    accum_steps):
  """Improved version of `u.accumulate_gradient()` that allows for states."""
  # This function handles the `loss_and_grad_fn` function which takes a state
  # argument and returns ((losses, states), grads).
  if accum_steps and accum_steps > 1:
    assert images.shape[0] % accum_steps == 0, (
        f'Bad accum_steps {accum_steps} for batch size {images.shape[0]}')
    step_size = images.shape[0] // accum_steps

    # Run the first step.
    (l, s), g = loss_and_grad_fn(params, states, images[:step_size],
                                 labels[:step_size])

    # Run the rest of the steps.
    def acc_grad_and_loss(i, l_s_g):
      # Extract data for current step.
      imgs = jax.lax.dynamic_slice(images, (i * step_size, 0, 0, 0),
                                   (step_size,) + images.shape[1:])
      lbls = jax.lax.dynamic_slice(labels, (i * step_size, 0),
                                   (step_size, labels.shape[1]))
      # Update state and accumulate gradient.
      l, s, g = l_s_g
      (li, si), gi = loss_and_grad_fn(params, s, imgs, lbls)
      return (l + li, si, jax.tree_multimap(lambda x, y: x + y, g, gi))

    l, s, g = jax.lax.fori_loop(1, accum_steps, acc_grad_and_loss, (l, s, g))
    l, g = jax.tree_map(lambda x: x / accum_steps, (l, g))
    return (l, s), g
  else:
    return loss_and_grad_fn(params, states, images, labels)


def get_gp_kwargs(gp_config):
  """Extract keyword argument parameters for the Gaussian process layer."""
  covmat_momentum = gp_config.get('covmat_momentum', 0.999)

  # Extracts model parameter.
  logging.info('gp_config.covmat_momentum = %s', covmat_momentum)
  covmat_momentum = None if covmat_momentum < 0. else covmat_momentum
  covmat_kwargs = dict(momentum=covmat_momentum)

  # Assembles into kwargs dictionary.
  gp_layer_kwargs = dict(covmat_kwargs=covmat_kwargs)

  return gp_layer_kwargs


def pretrained_sngp_load(init_params, init_file, model_params):
  """Load model parameters from checkpoint and align that with ViT-SNGP."""

  def _flatten_dict(params):
    return {'/'.join(k): v for k, v in trav_utils.flatten_dict(params).items()}

  def _unflatten_dict(flat_params):
    tuple_to_value = {tuple(k.split('/')): v for k, v in flat_params.items()}
    return trav_utils.unflatten_dict(tuple_to_value)

  # Restores parameters from the checkpoint.
  restored_params = checkpoint_utils.load_from_pretrained_checkpoint(
      init_params, init_file, model_params.representation_size,
      model_params.classifier,
      model_params.get('reinit_params',
                       ('head/output_layer/kernel', 'head/output_layer/bias',
                        'head/kernel', 'head/bias')))

  # Align restored parameter dict (restored_params) with model parameters
  # (init_params) by adding in missing parameters and removing extra parameters.
  # This is needed for ViT-GP to use pre-trained embeddings from
  # other models.
  restored_flat = _flatten_dict(restored_params)
  expected_flat = _flatten_dict(init_params)
  missing_keys = expected_flat.keys() - restored_flat.keys()
  extra_keys = restored_flat.keys() - expected_flat.keys()

  logging.info(
      'Restored params from checkpoint: %s.\n'
      'Expected params from code: %s.', restored_flat.keys(),
      expected_flat.keys())

  # Remove extra parameters.
  for k in extra_keys:
    del restored_flat[k]

  # Add missing parameters using initialized values.
  for k in missing_keys:
    restored_flat[k] = expected_flat[k]

  logging.info(
      'Added Missing params from checkpoint: %s.\n'
      'Removed Extra params in checkpoint: %s.\n', missing_keys, extra_keys)

  return _unflatten_dict(restored_flat)


def main(argv):
  del argv

  config = FLAGS.config
  output_dir = FLAGS.output_dir

  if config.get('dataset_dir'):
    logging.info('data_dir=%s', config.dataset_dir)
  logging.info('Output dir: %s', output_dir)

  save_checkpoint_path = None
  if config.get('checkpoint_steps'):
    gfile.makedirs(output_dir)
    save_checkpoint_path = os.path.join(output_dir, 'checkpoint.npz')

  # The pool is used to perform misc operations such as logging in async way.
  pool = multiprocessing.pool.ThreadPool()

  # TODO(dusenberrymw): Also add function-level seeds in the tf.data input
  # pipeline once that code is open-sourced.
  seed = config.get('seed', 0)
  rng = jax.random.PRNGKey(seed)
  tf.random.set_seed(seed)

  xm_xp = None
  xm_wu = None
  def write_note(note):
    if jax.host_id() == 0:
      logging.info('NOTE: %s', note)
  write_note('Initializing...')

  fillin = lambda *_: None
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

  local_batch_size = batch_size // jax.host_count()
  local_batch_size_eval = batch_size_eval // jax.host_count()
  logging.info(
      'Global batch size %d on %d hosts results in %d local batch size. '
      'With %d dev per host (%d dev total), that is a %d per-device batch size.',
      batch_size, jax.host_count(), local_batch_size, jax.local_device_count(),
      jax.device_count(), local_batch_size // jax.local_device_count())

  write_note('Initializing train dataset...')
  # TODO(dusenberrymw): Pass in seed for function-level seeds once open-sourced.
  train_ds = input_pipeline.get_data(
      dataset=config.dataset,
      split=config.train_split,
      data_dir=fillin(config.get('dataset_dir')),
      batch_size=local_batch_size,
      preprocess_fn=pp_builder.get_preprocess_fn(config.pp_train),
      shuffle_buffer_size=config.shuffle_buffer_size,
      prefetch=config.get('prefetch_to_host', 2),
      cache=False)
  logging.info('image_size = %s', train_ds.element_spec['image'].shape[1:])

  # Start prefetching already.
  train_iter = u.start_input_pipeline(
      train_ds, config.get('prefetch_to_device', 1), pad=local_batch_size)
  # We always pad to local_batch_size_eval even when less would be enough in
  # order to minimize memory fragmentation.

  write_note('Initializing val dataset(s)...')
  def _get_val_split(dataset, split, pp_eval, data_dir=None):
    # We do ceil rounding such that we include the last incomplete batch.
    nval_img = input_pipeline.get_num_examples(
        dataset, split, data_dir=fillin(data_dir))
    val_steps = int(np.ceil(nval_img / batch_size_eval))
    logging.info('Running validation for %d steps for %s, %s', val_steps,
                 dataset, split)

    val_it = input_pipeline.get_data(
        dataset=dataset,
        split=split,
        data_dir=fillin(data_dir),
        batch_size=local_batch_size_eval,
        preprocess_fn=pp_builder.get_preprocess_fn(pp_eval),
        cache=config.get('val_cache', 'batched'),
        repeat_after_batching=True,
        prefetch=0,  # Save memory since we cache.
        drop_remainder=False,
        shuffle_files=False)
    val_it = u.start_input_pipeline(
        val_it, config.get('prefetch_to_device', 1), pad=local_batch_size_eval)

    return (val_it, val_steps)

  if isinstance(config.val_split, str):
    val_iter_splits = {
        'val':
            _get_val_split(config.dataset, config.val_split, config.pp_eval,
                           config.get('dataset_dir'))
    }
  else:
    val_iter_splits = {t[0]: _get_val_split(*t[1:]) for t in config.val_split}

  if config.get('eval_on_cifar_10h'):
    val_steps = int(np.ceil(10000 / batch_size_eval))

    cifar10h_dataset = cifar10h_utils.load_ds()

    val_ds_cifar10h = input_pipeline.make_pipeline(
        data=cifar10h_dataset,
        batch_size=local_batch_size_eval,
        preprocess_fn=pp_builder.get_preprocess_fn(config.pp_eval_cifar_10h),
        cache=config.get('val_cache', 'batched'),
        repeats=None,
        repeat_after_batching=True,
        prefetch=0,
        drop_remainder=False,
        shuffle_buffer_size=None,
        ignore_errors=False,
        filter_fn=None)
    val_iter_cifar10h = u.start_input_pipeline(
        val_ds_cifar10h,
        config.get('prefetch_to_device', 1),
        pad=local_batch_size_eval)

    val_iter_splits['cifar_10h'] = (val_iter_cifar10h, val_steps)

  ood_ds = {}
  if config.get('ood_dataset'):
    logging.info('loading OOD dataset = %s', config.get('ood_dataset'))
    if isinstance(config.train_split, str):
      # Adds training set for fitting class conditional Gaussian for
      # Mahalanoabis distance method.
      ood_ds.update({
          'train_maha':
              _get_val_split(config.dataset, config.train_split, config.pp_eval,
                             config.get('data_dir'))
      })
    else:
      raise NotImplementedError(
          'Only string type of train_split is supported for OOD evaluation!'
          'Got train_split=%s!' % str(config.train_split))
    if isinstance(config.ood_split, str):
      ood_ds.update({
          'ind':
              _get_val_split(config.dataset, config.ood_split, config.pp_eval,
                             config.get('data_dir')),
          'ood':
              _get_val_split(config.ood_dataset, config.ood_split,
                             config.pp_eval, config.get('data_dir')),
      })
    else:
      raise NotImplementedError(
          'Only string type of ood_split is supported for OOD evaluation!'
          'Got ood_split=%s!' % str(config.ood_split))

  ntrain_img = input_pipeline.get_num_examples(
      config.dataset,
      config.train_split,
      data_dir=fillin(config.get('dataset_dir')))
  steps_per_epoch = ntrain_img / batch_size

  if config.get('num_epochs'):
    total_steps = int(config.num_epochs * steps_per_epoch)
    assert not config.get('total_steps'), 'Set either num_epochs or total_steps'
  else:
    total_steps = config.total_steps

  logging.info(
      'Running for %d steps, that means %f epochs and %f steps per epoch',
      total_steps, total_steps * batch_size / ntrain_img, steps_per_epoch)
  mw = u.BigVisionMetricWriter(xm_xp.id, xm_wu.id, steps_per_epoch)

  write_note('Initializing model...')
  logging.info('config.model = %s', config.get('model'))

  # Specify Gaussian process layer configs.
  use_gp_layer = True
  gp_config = config.get('gp_layer', {})
  gp_layer_kwargs = get_gp_kwargs(gp_config)

  # Process ViT backbone model configs.
  vit_kwargs = config.get('model')

  if vit_kwargs.get('reinit', None):
    # Remove parameters not related to model architecture.
    logging.info('Non-architecture parameter "reinit" found in vit_kwargs. '
                 'It will be removed before passing to the model.')
    vit_kwargs = vit_kwargs.copy_and_resolve_references()
    with vit_kwargs.unlocked():
      del vit_kwargs.reinit

  model = ub.models.vision_transformer_gp(
      num_classes=config.num_classes,
      use_gp_layer=use_gp_layer,
      vit_kwargs=vit_kwargs,
      gp_layer_kwargs=gp_layer_kwargs)

  # We want all parameters to be created in host RAM, not on any device, they'll
  # be sent there later as needed, otherwise we already encountered two
  # situations where we allocate them twice.
  @partial(jax.jit, backend='cpu')
  def init(rng):
    image_size = tuple(train_ds.element_spec['image'].shape[1:])
    dummy_input = jnp.zeros((local_batch_size,) + image_size, jnp.float32)
    variables = model.init(rng, dummy_input, train=False)
    # Split model parameters into trainable and untrainable collections.
    states, params = variables.pop('params')
    del variables

    # Set bias in the head to a low value, such that loss is small initially.
    params = flax.core.unfreeze(params)
    if use_gp_layer:
      # Modify the head parameter in the GP head.
      params['head']['output_layer']['bias'] = jnp.full_like(
          params['head']['output_layer']['bias'],
          config.get('init_head_bias', 0))
    else:
      params['vit_backbone']['head']['bias'] = jnp.full_like(
          params['vit_backbone']['head']['bias'],
          config.get('init_head_bias', 0))

    return params, states

  rng, rng_init = jax.random.split(rng)
  params_cpu, states_cpu = init(rng_init)

  if jax.host_id() == 0:
    num_params = sum(p.size for p in jax.tree_flatten(params_cpu)[0])
    parameter_overview.log_parameter_overview(params_cpu)
    mw.measure('num_params', num_params)

  @partial(jax.pmap, axis_name='batch')
  def evaluation_fn(params, states, images, labels, mask):
    # Ignore the entries with all zero labels for evaluation.
    mask *= labels.max(axis=1)
    variable_dict = {'params': flax.core.freeze(params), **states}
    logits, out = model.apply(
        variable_dict,
        images,
        train=False,
        mean_field_factor=gp_config.get('mean_field_factor', -1.))

    losses = getattr(u, config.get('loss', 'sigmoid_xent'))(
        logits=logits, labels=labels, reduction=False)
    loss = jax.lax.psum(losses * mask, axis_name='batch')

    top1_idx = jnp.argmax(logits, axis=1)
    # Extracts the label at the highest logit index for each image.
    top1_correct = jnp.take_along_axis(labels, top1_idx[:, None], axis=1)[:, 0]
    ncorrect = jax.lax.psum(top1_correct * mask, axis_name='batch')
    n = jax.lax.psum(mask, axis_name='batch')

    metric_args = jax.lax.all_gather([logits, labels, out['pre_logits'], mask],
                                     axis_name='batch')
    return ncorrect, loss, n, metric_args

  @partial(jax.pmap, axis_name='batch')
  def cifar_10h_evaluation_fn(params, states, images, labels, mask):
    variable_dict = {'params': flax.core.freeze(params), **states}
    logits, out = model.apply(
        variable_dict,
        images,
        train=False,
        mean_field_factor=gp_config.get('mean_field_factor', -1.))

    losses = getattr(u, config.get('loss', 'softmax_xent'))(
        logits=logits, labels=labels, reduction=False)
    loss = jax.lax.psum(losses, axis_name='batch')

    top1_idx = jnp.argmax(logits, axis=1)
    # Extracts the label at the highest logit index for each image.
    one_hot_labels = jnp.eye(10)[jnp.argmax(labels, axis=1)]

    top1_correct = jnp.take_along_axis(
        one_hot_labels, top1_idx[:, None], axis=1)[:, 0]
    ncorrect = jax.lax.psum(top1_correct, axis_name='batch')
    n = jax.lax.psum(one_hot_labels, axis_name='batch')

    metric_args = jax.lax.all_gather([logits, labels, out['pre_logits'], mask],
                                     axis_name='batch')
    return ncorrect, loss, n, metric_args

  # Setup function for computing representation.
  @partial(jax.pmap, axis_name='batch')
  def representation_fn(params, images, labels, mask, states):
    variable_dict = {'params': flax.core.freeze(params), **states}
    _, outputs = model.apply(
        variable_dict,
        images,
        train=False,
        mean_field_factor=gp_config.get('mean_field_factor', -1.))
    representation = outputs[config.fewshot.representation_layer]
    representation = jax.lax.all_gather(representation, 'batch')
    labels = jax.lax.all_gather(labels, 'batch')
    mask = jax.lax.all_gather(mask, 'batch')
    return representation, labels, mask

  # Load the optimizer from flax.
  opt_name = config.get('optim_name')
  write_note(f'Initializing {opt_name} optimizer...')
  opt_def = getattr(flax.optim, opt_name)(**config.get('optim', {}))

  # We jit this, such that the arrays that are created are created on the same
  # device as the input is, in this case the CPU. Else they'd be on device[0].
  opt_cpu = jax.jit(opt_def.create)(params_cpu)

  @partial(jax.pmap, axis_name='batch', donate_argnums=(0,))
  def update_fn(opt, states, lr, reset_covmat, images, labels, rng):
    """Update step."""
    measurements = {}

    if config.get('mixup') and config.mixup.p:
      rng, (images, labels), _ = u.mixup(rng, images, labels, **config.mixup)

    # Get device-specific loss rng.
    rng, rng_model = jax.random.split(rng, 2)
    rng_model_local = jax.random.fold_in(rng_model, jax.lax.axis_index('batch'))

    def loss_fn(params, states, images, labels):
      # Specify mutable collection to update untrainable GP parameters.
      variable_dict = {'params': flax.core.freeze(params), **states}
      model_results, updated_states = model.apply(
          variable_dict,
          images,
          train=True,
          rngs={'dropout': rng_model_local},
          mutable=list(states.keys()),
          mean_field_factor=gp_config.get('mean_field_factor', -1.))

      logits, _ = model_results
      loss = getattr(u, config.get('loss', 'sigmoid_xent'))(
          logits=logits, labels=labels)
      return loss, updated_states

    # Performs exact covariance update (i.e., reset precision matrix resetting
    # at begining of new epoch) if covmat_momentum is a null value.
    if gp_config.get('covmat_momentum', -1.) < 0:
      # Resets precision matrix to Identity * ridge_penalty if at the begining
      # of a new epoch. This should be done before accumulate gradient.
      ridge_penalty = gp_config.get('ridge_penalty', 1.)
      prec_mat_old = states['laplace_covariance']['head']['covmat_layer'][
          'precision_matrix']
      prec_mat_new = (
          (1. - reset_covmat) * prec_mat_old +
          reset_covmat * jnp.eye(prec_mat_old.shape[0]) * ridge_penalty)

      states = flax.core.unfreeze(states)
      states['laplace_covariance']['head']['covmat_layer'][
          'precision_matrix'] = prec_mat_new
      states = flax.core.freeze(states)

    # Implementation considerations compared and summarized at
    # https://docs.google.com/document/d/1g3kMEvqu1DOawaflKNyUsIoQ4yIVEoyE5ZlIPkIl4Lc/edit?hl=en#
    (l, s), g = accumulate_gradient_with_states(
        jax.value_and_grad(loss_fn, has_aux=True), opt.target, states, images,
        labels, config.get('grad_accum_steps'))
    l, g = jax.lax.pmean((l, g), axis_name='batch')

    # Log the gradient norm only if we need to compute it anyways (clipping)
    # or if we don't use grad_accum_steps, as they interact badly.
    do_grad_clip = config.get('grad_clip_norm', -1.) > 0.
    if config.get('grad_accum_steps', 1) == 1 or do_grad_clip:
      grads, _ = jax.tree_flatten(g)
      l2_g = jnp.sqrt(sum([jnp.vdot(p, p) for p in grads]))
      measurements['l2_grads'] = l2_g

    # Optionally resize the global gradient to a maximum norm. We found this
    # useful in some cases across optimizers, hence it's in the main loop.
    if do_grad_clip:
      g_factor = jnp.minimum(1.0, config.grad_clip_norm / l2_g)
      g = jax.tree_map(lambda p: g_factor * p, g)
    opt = opt.apply_gradient(g, learning_rate=lr)

    decay_rules = config.get('weight_decay', []) or []
    if isinstance(decay_rules, numbers.Number):
      decay_rules = [('.*kernel.*', decay_rules)]
    sched_m = lr/config.lr.base if config.get('weight_decay_decouple') else lr
    def decay_fn(v, wd):
      return (1.0 - sched_m * wd) * v
    opt = opt.replace(target=u.tree_map_with_regex(
        decay_fn, opt.target, decay_rules, name='weight decay'))

    params, _ = jax.tree_flatten(opt.target)
    measurements['l2_params'] = jnp.sqrt(sum([jnp.vdot(p, p) for p in params]))

    return opt, s, l, rng, measurements

  # Other things besides optimizer state to be stored.
  rng, rng_loop = jax.random.split(rng, 2)
  rngs_loop = flax_utils.replicate(rng_loop)
  checkpoint_extra = dict(accum_train_time=0.0, rngs_loop=rngs_loop)

  # Decide how to initialize training. The order is important.
  # 1. Always resumes from the existing checkpoint, e.g. resumes a finetune job.
  # 2. Resume from a previous checkpoint, e.g. start a cooldown training job.
  # 3. Initialize model from something, e,g, start a fine-tuning job.
  # 4. Train from scratch.
  resume_checkpoint_path = None
  if save_checkpoint_path and gfile.exists(save_checkpoint_path):
    resume_checkpoint_path = save_checkpoint_path
  elif config.get('resume'):
    resume_checkpoint_path = fillin(config.resume)
  if resume_checkpoint_path:
    write_note('Resume training from checkpoint...')
    checkpoint_tree = {
        'opt': opt_cpu,
        'states': states_cpu,
        'extra': checkpoint_extra
    }
    checkpoint = checkpoint_utils.load_checkpoint(checkpoint_tree,
                                                  resume_checkpoint_path)
    opt_cpu, states_cpu, checkpoint_extra = (checkpoint['opt'],
                                             checkpoint['states'],
                                             checkpoint['extra'])
  elif config.get('model_init'):
    # Load trainable parameters from the checkpoint.
    # This does not cause issue for SNGP since all non-trainable parameters
    # (random feature, precision matrix, etc) are last-layer parameters that
    # should be re-trained during fine-tuning.
    write_note(f'Initialize trainable parameters from {config.model_init}...')
    loaded = pretrained_sngp_load(params_cpu, config.model_init,
                                  config.get('model'))
    opt_cpu = opt_cpu.replace(target=loaded)
    if jax.host_id() == 0:
      logging.info('Restored parameter overview:')
      parameter_overview.log_parameter_overview(loaded)

  write_note('Kicking off misc stuff...')
  first_step = int(opt_cpu.state.step)  # Might be a DeviceArray type.
  chrono = u.Chrono(first_step, total_steps, batch_size,
                    checkpoint_extra['accum_train_time'])
  # Note: switch to ProfileAllHosts() if you need to profile all hosts.
  # (Xprof data become much larger and take longer to load for analysis)
  profiler = periodic_actions.Profile(
      # Create profile after every restart to analyze pre-emption related
      # problems and assure we get similar performance in every run.
      logdir=output_dir, first_profile=first_step + 10)

  # Prepare the learning-rate and pre-fetch it to device to avoid delays.
  lr_fn = u.create_learning_rate_schedule(
      batch_size, total_steps, steps_per_epoch, **config.get('lr', {}))
  # TODO(dusenberrymw): According to flax docs, prefetching shouldn't be
  # necessary for TPUs.
  lr_iter = u.prefetch_scalar(
      map(lr_fn, range(total_steps)), config.get('prefetch_to_device', 1))

  # Prepare the precision matrix resetting schedule, and pre-fetch it to device.
  reset_covmat_fn = lambda step: float(step % steps_per_epoch == 0)
  reset_covmat_iter = u.prefetch_scalar(
      map(reset_covmat_fn, range(first_step, total_steps)),
      nprefetch=config.get('prefetch_to_device', 1))

  write_note(f'Replicating...\n{chrono.note}')
  opt_repl = flax_utils.replicate(opt_cpu)
  states_repl = flax_utils.replicate(states_cpu)

  write_note(f'Initializing few-shotters...\n{chrono.note}')
  if 'fewshot' in config:
    fewshotter = fewshot.FewShotEvaluator(
        representation_fn, config.fewshot,
        config.fewshot.get('batch_size') or batch_size_eval)

  rngs_loop = checkpoint_extra['rngs_loop']
  checkpoint_writer = None

  # Note: we return the train loss, val loss, and fewshot best l2s for use in
  # reproducibility unit tests.
  train_loss = -jnp.inf
  val_loss = -jnp.inf
  results = {'dummy': {(0, 1): -jnp.inf}}

  write_note(f'First step compilations...\n{chrono.note}')
  logging.info('first_step = %s', first_step)
  # Advance the iterators if we are restarting from an earlier checkpoint.
  # TODO(dusenberrymw): Look into checkpointing dataset state instead.
  if first_step > 0:
    write_note('Advancing iterators after resuming from a checkpoint...')
    lr_iter = itertools.islice(lr_iter, first_step, None)
    train_iter = itertools.islice(train_iter, first_step, None)
    # NOTE: Validation eval is only run on certain steps, so determine how many
    # times it was run previously.
    num_val_runs = sum(
        map(lambda i: u.itstime(i, config.log_eval_steps, total_steps),
            range(1, first_step + 1)))
    for val_name, (val_iter, val_steps) in val_iter_splits.items():
      val_iter = itertools.islice(val_iter, num_val_runs * val_steps, None)
      val_iter_splits[val_name] = (val_iter, val_steps)

  # Using a python integer for step here, because opt.state.step is allocated
  # on TPU during replication.
  for step, train_batch, lr_repl, reset_covmat_repl in zip(
      range(first_step + 1, total_steps + 1), train_iter, lr_iter,
      reset_covmat_iter):
    mw.step_start(step)

    with jax.profiler.TraceContext('train_step', step_num=step, _r=1):
      # TODO(jereliu): Expand to allow precision matrix resetting.
      (opt_repl, states_repl, loss_value, rngs_loop,
       extra_measurements) = update_fn(
           opt_repl,
           states_repl,
           lr_repl,
           reset_covmat_repl,
           train_batch['image'],
           train_batch['labels'],
           rng=rngs_loop)

    if jax.host_id() == 0:
      profiler(step)

    # Checkpoint saving
    if u.itstime(step, config.get('checkpoint_steps'), total_steps, host=0):
      write_note('Checkpointing...')
      chrono.pause()
      u.checkpointing_timeout(checkpoint_writer,
                              config.get('checkpoint_timeout', 1))
      checkpoint_extra['accum_train_time'] = chrono.accum_train_time
      # We need to transfer the weights over now or else we risk keeping them
      # alive while they'll be updated in a future step, creating hard to debug
      # memory errors (see b/160593526). Also, takes device 0's params only.
      # For GP layer, we will also do the same for untrainable parameters
      # (`states`). This is ok since `random features` are frozen throughout
      # pre-training, and `precision matrix` is a finetuning-specific parameters
      # that will be re-learned in the finetuning task.
      opt_cpu = jax.tree_map(lambda x: np.array(x[0]), opt_repl)
      states_cpu = jax.tree_map(lambda x: np.array(x[0]), states_repl)

      # Check whether we want to keep a copy of the current checkpoint.
      copy_step = None
      if u.itstime(step, config.get('keep_checkpoint_steps'), total_steps):
        write_note('Keeping a checkpoint copy...')
        copy_step = step

      # Checkpoint should be a nested dictionary or FLAX datataclasses from
      # `flax.struct`. Both can be present in a checkpoint.
      checkpoint = {
          'opt': opt_cpu,
          'states': states_cpu,
          'extra': checkpoint_extra
      }
      checkpoint_writer = pool.apply_async(
          checkpoint_utils.save_checkpoint,
          (checkpoint, save_checkpoint_path, copy_step))
      chrono.resume()

    # Report training progress
    if u.itstime(step, config.log_training_steps, total_steps, host=0):
      write_note('Reporting training progress...')
      train_loss = loss_value[0]  # Keep to return for reproducibility tests.
      mw.measure('learning_rate', lr_repl[0])
      mw.measure('training_loss', loss_value[0])
      for name, value in extra_measurements.items():
        mw.measure(name, value[0])
      chrono.tick(step, mw.measure, write_note)

    # Report validation performance
    if u.itstime(step, config.log_eval_steps, total_steps):
      write_note('Evaluating on the validation set...')
      chrono.pause()
      for val_name, (val_iter, val_steps) in val_iter_splits.items():
        # Sets up evaluation metrics.
        ece_num_bins = config.get('ece_num_bins', 15)
        auc_num_bins = config.get('auc_num_bins', 1000)
        ece = rm.metrics.ExpectedCalibrationError(num_bins=ece_num_bins)
        calib_auc = rm.metrics.CalibrationAUC(correct_pred_as_pos_label=False)
        # TODO(jereliu): Extend to support soft multi-class probabilities.
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
        ncorrect, loss, nseen = 0, 0, 0
        for _, batch in zip(range(val_steps), val_iter):
          if val_name == 'cifar_10h':
            batch_ncorrect, batch_losses, batch_n, batch_metric_args = (
                cifar_10h_evaluation_fn(
                    opt_repl.target, states_repl, batch['image'],
                    batch['labels'], batch['mask']))
          else:
            batch_ncorrect, batch_losses, batch_n, batch_metric_args = (
                evaluation_fn(opt_repl.target, states_repl, batch['image'],
                              batch['labels'], batch['mask']))
          # All results are a replicated array shaped as follows:
          # (local_devices, per_device_batch_size, elem_shape...)
          # with each local device's entry being identical as they got psum'd.
          # So let's just take the first one to the host as numpy.
          ncorrect += np.sum(np.array(batch_ncorrect[0]))
          loss += np.sum(np.array(batch_losses[0]))
          nseen += np.sum(np.array(batch_n[0]))
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
            oc_auc_0_5.add_batch(d[m], label=l[m], custom_binning_score=c[m])
            oc_auc_1.add_batch(d[m], label=l[m], custom_binning_score=c[m])
            oc_auc_2.add_batch(d[m], label=l[m], custom_binning_score=c[m])
            oc_auc_5.add_batch(d[m], label=l[m], custom_binning_score=c[m])

            if val_name == 'cifar_10h':
              batch_label_diversity, batch_sample_diversity, batch_ged = cifar10h_utils.generalized_energy_distance(
                  label[m], p[m, :], 10)
              label_diversity.update_state(batch_label_diversity)
              sample_diversity.update_state(batch_sample_diversity)
              ged.update_state(batch_ged)

        val_loss = loss / nseen  # Keep to return for reproducibility tests.
        mw.measure(f'{val_name}_prec@1', ncorrect / nseen)
        mw.measure(f'{val_name}_loss', val_loss)
        mw.measure(f'{val_name}_ece', float(ece.result()['ece']))
        mw.measure(f'{val_name}_calib_auc',
                   float(calib_auc.result()['calibration_auc']))
        mw.measure(f'{val_name}_oc_auc_0.5%',
                   float(oc_auc_0_5.result()['collaborative_auc']))
        mw.measure(f'{val_name}_oc_auc_1%',
                   float(oc_auc_1.result()['collaborative_auc']))
        mw.measure(f'{val_name}_oc_auc_2%',
                   float(oc_auc_2.result()['collaborative_auc']))
        mw.measure(f'{val_name}_oc_auc_5%',
                   float(oc_auc_5.result()['collaborative_auc']))

        if val_name == 'cifar_10h':
          mw.measure(
              f'{val_name}_label_diversity', float(label_diversity.result()))
          mw.measure(
              f'{val_name}_sample_diversity', float(sample_diversity.result()))
          mw.measure(f'{val_name}_ged', float(ged.result()))

      # OOD eval
      # TODO(dusenberrymw): Add the OOD eval results to the training script
      # reproducibility tests.
      # There are two entries in the ood_ds dict (in-dist, ood), and that this
      # section computes metrics using both pieces. This is in contrast to
      # normal validation eval above where we eval metrics separately for each
      # val split in val_ds.
      if ood_ds:
        ood_metrics = {
            # MSP stands for maximum softmax probability, max(softmax(logits)).
            # MSP can be used as confidence score.
            'msp': {
                'score': [],
                'label': []
            },
            # Maha stands for Mahalanobis distance between the test input and
            # fitted class conditional Gaussian distributions based on the
            # embeddings. Mahalanobis distance can be used as uncertainty score
            # or in other words, negative Mahalanobis distance can be used as
            # confidence score.
            'maha': {
                'score': [],
                'label': []
            },
        }

        # Mean and cov of class conditional Guassian in Mahalanobis distance.
        mean_list, cov = None, None
        for val_name in ['train_maha', 'ind', 'ood']:
          # The dataset train_maha must come before ind and ood
          # because the train_maha will be used to esimate the class conditional
          # mean and shared covariance.
          val_iter, val_steps = ood_ds[val_name]
          ncorrect, loss, nseen = 0, 0, 0
          pre_logits_list, labels_list = [], []
          for _, batch in zip(range(val_steps), val_iter):
            batch_ncorrect, batch_losses, batch_n, batch_metric_args = (
                evaluation_fn(opt_repl.target, states_repl, batch['image'],
                              batch['labels'], batch['mask']))
            ncorrect += np.sum(np.array(batch_ncorrect[0]))
            loss += np.sum(np.array(batch_losses[0]))
            nseen += np.sum(np.array(batch_n[0]))
            # Here we parse batch_metric_args to compute OOD metrics.
            logits, labels, pre_logits, masks = batch_metric_args
            masks = np.array(masks[0], dtype=np.bool)
            if val_name == 'train_maha':
              # For Mahalanobis distance, we need to first fit class conditional
              # Gaussian using training data.
              pre_logits_list.append(np.array(pre_logits[0])[masks])
              labels_list.append(np.array(labels[0])[masks])
            else:
              # Computes Mahalanobis distance.
              if mean_list is not None and cov is not None:
                dists = ood_utils.compute_mahalanobis_distance(
                    np.array(pre_logits[0])[masks], mean_list, cov)
              else:
                raise ValueError(
                    'Mean and cov for Mahalanobis distance are not available.')
              # Computes Maximum softmax probability (MSP)
              probs = jax.nn.softmax(logits[0], axis=-1)[masks]
              # Update metric state for each metric in ood_metrics
              for metric_name, metric in ood_metrics.items():
                if 'msp' in metric_name:
                  ood_scores = np.max(probs, axis=-1)
                  ood_labels = np.ones_like(
                      ood_scores) if val_name == 'ind' else np.zeros_like(
                          ood_scores)
                elif 'maha' in metric_name:
                  ood_scores = np.min(dists, axis=-1)
                  ood_labels = np.zeros_like(
                      ood_scores) if val_name == 'ind' else np.ones_like(
                          ood_scores)
                else:
                  raise NotImplementedError(
                      'Only msp and maha are supported for OOD evaluation!'
                      'Got metric_name=%s!' % metric_name)
                metric['score'] += list(ood_scores)
                metric['label'] += list(ood_labels)

          if val_name == 'train_maha':
            pre_logits_train = np.vstack(np.vstack(pre_logits_list))
            labels_train = np.argmax(np.vstack(np.vstack(labels_list)), axis=-1)
            mean_list, cov = ood_utils.compute_mean_and_cov(
                pre_logits_train, labels_train)
          elif val_name == 'ind':
            mw.measure(f'{val_name}_prec@1', ncorrect / nseen)
            mw.measure(f'{val_name}_loss', loss / nseen)
        for metric_name, metric in ood_metrics.items():
          metric_values = ood_utils.ood_metrics(metric['label'],
                                                metric['score'])
          mw.measure(f'ood_{metric_name}_auroc', metric_values['auc-roc'])
          mw.measure(f'ood_{metric_name}_auprc', metric_values['auc-pr'])

      chrono.resume()

    if 'fewshot' in config:
      # Compute few-shot on-the-fly evaluation.
      if u.itstime(step, config.fewshot.log_steps, total_steps):
        chrono.pause()
        write_note(f'Few-shot evaluation...\n{chrono.note}')
        # Keep `results` to return for reproducibility tests.
        results, best_l2 = fewshotter.run_all(
            opt_repl.target,
            datasets=config.fewshot.datasets,
            states=states_repl)
        fewshotter.walk_results(mw.measure, results, best_l2)
        chrono.resume()
    mw.step_end()
    if config.get('testing_failure_step'):
      # Break early to simulate infra failures in test cases.
      if config.testing_failure_step == step:
        break

  write_note(f'Done!\n{chrono.note}')
  pool.close()
  pool.join()
  mw.close()

  # Return final training loss, validation loss, and fewshot results for
  # reproducibility test cases.
  return train_loss, val_loss, results


if __name__ == '__main__':
  # TODO(dusenberrymw): Refactor `main` such that there is a `train_eval`
  # function that returns values for tests and does not directly access flags,
  # and then have `main` return None.

  def _main(argv):
    main(argv)

  app.run(_main)  # Ignore the returned values from `main`.

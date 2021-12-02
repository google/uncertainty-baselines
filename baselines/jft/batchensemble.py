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

"""BatchEnsemble Vision Transformer."""

import copy
import functools
import itertools
import multiprocessing
import os
import time

from absl import app
from absl import flags
from absl import logging
from clu import metric_writers
from clu import preprocess_spec
import flax
import flax.jax_utils
import flax.struct
import jax
import jax.config
import jax.nn
import jax.numpy as jnp
import ml_collections
import numpy as np
import robustness_metrics as rm

import tensorflow as tf
from tensorflow.io import gfile
import uncertainty_baselines as ub
import batchensemble_utils  # local file import from baselines.jft
import checkpoint_utils  # local file import from baselines.jft
import input_utils  # local file import from baselines.jft
import preprocess_utils  # local file import from baselines.jft
import train_utils  # local file import from baselines.jft

# TODO(dusenberrymw): Open-source remaining imports.
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

# Adds jax flags to the program.
jax.config.parse_flags_with_absl()


def _log_average_probs(logits: jnp.ndarray) -> jnp.ndarray:
  # TODO(zmariet): dedicated eval loss function.
  ens_size, _, _ = logits.shape
  log_p = jax.nn.log_softmax(logits)  # (ensemble_size, batch_size, num_classes)
  log_p = jax.nn.logsumexp(log_p, axis=0) - jnp.log(ens_size)
  return log_p


def main(_):
  config = flags.FLAGS.config
  output_dir = flags.FLAGS.output_dir
  tf.io.gfile.makedirs(output_dir)

  seed = config.get('seed', 0)
  rng = jax.random.PRNGKey(seed)
  tf.random.set_seed(seed)

  save_checkpoint_path = None
  if config.get('checkpoint_steps'):
    gfile.makedirs(output_dir)
    save_checkpoint_path = os.path.join(output_dir, 'checkpoint.npz')

  # Create an asynchronous multi-metric writer.
  writer = metric_writers.create_default_writer(
      output_dir, just_logging=jax.process_index() > 0)
  checkpoint_writer = None

  # Loss to apply.
  loss_to_apply = getattr(
      train_utils, config.get('loss_to_apply', 'softmax_xent'))
  compute_ece = config.get('compute_ece', False)
  is_sigmoid = config.get('loss_to_apply', 'softmax_xent') == 'sigmoid_xent'
  if compute_ece and is_sigmoid:
    error_msg = 'Inconsistent config: ECE can only be used with "softmax_xent".'
    raise ValueError(error_msg)

  ens_size = config.get('model.transformer.ens_size', 1)

  # The pool is used to perform misc operations such as logging in async way.
  pool = multiprocessing.pool.ThreadPool()

  # Ideally, this should make code deterministic, but for many reasons we are
  # not there yet. For instance, tf.data.map is not determisntic.
  rng_generator = tf.random.Generator.from_seed(config.get('seed', 0))
  tf.random.set_global_generator(
      rng_generator.split(jax.process_count())[jax.process_index()])

  logging.info('Number of devices: %s  (process index: %s)', jax.device_count(),
               jax.process_index())
  logging.info('Config:\n%s', str(config))

  def write_note(note):
    if jax.host_id() == 0:
      logging.info('NOTE: %s', note)

  write_note('Initializing...')

  # Verify settings to make sure no checkpoints are accidentally missed.
  if config.get('keep_checkpoint_steps'):
    assert config.get('checkpoint_steps'), 'Specify `checkpoint_steps`.'
    assert config.keep_checkpoint_steps % config.checkpoint_steps == 0, (
        f'`keep_checkpoint_steps` ({config.checkpoint_steps}) should be'
        f'divisible by `checkpoint_steps ({config.checkpoint_steps}).`')

  if (config.batch_size % jax.device_count() != 0 or
      config.batch_size_eval % jax.device_count() != 0):
    raise ValueError(f'Batch sizes ({config.batch_size} and '
                     f'{config.batch_size_eval}) must be divisible by '
                     f'the number of devices ({jax.device_count()})')

  batch_size_per_host = config.batch_size // jax.process_count()
  batch_size_per_core = config.batch_size // jax.device_count()
  batch_size_per_host_eval = config.batch_size_eval // jax.process_count()

  rng, train_ds_rng = jax.random.split(rng)
  train_ds_rng = jax.random.fold_in(train_ds_rng, jax.process_index())

  train_ds = input_utils.get_data(
      dataset=config.dataset,
      split=config.train_split,
      rng=train_ds_rng,
      host_batch_size=batch_size_per_host,
      preprocess_fn=preprocess_spec.parse(
          spec=config.pp_train, available_ops=preprocess_utils.all_ops()),
      shuffle_buffer_size=config.shuffle_buffer_size,
      prefetch_size=config.get('prefetch_to_host', 2),
      data_dir=config.get('dataset_dir'))
  train_iter = input_utils.start_input_pipeline(
      train_ds, config.get('prefetch_to_device', 1))

  ntrain_img = input_utils.get_num_examples(
      config.dataset,
      split=config.train_split,
      host_batch_size=batch_size_per_host,
      data_dir=config.get('dataset_dir'))
  steps_per_epoch = ntrain_img / config.batch_size

  if config.get('num_epochs'):
    total_steps = int(config.num_epochs * steps_per_epoch)
    assert not config.get('total_steps'), 'Set either num_epochs or total_steps'
  else:
    total_steps = config.total_steps

  logging.info(
      'Running for %d steps, that means %f epochs and %f steps per epoch',
      total_steps, total_steps * config.batch_size / ntrain_img,
      steps_per_epoch)

  def _get_val_split(dataset, split, pp_eval, data_dir=None):
    # We do ceil rounding such that we include the last incomplete batch.
    nval_img = input_utils.get_num_examples(
        dataset,
        split=split,
        host_batch_size=batch_size_per_host_eval,
        drop_remainder=False,
        data_dir=data_dir)
    val_steps = int(np.ceil(nval_img / config.batch_size_eval))
    logging.info('Running validation for %d steps for %s, %s', val_steps,
                 dataset, split)

    if isinstance(pp_eval, str):
      pp_eval = preprocess_spec.parse(
          spec=pp_eval, available_ops=preprocess_utils.all_ops())

    val_ds = input_utils.get_data(
        dataset=dataset,
        split=split,
        rng=None,
        host_batch_size=batch_size_per_host_eval,
        preprocess_fn=pp_eval,
        cache=config.get('val_cache', 'batched'),
        repeat_after_batching=True,
        shuffle=False,
        prefetch_size=config.get('prefetch_to_host', 2),
        drop_remainder=False,
        data_dir=data_dir)
    val_iter = input_utils.start_input_pipeline(
        val_ds, config.get('prefetch_to_device', 1))

    return (val_iter, val_steps)

  val_iter_splits = {
      'val':
          _get_val_split(
              config.dataset,
              split=config.val_split,
              pp_eval=config.pp_eval,
              data_dir=config.get('data_dir'))
  }

  # Note: we return the train loss and val loss for use in reproducibility unit
  # tests.
  train_loss = -jnp.inf
  val_loss = {val_name: -jnp.inf for val_name, _ in val_iter_splits.items()}
  # TODO(zmariet): Add fewshot evaluation.
  fewshot_results = {'dummy': {(0, 1): -jnp.inf}}

  opt_name = config.get('optim_name')
  opt_def = getattr(flax.optim, opt_name)(**config.get('optim', {}))

  eval_config = copy.deepcopy(config)
  if config.get('eval_overrides'):
    with eval_config.unlocked():
      eval_config.update(config.eval_overrides)
  model = getattr(ub.models, config.model_name)

  model_train = model(
      num_classes=config.num_classes, train=True, **config.model)
  model_eval = model(
      num_classes=config.num_classes, train=False, **config.model)

  image_size = tuple(train_ds.element_spec['image'].shape[2:])
  logging.info('Model initialization: Starting.')

  @functools.partial(jax.jit, backend='cpu')
  def init(rng_init):
    dummy_input = jnp.zeros((batch_size_per_core,) + image_size, jnp.float32)
    params = model_eval.init(rng_init, dummy_input)['params']
    params = flax.core.unfreeze(params)

    # Set bias in the head to a low value, such that loss is small initially.
    params['head']['bias'] = jnp.full_like(
        params['head']['bias'], config.get('init_head_bias', 0))

    # init head kernel to all zeros for fine-tuning
    if config.get('model_init'):
      params['head']['kernel'] = jnp.full_like(params['head']['kernel'], 0)

    return params

  rng, rng_init = jax.random.split(rng)
  params_init = init(rng_init)
  opt_init = opt_def.create(flax.core.freeze(params_init))

  weight_decay_rules = config.get('weight_decay', []) or []
  rescale_value = config.lr.base if config.get('weight_decay_decouple') else 1.
  weight_decay_fn = train_utils.get_weight_decay_fn(
      weight_decay_rules=weight_decay_rules, rescale_value=rescale_value)

  def batch_loss_fn(params, inputs, labels, *args, **kwargs):
    logits, _ = model_train.apply(
        {'params': params}, inputs, *args, **kwargs)
    labels = jnp.tile(labels, (ens_size, 1))
    loss = jnp.mean(loss_to_apply(logits=logits, labels=labels))
    return loss, dict()

  @functools.partial(jax.pmap, axis_name='batch', donate_argnums=(0, 1))
  def pmap_update_fn(opt, rngs, lr, images, labels):
    return batchensemble_utils.update_fn_be(
        opt=opt, rngs=rngs, lr=lr, images=images, labels=labels,
        batch_loss_fn=batch_loss_fn,
        weight_decay_fn=weight_decay_fn,
        plot_grad_norm_name_fn=None,
        plot_grads_nan_inf=config.get('plot_grads_nan_inf', True),
        max_grad_norm_global=config.get('clip_grad_norm', None),
        frozen_vars_patterns=config.get('frozen_var_patterns', None),
        fast_weight_lr_multiplier=config.get('fast_weight_lr_multiplier', None))

  @functools.partial(jax.pmap, axis_name='batch')
  def evaluation_fn(params, images, labels, mask):
    # Ignore the entries with all zero labels for evaluation.
    mask *= labels.max(axis=1)
    tiled_logits, _ = model_eval.apply({'params': flax.core.freeze(params)},
                                       images)
    ens_logits = _log_average_probs(
        jnp.asarray(jnp.split(tiled_logits, ens_size)))

    losses = train_utils.softmax_xent(logits=ens_logits,
                                      labels=labels[:, :config.num_classes],
                                      reduction=False)
    loss = jax.lax.psum(losses * mask, axis_name='batch')

    top1_idx = jnp.argmax(ens_logits, axis=1)
    top1_correct = jnp.take_along_axis(labels, top1_idx[:, None], axis=1)[:, 0]
    ncorrect = jax.lax.psum(top1_correct * mask, axis_name='batch')
    n = jax.lax.psum(mask, axis_name='batch')

    metric_args = jax.lax.all_gather([ens_logits, labels, mask],
                                     axis_name='batch')
    return ncorrect, loss, n, metric_args

  checkpoint_data = checkpoint_utils.maybe_load_checkpoint(
      train_loop_rngs=rng,
      save_checkpoint_path=save_checkpoint_path,
      init_optimizer=opt_init,
      init_params=opt_init.target,
      init_fixed_model_states=None,
      default_reinit_params=['head/bias', 'head/kernel'],
      config=config)

  opt = checkpoint_data.optimizer
  train_loop_rngs = {'params': checkpoint_data.train_loop_rngs}
  opt = opt.replace(target=flax.core.freeze(opt.target))
  first_step = int(opt.state.step)

  opt = flax.jax_utils.replicate(opt)
  accumulated_train_time = checkpoint_data.accumulated_train_time

  start_time = time.time()

  # Prepare the learning-rate and pre-fetch it to device to avoid delays.
  lr_fn = train_utils.create_learning_rate_schedule(total_steps,
                                                    **config.get('lr', {}))
  lr_iter = train_utils.prefetch_scalar(
      map(lr_fn, range(first_step, total_steps)),
      config.get('prefetch_to_device', 1))

  if first_step > 0:
    write_note('Advancing iterators after resuming from a checkpoint...')
    lr_iter = itertools.islice(lr_iter, first_step, None)
    train_iter = itertools.islice(train_iter, first_step, None)
    # NOTE: Validation eval is only run on certain steps, so determine how many
    # times it was run previously.
    num_val_runs = sum(
        map(
            lambda i: train_utils.itstime(i, config.log_eval_steps, total_steps
                                         ), range(1, first_step + 1)))
    for val_name, (val_iter, val_steps) in val_iter_splits.items():
      val_iter = itertools.islice(val_iter, num_val_runs * val_steps, None)
      val_iter_splits[val_name] = (val_iter, val_steps)

  log_training_first_n_steps = config.get('log_training_first_n_steps', -1)
  with metric_writers.ensure_flushes(writer):
    for step, train_batch, lr_repl in zip(
        range(first_step + 1, total_steps + 1), train_iter, lr_iter):
      opt, train_loop_rngs, loss_value, _ = pmap_update_fn(
          opt,
          train_loop_rngs,
          lr_repl,
          train_batch['image'],
          train_batch['labels'])

      # Checkpoint saving.
      if train_utils.itstime(
          step=step,
          every_n_steps=config.get('checkpoint_steps'),
          total_steps=total_steps, host=0, first=False):
        write_note('Checkpointing...')
        train_utils.checkpointing_timeout(checkpoint_writer,
                                          config.get('checkpoint_timeout', 1))
        time_since_last_start = float(time.time() - start_time)
        accumulated_train_time = accumulated_train_time + time_since_last_start
        opt_cpu = jax.tree_util.tree_map(lambda x: np.array(x[0]), opt)

        copy_step = None
        if train_utils.itstime(step, config.get('keep_checkpoint_steps'),
                               total_steps):
          write_note('Keeping a checkpoint copy...')
          copy_step = step

        checkpoint_data = checkpoint_utils.CheckpointData(
            train_loop_rngs=train_loop_rngs,
            optimizer=opt_cpu,
            accumulated_train_time=accumulated_train_time)
        checkpoint_writer = pool.apply_async(
            checkpoint_utils.checkpoint_trained_model,
            (checkpoint_data, save_checkpoint_path, copy_step))

      # Report training progress
      if (jax.process_index() == 0 and config.log_training_every_n_steps > 0 and
          (step % config.log_training_every_n_steps == 0 or
           step == total_steps or step < log_training_first_n_steps)):
        write_note('Reporting training progress...')
        train_loss = loss_value[0]
        time_elapsed = time.time() - start_time + accumulated_train_time
        img_sec_core = (
            config.batch_size * step / time_elapsed / jax.device_count())
        writer.write_scalars(step, {'learning_rate': lr_repl[0],
                                    'training_loss': np.mean(loss_value),
                                    'img/sec/core': img_sec_core,
                                    'epoch': step / steps_per_epoch})

      # Evaluate the model.
      if train_utils.itstime(step, config.log_eval_steps, total_steps):
        for val_name, (val_iter, val_steps) in val_iter_splits.items():
          # Sets up evaluation metrics.
          ece_num_bins = config.get('ece_num_bins', 15)
          auc_num_bins = config.get('auc_num_bins', 1000)
          ece = rm.metrics.ExpectedCalibrationError(num_bins=ece_num_bins)
          calib_auc = rm.metrics.CalibrationAUC(correct_pred_as_pos_label=False)
          oc_auc_0_5 = rm.metrics.OracleCollaborativeAUC(oracle_fraction=0.005,
                                                         num_bins=auc_num_bins)
          oc_auc_1 = rm.metrics.OracleCollaborativeAUC(oracle_fraction=0.01,
                                                       num_bins=auc_num_bins)
          oc_auc_2 = rm.metrics.OracleCollaborativeAUC(oracle_fraction=0.02,
                                                       num_bins=auc_num_bins)
          oc_auc_5 = rm.metrics.OracleCollaborativeAUC(oracle_fraction=0.05,
                                                       num_bins=auc_num_bins)

          # Runs evaluation loop.
          ncorrect, loss, nseen = 0, 0, 0
          for _, batch in zip(range(val_steps), val_iter):
            batch_ncorrect, batch_losses, batch_n, batch_metric_args = (
                evaluation_fn(opt.target, batch['image'],
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
            logits, labels, masks = batch_metric_args
            masks = np.array(masks[0], dtype=np.bool)
            logits = np.array(logits[0])
            probs = jax.nn.softmax(logits)
            # From one-hot to integer labels, as required by ECE.
            int_labels = np.argmax(np.array(labels[0]), axis=-1)
            int_preds = np.argmax(logits, axis=-1)
            confidence = np.max(probs, axis=-1)
            for p, c, l, d, m in zip(probs, confidence, int_labels,
                                     int_preds, masks):
              ece.add_batch(p[m, :], label=l[m])
              calib_auc.add_batch(d[m], label=l[m], confidence=c[m])
              # TODO(jereliu): Extend to support soft multi-class probabilities.
              oc_auc_0_5.add_batch(d[m], label=l[m], custom_binning_score=c[m])
              oc_auc_1.add_batch(d[m], label=l[m], custom_binning_score=c[m])
              oc_auc_2.add_batch(d[m], label=l[m], custom_binning_score=c[m])
              oc_auc_5.add_batch(d[m], label=l[m], custom_binning_score=c[m])

          val_loss[val_name] = loss / nseen  # Keep for reproducibility tests.
          val_measurements = {
              f'{val_name}_prec@1': ncorrect / nseen,
              f'{val_name}_loss': val_loss[val_name],
              f'{val_name}_ece': ece.result()['ece'],
              f'{val_name}_calib_auc': calib_auc.result()['calibration_auc'],
              f'{val_name}_oc_auc_0.5%': oc_auc_0_5.result()[
                  'collaborative_auc'],
              f'{val_name}_oc_auc_1%': oc_auc_1.result()['collaborative_auc'],
              f'{val_name}_oc_auc_2%': oc_auc_2.result()['collaborative_auc'],
              f'{val_name}_oc_auc_5%': oc_auc_5.result()['collaborative_auc'],
          }
          writer.write_scalars(step, val_measurements)

  pool.close()
  pool.join()

  # Return final training loss, validation loss, and fewshot results for
  # reproducibility test cases.
  return train_loss, val_loss, fewshot_results

if __name__ == '__main__':
  app.run(main)

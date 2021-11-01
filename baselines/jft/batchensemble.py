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
import multiprocessing
import time
from typing import Any, Iterable, Mapping, Tuple

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
import uncertainty_baselines as ub
import batchensemble_utils  # local file import
import input_utils  # local file import
import preprocess_utils  # local file import
import train_utils  # local file import

# TODO(dusenberrymw): Open-source remaining imports.
ensemble = None
train = None
xprof = None
core = None
xm = None
xm_api = None
BIG_VISION_DIR = None

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


def restore_model_and_put_to_devices(
    config: ml_collections.ConfigDict,
    output_dir: str,
    model: flax.nn.Module,
    optimizer: flax.optim.Optimizer,
    train_iter: Iterable[Any],
    rngs: Mapping[str, jnp.ndarray],
    thread_pool: multiprocessing.pool.ThreadPool,
) -> Tuple[flax.optim.Optimizer, Iterable[Any], jnp.ndarray, Mapping[str, Any]]:
  """Restores from latest available checkpoint and puts model to devices."""
  (optimizer, train_iter, rng_state_tf, rngs,
   global_state) = train.restore_checkpoints(
       workdir=output_dir,
       step=None,
       partition_specs=[],
       optimizer=optimizer,
       train_iter=train_iter,
       rng_state_tf=tf.random.get_global_generator().state.numpy(),
       rng_state_jax=rngs,
       global_state={},
       thread_pool=thread_pool)
  if global_state:
    # 1. If a checkpoint is present in the current work dir, continue training.
    logging.info('Continuing training from step %d', global_state['step'])
    # Shard parameters and optim state and put to the corresponding device.
    optimizer = core.tree_shard(optimizer)
  elif config.get('model_init_prefix'):
    # 2. Alternatively, initialize from the given model_init_prefix checkpoint.
    logging.info('Fine-tuning model from %r...', config.model_init_prefix)
    if not hasattr(model, 'load'):
      # Note: Likely due to use of .partial, model may end up being e.g.
      # a flax.nn.Base.PatchTransformer instead of experts_nn.PatchTransformer
      # This causes explicit checks for class equivalence to fail, and also
      # causes static type checking to fail. Checking for .load attribute
      # circumvents both these issues.
      raise ValueError((f'Loaded model {model} has no load method. Are you sure'
                        ' it is one of "PatchTransformer" and "Resformer"?'))
    restored_params = model.load(
        prefix=config.model_init_prefix,
        init_params=optimizer.target,
        model_params=config.model,
        keep_head=config.get('keep_head', False),
        partition_specs=[])
    # Shard restored parameters and replicate original optimizer state.
    optimizer = optimizer.replace(
        target=core.tree_shard(restored_params),
        state=flax.jax_utils.replicate(optimizer.state))
    global_state = {'step': 0, 'accum_train_time': 0.0}
  else:
    # 3. Use model initialized from scratch.
    logging.info('Initializing training from scratch...')
    optimizer = flax.jax_utils.replicate(optimizer)
    global_state = {'step': 0, 'accum_train_time': 0.0}
  # Set TF's global RNG generator and JAX's per-device RNG keys.
  train.rng_tf_set_global_generator(rng_state_tf)
  rngs_per_device = jax.tree_map(train.rng_jax_fold_host_if_needed_and_shard,
                                 rngs)
  return optimizer, train_iter, rngs_per_device, global_state


def main(_):
  config = flags.FLAGS.config
  output_dir = flags.FLAGS.output_dir
  tf.io.gfile.makedirs(output_dir)

  seed = config.get('seed', 0)
  partition_specs = []
  rng = jax.random.PRNGKey(seed)
  tf.random.set_seed(seed)

  # Create an asynchronous multi-metric writer.
  writer = metric_writers.create_default_writer(
      output_dir, just_logging=jax.process_index() > 0)

  # Loss to apply.
  loss_to_apply = getattr(core, config.get('loss_to_apply', 'softmax_xent'))
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

  opt_def = train.get_optimizer_from_config(config, f'{BIG_VISION_DIR}.optims')
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
  opt, rngs = train.model_and_optim_init(
      model_train.init, opt_def, (batch_size_per_core,) + image_size,
      config.get('init_head_bias'), config.get('seed', 0),
      config.get('extra_rngs', ['dropout', 'gating']))
  logging.info('Model initialization: Done.')
  # TODO(jpuigcerver): Support logging parameter count with new sharding.

  weight_decay_fn = train.get_weight_decay_function_from_config(config)
  batch_loss_fn = ensemble.wrap_ensemble_module_with_auxiliary_loss_fn(
      module=model_train,
      loss_fn=loss_to_apply,
      auxiliary_loss_weight=config.get('auxiliary_loss_weight', 0.0),
      ens_size=ens_size)

  update_fn = functools.partial(
      batchensemble_utils.update_fn_be,
      weight_decay_fn=weight_decay_fn,
      plot_grad_norm_name_fn=None,
      plot_grads_nan_inf=config.get('plot_grads_nan_inf', True),
      max_grad_norm_global=config.get('clip_grad_norm', None),
      frozen_vars_patterns=config.get('frozen_var_patterns', None),
      fast_weight_lr_multiplier=config.get('fast_weight_lr_multiplier', None))
  pmap_update_fn = core.pmap_sorted(
      update_fn, axis_name='batch', donate_argnums=(0, 1),
      static_broadcasted_argnums=(5,))

  @functools.partial(jax.pmap, axis_name='batch')
  def evaluation_fn(params, images, labels, mask):
    # Ignore the entries with all zero labels for evaluation.
    mask *= labels.max(axis=1)
    tiled_logits, _ = model_eval.apply({'params': flax.core.freeze(params)},
                                       images)
    ens_logits = jnp.asarray(jnp.split(tiled_logits, ens_size))

    losses = ensemble.ensemble_softmax_xent(
        logits=ens_logits, labels=labels)
    loss = jax.lax.psum(losses * mask, axis_name='batch')

    ncorrect = ensemble.ensemble_softmax_correct_multilabel(
        ens_logits, labels, mask, psum_axis_name='batch')
    n = jax.lax.psum(mask, axis_name='batch')
    logits = jnp.log(jnp.mean(jax.nn.softmax(ens_logits), axis=0))
    metric_args = jax.lax.all_gather([logits, labels, mask],
                                     axis_name='batch')
    return ncorrect, loss, n, metric_args

  # Restore parameters from checkpoints (if possible) and put to TPU devices.
  opt, train_iter, rngs_per_device, global_state = restore_model_and_put_to_devices(
      config, output_dir, model, opt, train_iter, rngs,
      pool)
  del rngs
  first_step = global_state['step']
  accum_train_time = global_state['accum_train_time']
  start_time = time.time()
  logging.info('Initial step for training = %d.', first_step)

  # Prepare the learning-rate and pre-fetch it to device to avoid delays.
  lr_fn = train_utils.create_learning_rate_schedule(total_steps,
                                                    **config.get('lr', {}))
  lr_iter = train_utils.prefetch_scalar(
      map(lr_fn, range(first_step, total_steps)),
      config.get('prefetch_to_device', 1))

  checkpoint_async_results = []
  log_training_first_n_steps = config.get('log_training_first_n_steps', -1)
  with metric_writers.ensure_flushes(writer):
    callback_fn = lambda x: x  # Do nothing.
    xprof_session = xprof.MultiStepXprofSession(
        profile_steps=20,    # For how many steps to profile after warmup.
        warmup_steps=170,    # For how many steps to wait before profiling.
        stop_callback_fn=callback_fn)
    for step, train_batch, lr_repl in zip(
        range(first_step + 1, total_steps + 1), train_iter, lr_iter):
      with xprof_session:
        with jax.profiler.StepTraceAnnotation(name='train', step_num=step):
          opt, rngs_per_device, loss_value, _ = pmap_update_fn(
              opt,
              rngs_per_device,
              lr_repl,
              train_batch['image'],
              train_batch['labels'],
              batch_loss_fn)

      # Checkpoint saving.
      backup_checkpoints_every_n_steps = config.get('backup_checkpoint_steps')
      if (step % config.write_checkpoint_every_n_steps == 0 or
          (backup_checkpoints_every_n_steps is not None and
           step % backup_checkpoints_every_n_steps == 0) or
          step == total_steps):
        # Before writing new checkpoints, make sure that all the previous
        # checkpoint shards have been completely written (hosts are synced).
        train.wait_async_results(
            checkpoint_async_results,
            timeout_secs=config.checkpoint_write_timeout_secs)
        train.sync_all_hosts()
        # Now host 0 can remove all the checkpoints older than the previous
        # checkpointed step. The pool is used to remove files in parallel.
        if jax.process_index() == 0:
          train.remove_old_checkpoints(
              output_dir,
              keep_steps_from=step - config.write_checkpoint_every_n_steps,
              keep_steps_multiple_of=backup_checkpoints_every_n_steps,
              thread_pool=pool)
        # Save checkpoint for the current step, asynchronously.
        # Note: Parameters on TPU are sliced and copied to CPU before scheduling
        # the asynchronous copy, to prevent any extra TPU memory usage.
        time_since_last_start = float(time.time() - start_time)
        checkpoint_async_results = train.save_checkpoints(
            workdir=output_dir,
            step=step,
            partition_specs=partition_specs,
            optimizer=opt,
            # TODO(jpuigcerver): start_input_pipeline() does not return a
            # serializable iterator. Also, serialization of a 'memory heavy'
            # tf.data.Dataset iterator may cause OOM (e.g. big shuffle buffer).
            train_iter=None,
            rng_state_tf=tf.random.get_global_generator().state.numpy(),
            rng_state_jax=rngs_per_device,
            global_state={
                # Note: 'step' is automatically added to this dictionary.
                'accum_train_time': accum_train_time + time_since_last_start,
            },
            thread_pool=pool)

      # Report training progress
      if (jax.process_index() == 0 and config.log_training_every_n_steps > 0 and
          (step % config.log_training_every_n_steps == 0 or
           step == total_steps or step < log_training_first_n_steps)):
        train_loss = loss_value[0]
        time_elapsed = time.time() - start_time + accum_train_time
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
            print('logits', logits.shape)
            probs = jax.nn.softmax(logits)
            print('probs', probs.shape)
            # From one-hot to integer labels, as required by ECE.
            int_labels = np.argmax(np.array(labels[0]), axis=-1)
            int_preds = np.argmax(logits, axis=-1)
            confidence = np.max(probs, axis=-1)
            print('confidence', confidence.shape)
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

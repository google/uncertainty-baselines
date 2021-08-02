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
from typing import Any, Iterable, Mapping, Sequence, Tuple

from absl import app
from absl import flags
from absl import logging
import flax
import flax.jax_utils
import flax.struct
import jax
import jax.config
import jax.nn
import jax.numpy as jnp
import ml_collections
import numpy as np
import tensorflow as tf
import uncertainty_baselines as ub
import batchensemble_utils  # local file import

# TODO(dusenberrymw): Open-source remaining imports.


config_flags.DEFINE_config_file(
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
    partition_specs: Sequence[PartitionSpec],
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
       partition_specs=partition_specs,
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
        partition_specs=partition_specs)
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

  partition_specs = []

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
      rng_generator.split(jax.host_count())[jax.host_id()])

  logging.info('Number of devices: %s  (host_id: %s)', jax.device_count(),
               jax.host_id())
  logging.info('Config:\n%s', str(config))

  if (config.batch_size % jax.device_count() != 0 or
      config.batch_size_eval % jax.device_count() != 0):
    raise ValueError(f'Batch sizes ({config.batch_size} and '
                     f'{config.batch_size_eval}) must be divisible by '
                     f'the number of devices ({jax.device_count()})')

  batch_size_per_host = config.batch_size // jax.host_count()
  batch_size_per_core = config.batch_size // jax.device_count()
  batch_size_per_host_eval = config.batch_size_eval // jax.host_count()
  # TODO(basilm): Remove when JFT2.6B is properly submitted.
  if config.dataset in jft_latest_pipeline.DATA_INFO:
    input_pipeline = jft_latest_pipeline
    cache = 'loaded'
  else:
    input_pipeline = default_input_pipeline
    cache = 'batched'

  train_ds = input_pipeline.get_data(
      dataset=config.dataset,
      data_dir=config.get('dataset_dir'),
      split=config.train_split,
      batch_size=batch_size_per_host,
      preprocess_fn=pp_builder.get_preprocess_fn(config.pp_train),
      shuffle_buffer_size=config.shuffle_buffer_size,
      cache=False)
  steps_per_epoch = input_pipeline.get_num_examples(
      config.dataset, config.train_split,
      data_dir=config.get('dataset_dir')) / config.batch_size
  total_steps = train.get_total_steps_from_config(config, steps_per_epoch)
  logging.info('Running for %d steps per epoch (%d steps total)',
               steps_per_epoch, total_steps)

  opt_def = train.get_optimizer_from_config(config, f'{BIG_VISION_DIR}.optims')
  eval_config = copy.deepcopy(config)
  if config.get('eval_overrides'):
    with eval_config.unlocked():
      eval_config.update(config.eval_overrides)
  model = getattr(ub.models, config.model_name)
  model_train = model(
      num_classes=config.num_classes, train=True, **config.model)
  model_eval = model(
      num_classes=config.num_classes, train=False, **eval_config.model)

  image_size = tuple(train_ds.element_spec['image'].shape[1:])
  logging.info('Model initialization: Starting.')
  opt, rngs = train.model_and_optim_init(
      model_train.init, opt_def, (batch_size_per_core,) + image_size,
      config.get('init_head_bias'), config.get('seed', 0),
      config.get('extra_rngs', ['dropout', 'gating']))
  logging.info('Model initialization: Done.')
  # TODO(jpuigcerver): Support logging parameter count with new sharding.

  if config.get('plot_grad_norm_patterns'):
    plot_grad_norm_name_fn = experts_utils.make_match_fn_from_prefixes(
        config.plot_grad_norm_patterns)
  else:
    plot_grad_norm_name_fn = None

  weight_decay_fn = train.get_weight_decay_function_from_config(config)
  batch_loss_fn = ensemble.wrap_ensemble_module_with_auxiliary_loss_fn(
      module=model_train,
      loss_fn=loss_to_apply,
      auxiliary_loss_weight=config.get('auxiliary_loss_weight', 0.0),
      ens_size=ens_size)
  if ens_size == 1:
    evaluation_fn = functools.partial(
        train.evaluation_fn,
        apply_fn=model_eval.apply,
        loss_fn=loss_to_apply,
        correct_fn=train.correct_multilabel,
        return_metric_args=compute_ece)
  else:
    evaluation_fn = functools.partial(
        ensemble.evaluation_fn,
        apply_fn=model_eval.apply,
        return_metric_args=compute_ece,
        ens_size=ens_size)
  pmap_evaluation_fn = core.pmap_sorted(evaluation_fn, axis_name='batch')

  update_fn = functools.partial(
      batchensemble_utils.update_fn_be,
      weight_decay_fn=weight_decay_fn,
      plot_grad_norm_name_fn=plot_grad_norm_name_fn,
      plot_grads_nan_inf=config.get('plot_grads_nan_inf', True),
      max_grad_norm_global=config.get('clip_grad_norm', None),
      frozen_vars_patterns=config.get('frozen_var_patterns', None),
      fast_weight_lr_multiplier=config.get('fast_weight_lr_multiplier', None))
  pmap_update_fn = core.pmap_sorted(
      update_fn, axis_name='batch', donate_argnums=(0, 1),
      static_broadcasted_argnums=(5,))

  # Restore parameters from checkpoints (if possible) and put to TPU devices.
  opt, train_iter, rngs_per_device, global_state = restore_model_and_put_to_devices(
      config, output_dir, partition_specs, model, opt, iter(train_ds), rngs,
      pool)
  del rngs
  first_step = global_state['step']
  accum_train_time = global_state['accum_train_time']
  start_time = time.time()
  logging.info('Initial step for training = %d.', first_step)

  local_devices = sorted(jax.local_devices(), key=lambda device: device.id)
  if config.get('ema', {}):
    ema_updater = ema.ExponentialMovingAverage(
        target=partitioning.tree_unreplicate_using_partition_specs(
            jax.tree_map(np.zeros_like, opt.target),
            partition_specs=partition_specs,
            local_devices=local_devices),
        num_updates=0,
        **config.ema)
  else:
    ema_updater = None
  if first_step != 0 and ema_updater is not None:
    ema_updater = train.restore_ema_checkpoints(
        output_dir,
        first_step,
        partition_specs,
        ema_updater,
        local_devices=local_devices,
        thread_pool=pool)

  train_iter = u.start_input_pipeline(train_iter, config.prefetch_to_device)
  eval_iters = train.get_dataset_eval_iters_from_config(
      config, batch_size_per_host_eval, cache, input_pipeline)
  lr_fn = u.create_learning_rate_schedule(
      config.batch_size, total_steps, steps_per_epoch, **config.lr)
  lr_iter = u.prefetch_scalar(map(lr_fn, range(first_step, total_steps)),
                              config.get('prefetch_to_device', 1))

  writer = metric_writers.create_default_writer(
      output_dir, just_logging=jax.host_id() > 0,
      summary_writer=config.get('write_tf_summaries', False))

  checkpoint_async_results = []
  log_training_first_n_steps = config.get('log_training_first_n_steps', -1)
  with metric_writers.ensure_flushes(writer):
    callback_fn = lambda x: x  # Do nothing.
    xprof_session = xprof.MultiStepXprofSession(
        profile_steps=20,    # For how many steps to profile after warmup.
        warmup_steps=170,    # For how many steps to wait before profiling.
        stop_callback_fn=callback_fn)
    for step, lr_repl in zip(range(first_step + 1, total_steps + 1), lr_iter):
      train_batch = next(train_iter)
      with xprof_session:
        with jax.profiler.StepTraceAnnotation(name='train', step_num=step):
          opt, rngs_per_device, loss_value, aux_info = pmap_update_fn(
              opt,
              rngs_per_device,
              lr_repl,
              train_batch['image'],
              train_batch['labels'],
              batch_loss_fn)

      if (ema_updater is not None and
          step % config.get('ema', {}).get('period', 10) == 0):
        ema_updater = ema_updater.update(
            partitioning.tree_unreplicate_using_partition_specs(
                tree=opt.target,
                partition_specs=partition_specs,
                local_devices=local_devices))

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
        if jax.host_id() == 0:
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
        if ema_updater is not None:
          checkpoint_async_results.append(train.save_ema_checkpoints(
              workdir=output_dir,
              step=step,
              partition_specs=partition_specs,
              ema_updater=ema_updater,
              local_devices=local_devices,
              thread_pool=pool))

      # Report training progress
      if (jax.host_id() == 0 and config.log_training_every_n_steps > 0 and
          (step % config.log_training_every_n_steps == 0 or
           step == total_steps or step < log_training_first_n_steps)):
        time_elapsed = time.time() - start_time + accum_train_time
        img_sec_core = (
            config.batch_size * step / time_elapsed / jax.device_count())
        writer.write_scalars(step, {'learning_rate': lr_repl[0],
                                    'training_loss': np.mean(loss_value),
                                    'img/sec/core': img_sec_core,
                                    'epoch': step / steps_per_epoch})
        if aux_info:
          # Per-block info has to be dealt especially.
          if 'per_block_info' in aux_info:
            scalar_metrics_to_aggregate = config.get(
                'scalar_metrics_to_aggregate', ())
            metrics.write_info_to_metric_writer(
                metric_writer=writer,
                step=step,
                gating_info_dict=jax.tree_map(lambda x: np.mean(x, axis=0),
                                              aux_info['per_block_info']),
                scalar_metrics_to_aggregate=scalar_metrics_to_aggregate,
                write_matrices=True)
            del aux_info['per_block_info']
          # Plot rest of metrics as scalars.
          writer.write_scalars(
              step, {key: np.mean(value) for key, value in aux_info.items()})


      # Run checks to detect if the model partitioning is unhealthy.
      # Global health metrics will be logged, and in case of problems a
      # WARNING or ERROR message will be logged.
      train.monitor_partitioning_health(
          optimizer=opt,
          partition_specs=partition_specs,
          metric_writer=writer,
          step=step,
          first_step=first_step + 1,
          every_n_steps=config.get('check_partitioning_health_every_n_steps',
                                   total_steps // 20))
      # Evaluate model on validation, test, ...
      rngs_per_device = train.run_evaluation_on_multiple_splits(
          pmap_evaluation_fn, opt.target, eval_iters, rngs_per_device,
          step / steps_per_epoch, step, total_steps,
          config.run_evaluation_every_n_steps, writer, compute_ece,
          config.get('ece_num_bins', 15), suffix='')
      if ema_updater and config.run_evaluation_every_n_steps > 0 and (
          step == first_step + 1 or
          step % config.run_evaluation_every_n_steps == 0 or
          step == total_steps):
        logging.info('Evaluation with EMA weights at step %d: started.', step)
        # Copy current parameters to CPU. Only one replica of each local
        # partition is copied to prevent redundant data transfers (e.g.
        # non-expert parameters).
        curr_params = partitioning.tree_unreplicate_using_partition_specs(
            tree=opt.target,
            partition_specs=partition_specs,
            local_devices=local_devices)
        # Block curr_params until TPU->CPU copy has finished to prevent multiple
        # copies of the TPU parameters.
        curr_params = core.tree_block_until_ready(curr_params)
        # Allow TPU parameters to be freed.
        opt = opt.replace(target=None)
        # Copy EMA parameters to TPU and run evaluation.
        rngs_per_device = train.run_evaluation_on_multiple_splits(
            pmap_evaluation_fn,
            partitioning.tree_replicate_from_partitioned_tree(
                ema_updater.get(),
                partition_specs=partition_specs,
                local_devices=local_devices),
            eval_iters, rngs_per_device, step / steps_per_epoch, step,
            total_steps, config.run_evaluation_every_n_steps, writer,
            compute_ece, config.get('ece_num_bins', 15), suffix='_ema')
        rngs_per_device = core.tree_block_until_ready(rngs_per_device)
        # Copy current parameters back to the TPU.
        opt = opt.replace(
            target=partitioning.tree_replicate_from_partitioned_tree(
                curr_params,
                partition_specs=partition_specs,
                local_devices=local_devices))
        logging.info('Evaluation with EMA weights at step %d: finished.', step)
        del curr_params

  pool.close()
  pool.join()


if __name__ == '__main__':
  app.run(main)

"""
Custom segmentation_trainer.py

# cannot import train_step, eval_step due to tuple segmenter output in ub implementation
Minor changes to account for ub models which ouput a tuple (logits, dict)
"""

import functools
from typing import Any, Callable, Dict, Tuple, Optional, Type

import flax.linen as nn
import jax
import jax.numpy as jnp
import jax.profiler
import ml_collections
import numpy as np
from absl import logging
from clu import metric_writers
from clu import periodic_actions
from flax import jax_utils
from jax.experimental.optimizers import clip_grads

from scenic.dataset_lib import dataset_utils
from scenic.model_lib.base_models import base_model
from scenic.train_lib import lr_schedules
from scenic.train_lib import optimizers
from scenic.train_lib import train_utils

# instead of importing we use local functions
# from scenic.train_lib.segmentation_trainer import train_step, eval_step, _draw_side_by_side
Batch = Dict[str, jnp.ndarray]
MetricFn = Callable[[jnp.ndarray, Dict[str, jnp.ndarray]],
                    Dict[str, Tuple[float, int]]]
LossFn = Callable[[jnp.ndarray, Batch, Optional[jnp.ndarray]], float]

from scenic.train_lib.segmentation_trainer import _draw_side_by_side, get_confusion_matrix
from flax.training.checkpoints import restore_checkpoint as flax_restore_checkpoint
from scenic.train_lib import pretrain_utils

from pretrainer_utils import load_bb_config


def train_step(
    *,
    flax_model: nn.Module,
    train_state: train_utils.TrainState,
    batch: Batch,
    learning_rate_fn: Callable[[int], float],
    loss_fn: LossFn,
    metrics_fn: MetricFn,
    config: ml_collections.ConfigDict,
    debug: Optional[bool] = False
) -> Tuple[train_utils.TrainState, Dict[str, Tuple[float, int]], float,
           jnp.ndarray]:
  """Runs a single step of training.

  Given the state of the training and a batch of data, computes
  the loss and updates the parameters of the model.

  Note that in this code, the buffers of the first (train_state) and second
  (batch) arguments are donated to the computation.

  Args:
    flax_model: A Flax model.
    train_state: The state of training including the current
      global_step, model_state, rng, and optimizer. The buffer of this argument
      can be donated to the computation.
    batch: A single batch of data. The buffer of this argument can be donated to
      the computation.
    learning_rate_fn: learning rate scheduler which give the global_step
      generates the learning rate.
    loss_fn: A loss function that given logits, a batch, and parameters of the
      model calculates the loss.
    metrics_fn: A metrics function that given logits and batch of data,
      calculates the metrics as well as the loss.
    config: Configurations of the experiment.
    debug: Whether the debug mode is enabled during training.
      `debug=True` enables model specific logging/storing some values using
      jax.host_callback.

  Returns:
    Updated state of training, computed metrics, learning rate, and predictions
      for logging.
  """
  new_rng, rng = jax.random.split(train_state.rng)
  # Bind the rng to the host/device we are on.
  dropout_rng = train_utils.bind_rng_to_host_device(
      rng, axis_name='batch', bind_to='device')

  def training_loss_fn(params):
    variables = {'params': params, **train_state.model_state}
    (logits, _), new_model_state = flax_model.apply(
        variables,
        batch['inputs'],
        mutable=['batch_stats'],
        train=True,
        rngs={'dropout': dropout_rng},
        debug=debug)
    loss = loss_fn(logits, batch, variables['params'])
    return loss, (new_model_state, logits)

  compute_gradient_fn = jax.value_and_grad(training_loss_fn, has_aux=True)
  step = train_state.global_step
  lr = learning_rate_fn(step)
  (train_cost,
   (new_model_state,
    logits)), grad = compute_gradient_fn(train_state.optimizer.target)

  del train_cost
  # Re-use same axis_name as in the call to `pmap(...train_step...)` below.
  grad = jax.lax.pmean(grad, axis_name='batch')

  if config.get('max_grad_norm', None) is not None:
    grad = clip_grads(grad, config.max_grad_norm)

  new_optimizer = train_state.optimizer.apply_gradient(grad, learning_rate=lr)

  # Explicit weight decay, if necessary.
  if config.get('explicit_weight_decay', None) is not None:
    new_optimizer = new_optimizer.replace(
        target=optimizers.tree_map_with_names(
            functools.partial(
                optimizers.decay_weight_fn,
                lr=lr,
                decay=config.explicit_weight_decay),
            new_optimizer.target,
            match_name_fn=lambda name: 'kernel' in name))

  metrics = metrics_fn(logits, batch)
  new_train_state = train_state.replace(  # pytype: disable=attribute-error
      global_step=step + 1,
      optimizer=new_optimizer,
      model_state=new_model_state,
      rng=new_rng)
  return new_train_state, metrics, lr, jnp.argmax(logits, axis=-1)


def eval_step(
    *,
    flax_model: nn.Module,
    train_state: train_utils.TrainState,
    batch: Batch,
    metrics_fn: MetricFn,
    debug: Optional[bool] = False
) -> Tuple[Batch, jnp.ndarray, Dict[str, Tuple[float, int]], jnp.ndarray]:
  """Runs a single step of training.

  Note that in this code, the buffer of the second argument (batch) is donated
  to the computation.

  Assumed API of metrics_fn is:
  ```metrics = metrics_fn(logits, batch)
  where batch is yielded by the batch iterator, and metrics is a dictionary
  mapping metric name to a vector of per example measurements. eval_step will
  aggregate (by summing) all per example measurements and divide by the
  aggregated normalizers. For each given metric we compute:
  1/N sum_{b in batch_iter} metric(b), where  N is the sum of normalizer
  over all batches.

  Args:
    flax_model: A Flax model.
    train_state: TrainState, the state of training including the current
      global_step, model_state, rng, and optimizer. The buffer of this argument
      can be donated to the computation.
    batch: A single batch of data. a metrics function, that given logits and
      batch of data, calculates the metrics as well as the loss.
    metrics_fn: A metrics function, that given logits and batch of data,
      calculates the metrics as well as the loss.
    debug: Whether the debug mode is enabled during evaluation.
      `debug=True` enables model specific logging/storing some values using
      jax.host_callback.

  Returns:
    Batch, predictions and calculated metrics.
  """
  variables = {
      'params': train_state.optimizer.target,
      **train_state.model_state
  }
  (logits, _) = flax_model.apply(
      variables, batch['inputs'], train=False, mutable=False, debug=debug)
  metrics = metrics_fn(logits, batch)

  confusion_matrix = get_confusion_matrix(
      labels=batch['label'], logits=logits, batch_mask=batch['batch_mask'])

  # Collect predictions and batches from all hosts.
  predictions = jnp.argmax(logits, axis=-1)
  predictions = jax.lax.all_gather(predictions, 'batch')
  batch = jax.lax.all_gather(batch, 'batch')
  confusion_matrix = jax.lax.all_gather(confusion_matrix, 'batch')

  return batch, predictions, metrics, confusion_matrix


def train(
    *,
    rng: jnp.ndarray,
    config: ml_collections.ConfigDict,
    model_cls: Type[base_model.BaseModel],
    dataset: dataset_utils.Dataset,
    workdir: str,
    writer: metric_writers.MetricWriter,
) -> Tuple[train_utils.TrainState, Dict[str, Any], Dict[str, Any]]:
  """Main training loop lives in this function.

  Given the model class and dataset, it prepares the items needed to run the
  training, including the TrainState.

  Args:
    rng: Jax rng key.
    config: Configurations of the experiment.
    model_cls: Model class; A model has a flax_module, a loss_fn, and a
      metrics_fn associated with it.
    dataset: The dataset that has train_iter, eval_iter, meta_data, and
      optionally, test_iter.
    workdir: Directory for checkpointing.
    writer: CLU metrics writer instance.

  Returns:
    train_state that has the state of training (including current
      global_step, model_state, rng, and the optimizer), train_summary
      and eval_summary which are dict of metrics. These outputs are used for
      regression testing.

  Timeline:
  - Updated from scenic.train_lib.segmentation_trainer.train
  """
  lead_host = jax.process_index() == 0
  # Build the loss_fn, metrics, and flax_model.
  model = model_cls(config, dataset.meta_data)

  # Initialize model.
  rng, init_rng = jax.random.split(rng)
  (params, model_state, num_trainable_params,
   gflops) = train_utils.initialize_model(
       model_def=model.flax_model,
       input_spec=[(dataset.meta_data['input_shape'],
                    dataset.meta_data.get('input_dtype', jnp.float32))],
       config=config,
       rngs=init_rng)

  # Create optimizer.
  # We jit this, such that the arrays that are created are created on the same
  # device as the input is, in this case the CPU. Else they'd be on device[0].
  optimizer = jax.jit(
      optimizers.get_optimizer(config).create, backend='cpu')(
          params)
  rng, train_rng = jax.random.split(rng)
  train_state = train_utils.TrainState(
      global_step=0,
      optimizer=optimizer,
      model_state=model_state,
      rng=train_rng,
      accum_train_time=0)
  start_step = train_state.global_step

  # Load pretrained backbone
  if start_step == 0 and config.get('load_pretrained_backbone', False):
    # TODO(kellybuchanan): check out partial loader in
    # https://github.com/google/uncertainty-baselines/commit/083b1dcc52bb1964f8917d15552ece8848d582ae#

    bb_checkpoint_path = config.pretrained_backbone_configs.get('checkpoint_path')
    checkpoint_format = config.pretrained_backbone_configs.get('checkpoint_format', 'ub')
    # bb_model_cfg_file = config.pretrained_backbone_configs.get('checkpoint_cfg')

    # Loader from scenic
    # cannot restore using flax
    # Mathias suggested to try flax_restore_checkpoint
    # bb_train_state = flax_restore_checkpoint(bb_checkpoint_path, target=None)
    # but we get an error *** msgpack.exceptions.ExtraData: unpack(b) received extra data.

    # TODO(kellybuchanan): read config file directly from bb_model_cfg_file
    restored_model_cfg = load_bb_config(config)

    if checkpoint_format == 'ub':
        # import pdb; pdb.set_trace()
        # load params from checkpoint
        bb_train_state = pretrain_utils.convert_bigvision_to_scenic_checkpoint(
          checkpoint_path=bb_checkpoint_path,
          convert_to_linen=False)

        # option 1: failed as variables are a frozen dictionary
        # could be used with flax.core.unfreeze, flax.core.freeze
        train_state = model.init_backbone_from_train_state(train_state,
                                             bb_train_state,
                                             restored_model_cfg,
                                             model_prefix_path=['backbone'])

        # option2: it fails for embeddings as this mode
        # doesn't allow to specify loaded params .
        # model_prefix_path = ['backbone']
        # train_state = pretrain_utils.init_from_pretrain_state(
        #    train_state, bb_train_state, model_prefix_path=model_prefix_path)


    else:
      raise NotImplementedError("")

  elif start_step == 0:
    logging.info('Not restoring from any pretrained_backbone.')

  if config.checkpoint:
    train_state, start_step = train_utils.restore_checkpoint(
        workdir, train_state)
  else:
    logging.info('Not restoring from any checkpoints.')

  # Replicate the optimzier, state, and rng.
  train_state = jax_utils.replicate(train_state)
  del params  # Do not keep a copy of the initial params.

  # Calculate the total number of training steps.
  total_steps, steps_per_epoch = train_utils.get_num_training_steps(
      config, dataset.meta_data)
  # Get learning rate scheduler.
  learning_rate_fn = lr_schedules.get_learning_rate_fn(config)

  train_step_pmapped = jax.pmap(
      functools.partial(
          train_step,
          flax_model=model.flax_model,
          learning_rate_fn=learning_rate_fn,
          loss_fn=model.loss_function,
          metrics_fn=model.get_metrics_fn('train'),
          config=config,
          debug=config.debug_train),
      axis_name='batch',
      # We can donate both buffers of train_state and train_batch.
      donate_argnums=(0, 1),
  )

  ############### EVALUATION CODE #################

  eval_step_pmapped = jax.pmap(
      functools.partial(
          eval_step,
          flax_model=model.flax_model,
          metrics_fn=model.get_metrics_fn('validation'),
          debug=config.debug_eval),
      axis_name='batch',
      # We can donate the eval_batch's buffer.
     )

  # Ceil rounding such that we include the last incomplete batch.
  total_eval_steps = int(
      np.ceil(dataset.meta_data['num_eval_examples'] / config.batch_size))
  steps_per_eval = config.get('steps_per_eval') or total_eval_steps

  def evaluate(train_state: train_utils.TrainState,
               step: int) -> Dict[str, Any]:
    eval_metrics = []
    eval_all_confusion_mats = []
    # Sync model state across replicas.
    train_state = train_utils.sync_model_state_across_replicas(train_state)
    def to_cpu(x):
      return jax.device_get(dataset_utils.unshard(jax_utils.unreplicate(x)))
    for _ in range(steps_per_eval):
      eval_batch = next(dataset.valid_iter)
      e_batch, e_predictions, e_metrics, confusion_matrix = eval_step_pmapped(
          train_state=train_state, batch=eval_batch)
      eval_metrics.append(train_utils.unreplicate_and_get(e_metrics))
      # Evaluate global metrics on one of the hosts (lead_host), but given
      # intermediate values collected from all hosts.
      if lead_host and global_metrics_fn is not None:
        # Collect data to be sent for computing global metrics.
        eval_all_confusion_mats.append(to_cpu(confusion_matrix))

    eval_global_metrics_summary = {}
    if lead_host and global_metrics_fn is not None:
      eval_global_metrics_summary = global_metrics_fn(eval_all_confusion_mats,
                                                      dataset.meta_data)

    ############### LOG EVAL SUMMARY ###############
    eval_summary = train_utils.log_eval_summary(
        step=step,
        eval_metrics=eval_metrics,
        extra_eval_summary=eval_global_metrics_summary,
        writer=writer)
    # Visualize val predictions for one batch:
    if lead_host:
      images = _draw_side_by_side(to_cpu(e_batch), to_cpu(e_predictions))
      example_viz = {
          f'val/example_{i}': image[None, ...] for i, image in enumerate(images)
      }
      writer.write_images(step, example_viz)

    writer.flush()
    del eval_metrics
    return eval_summary

  log_eval_steps = config.get('log_eval_steps') or steps_per_epoch
  if not log_eval_steps:
    raise ValueError("'log_eval_steps' should be specified in the config.")
  log_summary_steps = config.get('log_summary_steps') or log_eval_steps
  checkpoint_steps = config.get('checkpoint_steps') or log_eval_steps

  train_metrics, extra_training_logs = [], []
  train_summary, eval_summary = None, None
  global_metrics_fn = model.get_global_metrics_fn()  # pytype: disable=attribute-error

  chrono = train_utils.Chrono(
      first_step=start_step,
      total_steps=total_steps,
      steps_per_epoch=steps_per_epoch,
      global_bs=config.batch_size,
      accum_train_time=int(jax_utils.unreplicate(train_state.accum_train_time)))

  logging.info('Starting training loop at step %d.', start_step + 1)
  report_progress = periodic_actions.ReportProgress(
      num_train_steps=total_steps, writer=writer)
  hooks = [report_progress]
  if config.get('xprof', True) and lead_host:
    hooks.append(periodic_actions.Profile(num_profile_steps=5, logdir=workdir))

  if start_step == 0:
    step0_log = {'num_trainable_params': num_trainable_params}
    if gflops:
      step0_log['gflops'] = gflops
    writer.write_scalars(1, step0_log)

  # Early stopping flags
  best_opt_accuracy = -1
  best_epoch = 1
  current_epoch=1
  force_out = 0
  early_stopping_patience = config.get('early_stopping_patience') or 20

  for step in range(start_step + 1, total_steps + 1):
    with jax.profiler.StepTraceContext('train', sfLtep_num=step):
      train_batch = next(dataset.train_iter)
      train_state, t_metrics, lr, train_predictions = train_step_pmapped(
          train_state=train_state, batch=train_batch)
      # This will accumulate metrics in TPU memory up to the point that we log
      # them. This is no problem for small metrics but may be a problem for
      # large (e.g. segmentation) metrics. An alternative is to set
      # `log_summary_steps` to a small number, or to use
      # `train_utils.unreplicate_and_get` here instead of right before writing
      # summaries, but that means in each step, we have data transfer between
      # tpu and host, which might slow down the training.
      train_metrics.append(t_metrics)
      # Additional training logs: learning rate:
      extra_training_logs.append({'learning_rate': lr})

    for h in hooks:
      h(step)
    chrono.pause()  # Below are once-in-a-while ops -> pause.

    if step % log_summary_steps == 0 or (step == total_steps):
      ############### LOG TRAIN SUMMARY ###############
      if lead_host:
        chrono.tick(step, writer=writer)
        # Visualize segmentations using side-by-side gt-pred images:
        images = _draw_side_by_side(
            jax.device_get(dataset_utils.unshard(train_batch)),
            jax.device_get(dataset_utils.unshard(train_predictions)))
        example_viz = {
            f'train/example_{i}': image[None, ...]
            for i, image in enumerate(images)
        }
        writer.write_images(step, example_viz)

      train_summary = train_utils.log_train_summary(
          step=step,
          train_metrics=jax.tree_map(train_utils.unreplicate_and_get,
                                     train_metrics),
          extra_training_logs=jax.tree_map(train_utils.unreplicate_and_get,
                                           extra_training_logs),
          writer=writer)

      # Reset metric accumulation for next evaluation cycle.
      train_metrics, extra_training_logs = [], []

    if (step % log_eval_steps == 0) or (step == total_steps):
      with report_progress.timed('eval'):
        # Sync model state across replicas (in case of having model state, e.g.
        # batch statistic when using batch norm).
        train_state = train_utils.sync_model_state_across_replicas(train_state)
        eval_summary = evaluate(train_state, step)

        # here check value
        #current_epoch = int(step // log_eval_steps)
        val_accuracy = eval_summary['accuracy']
        if val_accuracy >= best_opt_accuracy:
          best_epoch = current_epoch
          best_opt_accuracy = val_accuracy
          # best_opt_repl = jax.device_get(opt_repl)
        else:
          logging.info(
                msg=(f'Current val accuracy {val_accuracy} '
                     f'(vs {best_opt_accuracy})'))
          if current_epoch - best_epoch >= early_stopping_patience:
            logging.info(msg='Early stopping, returning best opt!')
            # force checkpoint
            force_out = 1
        current_epoch+=1

    if ((step % checkpoint_steps == 0 and step > 0) or
        (step == total_steps) or (force_out == 1)) and config.checkpoint:
      ################### CHECK POINTING ##########################
      with report_progress.timed('checkpoint'):
        # Sync model state across replicas.
        train_state = train_utils.sync_model_state_across_replicas(train_state)
        if lead_host:
          train_state.replace(  # pytype: disable=attribute-error
              accum_train_time=chrono.accum_train_time)
          train_utils.save_checkpoint(workdir, train_state)

    if force_out == 1:
      # flag turned on due to early stopping
      break

    chrono.resume()  # Un-pause now.

  # Wait until computations are done before exiting.
  jax.random.normal(jax.random.PRNGKey(0), ()).block_until_ready()
  # Return the train and eval summary after last step for regresesion testing.
  return train_state, train_summary, eval_summary


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

"""Custom segmentation_trainer."""

import functools
from typing import Any, Callable, Dict, Optional, Tuple

from absl import logging
from clu import metric_writers
from clu import periodic_actions
from flax import jax_utils
import flax.linen as nn
import jax
from jax.example_libraries.optimizers import clip_grads
from jax.experimental import multihost_utils
import jax.numpy as jnp
import jax.profiler
import ml_collections
import numpy as np
from scenic.dataset_lib import dataset_utils
from scenic.projects.robust_segvit.datasets import cityscapes_variants
from scenic.projects.robust_segvit.datasets import datasets_info
from scenic.train_lib_deprecated import lr_schedules
from scenic.train_lib_deprecated import optimizers
from scenic.train_lib_deprecated import pretrain_utils
from scenic.train_lib_deprecated import train_utils
from scenic.train_lib_deprecated.segmentation_trainer import _draw_side_by_side
from scenic.train_lib_deprecated.segmentation_trainer import get_confusion_matrix
import tensorflow as tf
import eval_utils  # local file import from experimental.robust_segvit
from ensemble_utils import log_average_softmax_probs  # local file import from experimental.robust_segvit
from inference import process_batch  # local file import from experimental.robust_segvit
from ood_metrics import get_ood_metrics  # local file import from experimental.robust_segvit
from pretrainer_utils import convert_torch_to_jax_checkpoint  # local file import from experimental.robust_segvit
from pretrainer_utils import convert_vision_transformer_to_scenic  # local file import from experimental.robust_segvit
from uncertainty_metrics import get_uncertainty_confusion_matrix  # local file import from experimental.robust_segvit
from checkpoint_utils import load_checkpoints_eval
from checkpoint_utils import load_checkpoints_backbone
import h5py
import os
import resource
import sys
import robustness_metrics as rm
from metrics_multihost import ComputeOODAUCMetric, ComputeScoreAUCMetric
from metrics_multihost import host_all_gather_metrics

Batch = Dict[str, jnp.ndarray]
MetricFn = Callable[[jnp.ndarray, Dict[str, jnp.ndarray]],
                    Dict[str, Tuple[float, int]]]
LossFn = Callable[[jnp.ndarray, Batch, Optional[jnp.ndarray]], float]

# JAX team is working on type annotation for pytree:
# https://github.com/google/jax/issues/1555
PyTree = Any


def to_cpu(x, all_gather=False):
  """Send x to cpu.

  All_gather sends all the examples to all the devices.
  Because this consumes a lot of memory, some variables use all_gather while
  other do not.

  Args:
    x : Dict or jnp.array, input so send to cpu.
    all_gather: bool, flag that indicates whether all gather was used on x.

  Returns:
    x: devices.
  """
  if all_gather:
    # num devices x num_examples x h x w x c
    return jax.device_get(dataset_utils.unshard(jax_utils.unreplicate(x)))
  else:
    # num devices x num_examples x h x w x c
    return jax.device_get(dataset_utils.unshard(x))


@functools.partial(jax.pmap, axis_name='x')
def pmap_mean(x: PyTree) -> PyTree:
  # An axis_name is passed to pmap which can then be used by pmean.
  # In this case each device has its own version of the batch statistics and
  # we average them.
  return jax.lax.pmean(x, 'x')


def sync_model_state_across_replicas(
    train_state: train_utils.TrainState) -> train_utils.TrainState:
  """Sync the model_state (like batch statistics) across replicas.

  Edited from scenic/train_lib/train_utils
  Args:
    train_state: TrainState; Current state of training.

  Returns:
    Updated state of training in which model_state is synced across replicas.
  """
  # TODO(dehghani): We simply do "mean" here and this doesn't work with
  #   statistics like variance. (check the discussion in Flax for fixing this).
  if jax.tree_util.tree_leaves(
      train_state.model_state
  ) and 'batch_stats' in train_state.model_state.keys():
    # If the model_state has batch_stats
    new_model_state = train_state.model_state.copy(
        {'batch_stats': pmap_mean(train_state.model_state['batch_stats'])})
    return train_state.replace(  # pytype: disable=attribute-error
        model_state=new_model_state)
  else:
    return train_state


def evaluate(train_state: train_utils.TrainState,
             dataset: Any,
             config: ml_collections.ConfigDict,
             step: int,
             eval_step_pmapped: Any,
             writer: metric_writers.MetricWriter,
             lead_host: Any,
             global_metrics_fn: Any,
             global_unc_metrics_fn: Optional[Any],
             prefix: str = 'valid',
             workdir: str = '',
             ) -> Dict[str, Any]:
  """Model evaluator.

  Args:
    train_state: train state.
    dataset: evaluation dataset.
    config: experiment configuration.
    step: step logged for evaluation.
    eval_step_pmapped: eval state
    writer: CLU metrics writer instance.
    lead_host: Evaluate global metrics on one of the hosts (lead_host) given
      intermediate values collected from all hosts.
    global_metrics_fn: global metrics to evaluate.
    global_unc_metrics_fn: global uncertainty metrics to evaluate.
    prefix: str; Prefix added to the name of the summaries writen by this fctn.

  Returns:
    eval_summary: summary evaluation
  """
  eval_metrics = []
  eval_all_confusion_mats = []
  eval_all_unc_confusion_mats = []

  # Sync model state across replicas.
  train_state = sync_model_state_across_replicas(train_state)

  # Ceil rounding such that we include the last incomplete batch.
  total_eval_steps = int(
      np.ceil(dataset.meta_data['num_eval_examples'] / config.batch_size))
  steps_per_eval = config.get('steps_per_eval') or total_eval_steps

  # Evaluate global metrics on one of the hosts (lead_host), but given
  # intermediate values collected from all hosts.

  # setup calibration evaluation
  ece_num_bins = config.get('ece_num_bins', 15)
  ece_metric = rm.metrics.ExpectedCalibrationError(num_bins=ece_num_bins)._metric
  calib_auc = rm.metrics.CalibrationAUC(correct_pred_as_pos_label=False)._metric

  # store logits
  store_logits = config.eval_configs.get('store_logits', False)

  if store_logits:
    store_logits_fname = os.path.join(workdir, "{}_{}_val.h5py".format(prefix,"logits"))
    f = h5py.File(store_logits_fname, 'w', libver='latest')
    f.swmr_mode = True  # single write multi-read
    input_shape = dataset.meta_data['input_shape'][1:3]
    num_classes = dataset.meta_data['num_classes']
    num_eval_examples = int(steps_per_eval * config.batch_size)
    logits_out = f.create_dataset('logits', (num_eval_examples,) + input_shape + (num_classes,))
    inputs_out = f.create_dataset('inputs', (num_eval_examples,) + input_shape + (3,))
    labels_out = f.create_dataset('labels', (num_eval_examples,) + input_shape)

  for step_ in range(steps_per_eval):
    eval_batch = next(dataset.valid_iter)
    e_batch, e_logits, e_metrics, confusion_matrix, unc_confusion_matrix = eval_step_pmapped(
        train_state=train_state, batch=eval_batch)
    eval_metrics.append(train_utils.unreplicate_and_get(e_metrics))

    probs = jax.nn.softmax(e_logits, axis=-1)
    # updates on each host separately
    ece_metric.update_state(labels=e_batch['label'], probabilities=probs, sample_weight=e_batch['batch_mask'])
    y_pred = jnp.argmax(probs, axis=-1)  # predicted label indices
    confidence = jnp.max(probs, axis=-1)  # confidence score for predicted labels
    calib_auc.update_state(y_true=e_batch['label'], y_pred=y_pred, confidence=confidence, sample_weight=e_batch['batch_mask'])

    if lead_host and global_metrics_fn is not None:
      # Collect data to be sent for computing global metrics.
      eval_all_confusion_mats.append(to_cpu(confusion_matrix, all_gather=True))
      eval_all_unc_confusion_mats.append(
          to_cpu(unc_confusion_matrix, all_gather=True))

      if store_logits:
        start_idx = step_ * config.batch_size
        end_idx = start_idx + config.batch_size
        logits_out[start_idx:end_idx] = e_logits
        inputs_out[start_idx:end_idx] = e_batch['inputs']
        labels_out[start_idx:end_idx] = e_batch['label']

  if store_logits:
    f.close()

  # Compute global metrics
  eval_global_metrics_summary = {}
  if lead_host and global_metrics_fn is not None:
    eval_global_metrics_summary = global_metrics_fn(eval_all_confusion_mats,
                                                    dataset.meta_data)
  if lead_host and global_unc_metrics_fn is not None:
    eval_global_unc_metrics_summary = global_unc_metrics_fn(
        eval_all_unc_confusion_mats)
    eval_global_metrics_summary.update(eval_global_unc_metrics_summary)

  ############### LOG EVAL SUMMARY ###############
  eval_summary = train_utils.log_eval_summary(
      step=step,
      eval_metrics=eval_metrics,
      extra_eval_summary=eval_global_metrics_summary,
      writer=writer,
      prefix=prefix,
      )

  # Gather uncertainty metrics from all hosts and write value:
  ece_metric = host_all_gather_metrics(ece_metric)
  calib_auc = host_all_gather_metrics(calib_auc)
  writer.write_scalars(step=step, scalars={'{}_ece'.format(prefix) : ece_metric.result(),
                                           '{}_calib_auc'.format(prefix): calib_auc.result(),
                                           } )

  # Visualize val predictions for one batch:
  if lead_host:
    # in eval_step we do not use all_gather in batch or logits
    # so the visualization will only include the subset of logits in lead_host
    logits = to_cpu(e_logits)
    e_predictions = jnp.argmax(logits, axis=-1)
    images = _draw_side_by_side(to_cpu(e_batch), e_predictions)
    example_viz = {
        f'{prefix}/example_{i}': image[None, ...]
        for i, image in enumerate(images)
    }
    writer.write_images(step, example_viz)

  writer.flush()

  # Free some memory
  del eval_metrics
  del eval_global_metrics_summary
  del eval_all_confusion_mats
  del eval_all_unc_confusion_mats
  return eval_summary


def evaluate_ood(
    train_state: train_utils.TrainState,
    dataset: Any,
    config: ml_collections.ConfigDict,
    step: int,
    eval_step_pmapped: Any,
    writer: metric_writers.MetricWriter,
    lead_host: Any,
    prefix: str = 'valid',
    workdir: str ='',
    **kwargs,
) -> Dict[str, Any]:
  """Model evaluator.

  Args:
    train_state: train state.
    dataset: evaluation dataset.
    config: experiment configuration.
    step: step logged for evaluation.
    eval_step_pmapped: eval state
    writer: CLU metrics writer instance.
    lead_host: Evaluate global metrics on one of the hosts (lead_host) given
      intermediate values collected from all hosts.
    prefix: str; Prefix added to the name of the summaries writen by this fctn.
    **kwargs: dict; additional parameters for ood evaluation
  Returns:
    eval_summary: summary evaluation
  """
  # Sync model state across replicas.
  train_state = sync_model_state_across_replicas(train_state)

  # Ceil rounding such that we include the last incomplete batch.
  total_eval_steps = int(
      np.ceil(dataset.meta_data['num_eval_examples'] / config.batch_size))
  steps_per_eval = config.get('steps_per_eval') or total_eval_steps

  auc_online = kwargs.pop('auc_online', False)

  # store logits
  store_logits = config.eval_configs.get('store_logits', False)

  if store_logits:
    store_logits_fname = os.path.join(workdir, "{}_{}_val.h5py".format(prefix,"logits"))
    f = h5py.File(store_logits_fname, 'w', libver='latest')
    f.swmr_mode = True  # single write multi-read
    input_shape = dataset.meta_data['input_shape'][1:3]
    num_classes = dataset.meta_data['num_classes']
    num_eval_examples = int(steps_per_eval * config.batch_size)
    logits_out = f.create_dataset('logits', (num_eval_examples,) + input_shape + (num_classes,))
    inputs_out = f.create_dataset('inputs', (num_eval_examples,) + input_shape + (3,))
    labels_out = f.create_dataset('labels', (num_eval_examples,) + input_shape)

  if auc_online:
    # TODO(kellybuchanan): check split of data across devices.
    # initialize metrics: ideally in each device in each host/process/machine
    # keras initializes one metric in each host because it runs in cpu.
    # so we need to convert the function to run metrics in each device/host.
    auc_pr = ComputeOODAUCMetric(curve='PR', num_thresholds=100)
    auc_roc = ComputeOODAUCMetric(curve='ROC', num_thresholds=100)

    # Loop through each machine:
    for step_ in range(steps_per_eval):
      eval_batch = next(dataset.valid_iter)
      e_batch, e_logits = eval_step_pmapped(
          train_state=train_state, batch=eval_batch)

      if store_logits:
        start_idx = step_ * config.batch_size
        end_idx = start_idx + config.batch_size
        logits_out[start_idx:end_idx] = e_logits
        inputs_out[start_idx:end_idx] = e_batch['inputs']
        labels_out[start_idx:end_idx] = e_batch['labels']

      # In eval_step_pmapped we have not used all gather, so each metric is in each device
      # and we should be able to compute metrics in devices separately.
      auc_pr.calculate_and_update_scores(logits=e_logits, label=e_batch['label'],
                                         sample_weight=e_batch['batch_mask'], **kwargs)
      auc_roc.calculate_and_update_scores(logits=e_logits, label=e_batch['label'],
                                         sample_weight=e_batch['batch_mask'], **kwargs)

    if store_logits:
      f.close()

    eval_summary = {'auroc': float(auc_roc.gather_metrics()),
                    'auprc': float(auc_pr.gather_metrics()),
                    }

  else:
    eval_logits = []
    eval_ood_masks = []
    eval_ood_labels = []

    # store all logits locally
    for _ in range(steps_per_eval):
      eval_batch = next(dataset.valid_iter)
      e_batch, e_logits = eval_step_pmapped(
          train_state=train_state, batch=eval_batch)

      # Store all logits in cpu:
      if lead_host:
        e_batch = to_cpu(e_batch, all_gather=False)
        e_logits = to_cpu(e_logits, all_gather=False)

        eval_logits.append(e_logits)
        eval_ood_labels.append(e_batch['label'])
        eval_ood_masks.append(e_batch['batch_mask'])

    # Compute metrics
    eval_logits = jnp.concatenate(eval_logits)
    eval_ood_labels = jnp.concatenate(eval_ood_labels)
    eval_ood_masks = jnp.concatenate(eval_ood_masks)

    eval_summary = get_ood_metrics(
        logits=eval_logits,
        ood_mask=eval_ood_labels,
        weights=eval_ood_masks,
        **kwargs)

  ############### LOG EVAL SUMMARY ###############
  writer.write_scalars(
      step, {
          '_'.join((prefix, key)): val
          for key, val in eval_summary.items()
      })
  # TODO(kellybuchanan): add visualize ood_masks.
  writer.flush()

  return eval_summary


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
    train_state: The state of training including the current global_step,
      model_state, rng, and optimizer. The buffer of this argument can be
      donated to the computation.
    batch: A single batch of data. The buffer of this argument can be donated to
      the computation.
    learning_rate_fn: learning rate scheduler which give the global_step
      generates the learning rate.
    loss_fn: A loss function that given logits, a batch, and parameters of the
      model calculates the loss.
    metrics_fn: A metrics function that given logits and batch of data,
      calculates the metrics as well as the loss.
    config: Configurations of the experiment.
    debug: Whether the debug mode is enabled during training. `debug=True`
      enables model specific logging/storing some values using
      jax.host_callback.

  Returns:
    Updated state of training, computed metrics, learning rate, and predictions
      for logging.
  """
  new_rng, rng = jax.random.split(train_state.rng)

  # Bind the rng to the host/device we are on.
  rng_model_local = jax.random.fold_in(rng, jax.lax.axis_index('batch'))
  rngs = {'dropout': rng_model_local}

  def training_loss_fn(params):
    variables = {'params': params, **train_state.model_state}
    mutable = ['batch_stats'] + list(train_state.model_state.keys())
    (logits, _), new_model_state = flax_model.apply(
        variables,
        batch['inputs'],
        mutable=mutable,
        train=True,
        rngs=rngs,
        debug=debug)

    # logits [batch_size*ens_size x h x w x num_classes]
    if config.model.backbone.get('ens_size', 1) > 1:
      # Given an ensemble, average the loss following:
      # https://github.com/google/uncertainty-baselines/blob/main/baselines/jft/batchensemble.py#L391
      # cannot use pmap (https://github.com/google/jax/discussions/4198)

      # take gradient wrt model params
      # loss_fn will use variables["params"] for an l2 regularization term,
      # given config.l2_decay_factor > 0

      # gp model includes (['params', 'laplace_covariance', 'random_features'])
      # ["random_features"] is fixed
      # ["laplace_covariance"] is not directly trained via gradients
      # so it is closer to a model state, and it is updated in place
      # google3/third_party/py/edward2/jax/nn/random_feature.py;l=283

      ens_logits = jnp.asarray(
          jnp.split(logits, config.model.backbone.ens_size))
      single_loss = jax.vmap(lambda x: loss_fn(x, batch, variables['params']))
      loss = jnp.mean(single_loss(ens_logits))
    else:
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

  if config.get('fast_weight_lr_multiplier', 1.0) != 1.0:
    fast_weights_lr_fn = lambda x: x * config.fast_weight_lr_multiplier
    match_fn = lambda name: ('fast_weight_alpha' in name or 'fast_weight_gamma'  # pylint: disable=g-long-lambda
                             in name)
    grad = optimizers.tree_map_with_names(fast_weights_lr_fn, grad, match_fn)

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

  if config.model.backbone.get('ens_size', 1) > 1:
    # calculate ensemble consensus logits to compute training metrics
    logits = jnp.asarray(jnp.split(logits, config.model.backbone.ens_size))
    logits = log_average_softmax_probs(
        logits)  # batch_size x h x w x num_classes

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
    config: ml_collections.ConfigDict,
    debug: Optional[bool] = False
) -> Tuple[Batch, jnp.ndarray, Dict[str, Tuple[float, int]],
           jnp.ndarray, jnp.ndarray]:
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
    config: Configurations of the experiment.
    debug: Whether the debug mode is enabled during evaluation. `debug=True`
      enables model specific logging/storing some values using
      jax.host_callback.

  Returns:
    Batch, predictions and calculated metrics.
  """
  variables = {
      'params': train_state.optimizer.target,
      **train_state.model_state
  }

  ens_size = config.model.backbone.get('ens_size', 1)
  if config.eval_configs.mode == 'segmm':
    window_size = config.model.input_shape[0]
    window_stride = config.eval_configs.window_stride
    logits = process_batch(
        model=flax_model,
        variables=variables,
        inputs=batch['inputs'],
        window_size=window_size,
        window_stride=window_stride,
        ori_shape=config.dataset_configs.target_size,
        ens_size=ens_size)
  elif config.eval_configs.mode == 'standard':
    (logits, _) = flax_model.apply(
        variables, batch['inputs'], train=False, mutable=False, debug=debug)
  else:
    raise NotImplementedError('Did not implement eval mode {}'.format(
        config.eval_configs.mode))

  if ens_size > 1:
    # calculate ensemble consensus logits to compute eval metrics
    logits = jnp.asarray(jnp.split(logits, config.model.backbone.ens_size))
    logits = log_average_softmax_probs(
        logits)  # batch_size x h x w x num_classes

  metrics = metrics_fn(logits, batch)

  confusion_matrix = get_confusion_matrix(
      labels=batch['label'], logits=logits, batch_mask=batch['batch_mask'])

  unc_confusion_matrix = get_uncertainty_confusion_matrix(
      labels=batch['label'], logits=logits, weights=batch['batch_mask'])

  # Collect predictions and batches from all hosts.
  # use all_gather to copy and replicate across all hosts
  # we skip doing this for batch and logits to save memory
  # unless we want to store the logits
  # predictions = jnp.argmax(logits, axis=-1)
  # predictions = jax.lax.all_gather(predictions, 'batch')
  if config.eval_configs.get('store_logits', False):
    logits = jax.lax.all_gather(logits, 'batch')
    batch = jax.lax.all_gather(batch, 'batch')
  confusion_matrix = jax.lax.all_gather(confusion_matrix, 'batch')
  unc_confusion_matrix = jax.lax.all_gather(unc_confusion_matrix, 'batch')

  return batch, logits, metrics, confusion_matrix, unc_confusion_matrix


def eval_step_baseline(
    *,
    flax_model: nn.Module,
    train_state: train_utils.TrainState,
    batch: Batch,
    config: ml_collections.ConfigDict,
    debug: Optional[bool] = False
) -> Tuple[Batch, jnp.ndarray]:
  """Runs a single eval step.

  Args:
    flax_model: A Flax model.
    train_state: TrainState, the state of training including the current
      global_step, model_state, rng, and optimizer. The buffer of this argument
      can be donated to the computation.
    batch: A single batch of data. a metrics function, that given logits and
      batch of data, calculates the metrics as well as the loss.
    config: Configurations of the experiment.
    debug: Whether the debug mode is enabled during evaluation. `debug=True`
      enables model specific logging/storing some values using
      jax.host_callback.

  Returns:
    Batch, predictions and calculated metrics.
  """
  variables = {
      'params': train_state.optimizer.target,
      **train_state.model_state
  }

  ens_size = config.model.backbone.get('ens_size', 1)
  if config.eval_configs.mode == 'segmm':
    window_size = config.model.input_shape[0]
    window_stride = config.eval_configs.window_stride
    logits = process_batch(
        model=flax_model,
        variables=variables,
        inputs=batch['inputs'],
        window_size=window_size,
        window_stride=window_stride,
        ori_shape=config.dataset_configs.target_size,
        ens_size=ens_size)
  elif config.eval_configs.mode == 'standard':
    (logits, _) = flax_model.apply(
        variables, batch['inputs'], train=False, mutable=False, debug=debug)
  else:
    raise NotImplementedError('Did not implement eval mode {}'.format(
        config.eval_configs.mode))

  if ens_size > 1:
    # calculate ensemble consensus logits to compute eval metrics
    logits = jnp.asarray(jnp.split(logits, config.model.backbone.ens_size))
    logits = log_average_softmax_probs(
        logits)  # batch_size x h x w x num_classes

  # Collect predictions and batches from all hosts.
  # use all_gather to copy and replicate across all hosts
  # we can skip doing this for batch and logits to save memory
  # jis the OOM in tpu or cpu?
  if config.eval_configs.get('store_logits', False):
    batch = jax.lax.all_gather(batch, 'batch')
    logits = jax.lax.all_gather(logits, 'batch')

  return batch, logits


def train(
    *,
    rng: jnp.ndarray,
    config: ml_collections.ConfigDict,
    model_cls: Any,
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
  - Updated from scenic.train_lib_deprecated.segmentation_trainer.train
  """
  lead_host = jax.process_index() == 0
  # Build the loss_fn, metrics, and flax_model.
  model = model_cls(config, dataset.meta_data)

  # Model input shape
  input_shape = config.model.get('input_shape',
                                 dataset.meta_data['input_shape'])

  if len(input_shape) == 2:
    input_shape = (-1, *input_shape, 3)

  # Initialize model.
  rng, init_rng = jax.random.split(rng)

  if config.model.decoder.type == 'het':
    keys = [
        'dropout', 'diag_noise_samples', 'standard_norm_noise_samples', 'params'
    ]
    init_rng = dict(zip(keys, jax.random.split(init_rng, 4)))

  (params, model_state, num_trainable_params,
   gflops) = train_utils.initialize_model(
       model_def=model.flax_model,
       input_spec=[(input_shape,
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
    train_state = load_checkpoints_backbone(config, model, train_state, workdir)
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
          config=config,
          debug=config.debug_eval),
      axis_name='batch',
      # We can donate the eval_batch's buffer.
  )

  log_eval_steps = config.get('log_eval_steps') or steps_per_epoch
  if not log_eval_steps:
    raise ValueError("'log_eval_steps' should be specified in the config.")
  log_summary_steps = config.get('log_summary_steps') or log_eval_steps
  checkpoint_steps = config.get('checkpoint_steps') or log_eval_steps

  train_metrics, extra_training_logs = [], []
  train_summary, eval_summary = {}, {}
  global_metrics_fn = model.get_global_metrics_fn()  # pytype: disable=attribute-error
  global_unc_metrics_fn = model.get_global_unc_metrics_fn()  # pytype: disable=attribute-error

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

  # Early stopping flags (not necessary when we use wandb)
  best_opt_accuracy = -1
  best_epoch = 1
  current_epoch = 0
  force_out = 0
  early_stopping_patience = config.get('early_stopping_patience', 100)

  for step in range(start_step + 1, total_steps + 1):
    with jax.profiler.StepTraceAnnotation('train', sfLtep_num=step):
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
        train_state = sync_model_state_across_replicas(train_state)

        # Eval model:
        eval_summary = evaluate(train_state=train_state,
                                dataset=dataset,
                                config=config,
                                step=step,
                                eval_step_pmapped=eval_step_pmapped,
                                writer=writer,
                                lead_host=lead_host,
                                global_metrics_fn=global_metrics_fn,
                                global_unc_metrics_fn=global_unc_metrics_fn,
                                workdir=workdir,
                                )

        # check accuracy for early stopping.
        val_accuracy = eval_summary['accuracy']
        if val_accuracy >= best_opt_accuracy:
          best_epoch = current_epoch
          best_opt_accuracy = val_accuracy
        else:
          logging.info(
              msg=(f'Current val accuracy {val_accuracy} '
                   f'(vs {best_opt_accuracy})'))
          if current_epoch - best_epoch >= early_stopping_patience:
            logging.info(msg='Early stopping, returning best opt!')
            # force checkpoint
            force_out = 1
        current_epoch += 1

    if ((step % checkpoint_steps == 0 and step > 0) or (step == total_steps) or
        (force_out == 1)) and config.checkpoint:
      ################### CHECK POINTING ##########################
      with report_progress.timed('checkpoint'):
        # Sync model state across replicas.
        train_state = sync_model_state_across_replicas(train_state)
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

  del dataset

  # ----------------------------------------------------------------------------
  # Evaluate OOD datasets
  eval_summary_ood = evaluate_ood_step(
      train_state=train_state,
      config=config,
      rng=rng,
      model=model,
      lead_host=lead_host,
      writer=writer,
      workdir=workdir,
  )

  eval_summary.update(eval_summary_ood)

  # Return the train and eval summary after last step for testing.
  return train_state, train_summary, eval_summary


def eval_ckpt(
    *,
    rng: jnp.ndarray,
    config: ml_collections.ConfigDict,
    model_cls: Any,
    dataset: dataset_utils.Dataset,
    workdir: str,
    writer: metric_writers.MetricWriter,
) -> Tuple[train_utils.TrainState, Any, Dict[str, Any]]:
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

  logging.info('Running eval code')

  lead_host = jax.process_index() == 0
  # Build the loss_fn, metrics, and flax_model.
  model = model_cls(config, dataset.meta_data)

  # Model input shape
  input_shape = config.model.get('input_shape',
                                 dataset.meta_data['input_shape'])

  if len(input_shape) == 2:
    input_shape = (-1, *input_shape, 3)

  # Initialize model.
  rng, init_rng = jax.random.split(rng)

  if config.model.decoder.type == 'het':
    keys = [
        'dropout', 'diag_noise_samples', 'standard_norm_noise_samples', 'params'
    ]
    init_rng = dict(zip(keys, jax.random.split(init_rng, 4)))

  (params, model_state, _, _) = train_utils.initialize_model(
      model_def=model.flax_model,
      input_spec=[(input_shape, dataset.meta_data.get('input_dtype',
                                                      jnp.float32))],
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

  # Load checkpoint
  checkpoint_configs = config.get('checkpoint_configs', False)

  if checkpoint_configs:
    train_state = load_checkpoints_eval(config, model, train_state, workdir)
  else:
    logging.info('Not loading any checkpoints')

  # Replicate the optimizer, state, and rng.
  train_state = jax_utils.replicate(train_state)
  del params  # Do not keep a copy of the initial params.

  ############### EVALUATION CODE #################

  eval_step_pmapped = jax.pmap(
      functools.partial(
          eval_step,
          flax_model=model.flax_model,
          metrics_fn=model.get_metrics_fn('validation'),
          config=config,
          debug=config.debug_eval),
      axis_name='batch',
      # We can donate the eval_batch's buffer.
  )

  eval_summary = None
  global_metrics_fn = model.get_global_metrics_fn()  # pytype: disable=attribute-error
  global_unc_metrics_fn = model.get_global_unc_metrics_fn()  # pytype: disable=attribute-error

  # Eval model:
  prefix = dataset.meta_data.get('prefix', 'valid')
  eval_summary = evaluate(train_state=train_state,
                          dataset=dataset,
                          config=config,
                          step=0,
                          eval_step_pmapped=eval_step_pmapped,
                          writer=writer,
                          lead_host=lead_host,
                          global_metrics_fn=global_metrics_fn,
                          global_unc_metrics_fn=global_unc_metrics_fn,
                          prefix=prefix,
                          workdir=workdir,
                          )

  # Wait until computations are done before running robustness evaluator.
  jax.random.normal(jax.random.PRNGKey(0), ()).block_until_ready()
  del dataset

  # ----------------------------------------------------------------------------
  # Evaluate OOD datasets
  logging.info('Evaluating OOD datasets')
  eval_summary_ood = evaluate_ood_step(
      train_state=train_state,
      config=config,
      rng=rng,
      model=model,
      lead_host=lead_host,
      writer=writer,
      workdir=workdir,
  )

  eval_summary.update(eval_summary_ood)

  # Return the train and eval summary after last step for testing.
  return train_state, _, eval_summary


def evaluate_ood_step(
    *,
    train_state: train_utils.TrainState,
    config: ml_collections.ConfigDict,
    rng: jnp.ndarray,
    model: Any,
    lead_host: Any,
    writer: metric_writers.MetricWriter,
    workdir: str,
) -> Dict[str, Any]:
  """OOD evaluation given loaded model.

  The datasets are loaded given for each type of corruption given the flags.

  Args:
    train_state: train state.
    config: experiment configuration.
    rng: jax rng.
    model: model with loaded checkpoint.
    lead_host: Evaluate global metrics on one of the hosts (lead_host) given
    writer: CLU metrics writer instance.
      intermediate values collected from all hosts.
    workdir: Directory where to store outputs.
  Returns:
    eval_summary: summary evaluation
  """
  eval_summary = {}

  if config.get('eval_covariate_shift', False):

    eval_step_pmapped = jax.pmap(
        functools.partial(
            eval_step,
            flax_model=model.flax_model,
            metrics_fn=model.get_metrics_fn('validation'),
            config=config,
            debug=config.debug_eval),
        axis_name='batch',
        # We can donate the eval_batch's buffer.
    )

    global_metrics_fn = model.get_global_metrics_fn()  # pytype: disable=attribute-error
    global_unc_metrics_fn = model.get_global_unc_metrics_fn()  # pytype: disable=attribute-error

    eval_ood_covariate = {'cityscapes_c': evaluate_cityscapes_c,
                          'ade20k_ind_c': evaluate_ade20k_corrupted,}

    # TODO(kellybuchanan): merge data sources.
    # The form of the ind dataset name depends on the source of the data.
    ind_names = [
        config.dataset_name,
        config.dataset_configs.get('dataset_name', ''),
        config.dataset_configs.get('name', '')
    ]

    if any('cityscapes' in ind_name for ind_name in ind_names):
      logging.info('Loading Cityscapes_c...')
      ood_dataset = 'cityscapes_c'
    elif any('ade20k' in ind_name for ind_name in ind_names):
      logging.info('Loading Ade20k_ind_c')
      ood_dataset = 'ade20k_ind_c'
    else:
      logging.info('OOD Covariate shift dataset is not implemented')
      ood_dataset = None

    if ood_dataset:
      eval_summary = eval_ood_covariate[ood_dataset](
          train_state=train_state,
          config=config,
          rng=rng,
          eval_step_pmapped=eval_step_pmapped,
          writer=writer,
          lead_host=lead_host,
          global_metrics_fn=global_metrics_fn,
          global_unc_metrics_fn=global_unc_metrics_fn,
          workdir=workdir,
      )

      # Wait until computations are done before exiting.
      jax.random.normal(jax.random.PRNGKey(0), ()).block_until_ready()

  # ----------------------------------------------------------------------------
  if config.get('eval_label_shift', False):

    eval_step_ood_pmapped = jax.pmap(
        functools.partial(
            eval_step_baseline,
            flax_model=model.flax_model,
            config=config,
            debug=config.debug_eval),
        axis_name='batch',
        # We can donate the eval_batch's buffer.
    )

    eval_label_shift = {
        'fishyscapes': evaluate_fishyscapes,
        'ade20k_ood_open': evaluate_ade20k_ood_open,
        'street_hazards_ood_open': evaluate_street_hazards_ood_open,
    }

    # The form of the ind dataset name depends on the source of the data.
    ind_names = [
        config.dataset_name,
        config.dataset_configs.get('dataset_name', ''),
        config.dataset_configs.get('name', '')
    ]

    if any('cityscapes' in ind_name for ind_name in ind_names):
      logging.info('Loading Fishyscapes...')
      ood_dataset = 'fishyscapes'
    elif any('ade20k' in ind_name for ind_name in ind_names):
      logging.info('Loading ADE20k OOD OPEN...')
      ood_dataset = 'ade20k_ood_open'
    elif any('street_hazards' in ind_name for ind_name in ind_names):
      logging.info('Loading StreetHazards OPEN...')
      ood_dataset = 'street_hazards_ood_open'
    else:
      logging.info('OOD Label shift dataset is not implemented')
      ood_dataset = None

    if ood_dataset:
      eval_summary = eval_label_shift[ood_dataset](
          train_state=train_state,
          config=config,
          rng=rng,
          eval_step_pmapped=eval_step_ood_pmapped,
          writer=writer,
          lead_host=lead_host,
          workdir=workdir,
      )

      # Wait until computations are done before exiting.
      jax.random.normal(jax.random.PRNGKey(0), ()).block_until_ready()
  return eval_summary


def evaluate_cityscapes_c(
    train_state: train_utils.TrainState,
    config: ml_collections.ConfigDict,
    rng: Any,
    eval_step_pmapped: Any,
    writer: metric_writers.MetricWriter,
    lead_host: Any,
    global_metrics_fn: Any,
    global_unc_metrics_fn: Any,
    workdir: str = None,
) -> Dict[str, Any]:
  """Evaluate cityscapes-c dataset.

  Args:
    train_state: train state.
    config: experiment configuration.
    rng: jax rng.
    eval_step_pmapped: eval state
    writer: CLU metrics writer instance.
    lead_host: Evaluate global metrics on one of the hosts (lead_host) given
      intermediate values collected from all hosts.
    global_metrics_fn: global metrics to evaluate.
    global_unc_metrics_fn: global uncertainty metrics to evaluate.
  Returns:
    eval_summary: summary evaluation
  """
  # Load cityscapes-c datasets
  # set resource limit to debug in mac osx
  # (see https://github.com/tensorflow/datasets/issues/1441)
  if jax.process_index() == 0 and sys.platform == 'darwin':
   low, high = resource.getrlimit(resource.RLIMIT_NOFILE)
   resource.setrlimit(resource.RLIMIT_NOFILE, (low, high))

  # update config:
  ood_config = ml_collections.ConfigDict()
  ood_config.update(**config)
  ood_config.update({'dataset_name': 'cityscapes_variants'})

  accuracy_per_corruption = {}
  prefix = 'citycvalid'
  for corruption in cityscapes_variants.CITYSCAPES_C_CORRUPTIONS:
    local_list = []  # list to compute macro average per corruption
    for severity in cityscapes_variants.CITYSCAPES_C_SEVERITIES:

      with ood_config.unlocked():
        ood_config.dataset_configs.dataset_name = f'cityscapes_corrupted/semantic_segmentation_{corruption}_{severity}'

      rng, data_rng = jax.random.split(rng)
      dataset = train_utils.get_dataset(ood_config, data_rng)
      dataset.meta_data['dataset_name'] = 'cityscapes_c'
      dataset.meta_data['prefix'] = prefix + f'_{corruption}_{severity}'

      eval_summary = evaluate(
          train_state=train_state,
          dataset=dataset,
          config=ood_config,
          step=0,
          eval_step_pmapped=eval_step_pmapped,
          writer=writer,
          lead_host=lead_host,
          global_metrics_fn=global_metrics_fn,
          global_unc_metrics_fn=global_unc_metrics_fn,
          prefix=dataset.meta_data['prefix'],
          workdir=workdir,
      )

      local_list.append(eval_summary)

    accuracy_per_corruption[corruption] = eval_utils.average_list_of_dicts(
        local_list)

  cityscapes_c_metrics = eval_utils.average_list_of_dicts(
      accuracy_per_corruption.values())

  # append name to metrics
  key_separator = '_'
  avg_cityscapes_c_metrics = {
      key_separator.join((prefix, key)): val
      for key, val in cityscapes_c_metrics.items()
  }
  # update metrics
  eval_summary.update(avg_cityscapes_c_metrics)
  writer.write_scalars(0, avg_cityscapes_c_metrics)
  writer.flush()
  return eval_summary


def evaluate_fishyscapes(
    train_state: train_utils.TrainState,
    config: ml_collections.ConfigDict,
    rng: Any,
    eval_step_pmapped: Any,
    writer: metric_writers.MetricWriter,
    lead_host: Any,
    workdir: str = '',
) -> Dict[str, Any]:
  """Evaluate Fishyscapes dataset.

  Args:
    train_state: train state.
    config: experiment configuration.
    rng: jax rng.
    eval_step_pmapped: eval state
    writer: CLU metrics writer instance.
    lead_host: Evaluate global metrics on one of the hosts (lead_host) given
      intermediate values collected from all hosts.
  Returns:
    eval_summary: summary evaluation
  """
  # Load Fishyscapes datasets
  # set resource limit to debug in mac osx
  # (see https://github.com/tensorflow/datasets/issues/1441)
  if jax.process_index() == 0 and sys.platform == 'darwin':
   low, high = resource.getrlimit(resource.RLIMIT_NOFILE)
   resource.setrlimit(resource.RLIMIT_NOFILE, (low, high))

  # update config:
  ood_config = ml_collections.ConfigDict()
  ood_config.update(**config)
  ood_config.update({'dataset_name': 'cityscapes_variants'})

  device_count = jax.device_count()
  accuracy_per_corruption = {}
  prefix = 'fishyvalid'
  for corruption in cityscapes_variants.FISHYSCAPES_CORRUPTIONS:

    with ood_config.unlocked():
      ood_config.dataset_configs.dataset_name = f'fishyscapes/{corruption}'
      ood_config.batch_size = device_count

    data_rng, rng = jax.random.split(rng)
    dataset = train_utils.get_dataset(ood_config, data_rng)
    dataset.meta_data['dataset_name'] = 'fishyscapes'
    dataset.meta_data['prefix'] = prefix + f'_{corruption}'

    eval_summary = evaluate_ood(
        train_state=train_state,
        dataset=dataset,
        config=ood_config,
        step=0,
        eval_step_pmapped=eval_step_pmapped,
        writer=writer,
        lead_host=lead_host,
        prefix=dataset.meta_data['prefix'],
        workdir=workdir,
        **config.get('eval_robustness_configs', {}),
    )

    accuracy_per_corruption[corruption] = eval_summary

  fishyscapes_metrics = eval_utils.average_list_of_dicts(
      accuracy_per_corruption.values())

  # append name to metrics
  key_separator = '_'
  avg_fishyscapes_metrics = {
      key_separator.join((prefix, key)): val
      for key, val in fishyscapes_metrics.items()
  }
  # update metrics
  eval_summary.update(avg_fishyscapes_metrics)
  writer.write_scalars(0, avg_fishyscapes_metrics)
  writer.flush()
  return eval_summary


def evaluate_ade20k_ood_open(
    train_state: train_utils.TrainState,
    config: ml_collections.ConfigDict,
    rng: Any,
    eval_step_pmapped: Any,
    writer: metric_writers.MetricWriter,
    lead_host: Any,
    workdir: str = '',
) -> Dict[str, Any]:
  """Evaluate ADE20k OOD dataset.

  Args:
    train_state: train state.
    config: experiment configuration.
    rng: jax rng.
    eval_step_pmapped: eval state
    writer: CLU metrics writer instance.
    lead_host: Evaluate global metrics on one of the hosts (lead_host) given
      intermediate values collected from all hosts.

  Returns:
    eval_summary: summary evaluation
  """
  # set resource limit to debug in mac osx
  # (see https://github.com/tensorflow/datasets/issues/1441)
  if jax.process_index() == 0 and sys.platform == 'darwin':
   low, high = resource.getrlimit(resource.RLIMIT_NOFILE)
   resource.setrlimit(resource.RLIMIT_NOFILE, (low, high))

  # update config:
  ood_config = ml_collections.ConfigDict()
  ood_config.update(**config)
  ood_config.update({'dataset_name': 'robust_segvit_segmentation'})

  device_count = jax.device_count()
  prefix = 'ade20k_ood_open'

  with ood_config.unlocked():
    ood_config.dataset_configs.name = 'ade20k_ood_open'
    ood_config.batch_size = device_count

  data_rng, rng = jax.random.split(rng)
  dataset = train_utils.get_dataset(ood_config, data_rng)
  dataset.meta_data['prefix'] = prefix

  eval_summary = evaluate_ood(
      train_state=train_state,
      dataset=dataset,
      config=ood_config,
      step=0,
      eval_step_pmapped=eval_step_pmapped,
      writer=writer,
      lead_host=lead_host,
      prefix=dataset.meta_data['prefix'],
      workdir=workdir,
      **config.get('eval_robustness_configs', {}),
  )

  # append name to metrics
  key_separator = '_'
  avg_open_set_metrics = {
      key_separator.join((prefix, key)): val
      for key, val in eval_summary.items()
  }
  # update metrics
  eval_summary.update(avg_open_set_metrics)
  writer.write_scalars(0, avg_open_set_metrics)
  writer.flush()

  return eval_summary


def evaluate_ade20k_corrupted(
    train_state: train_utils.TrainState,
    config: ml_collections.ConfigDict,
    rng: Any,
    eval_step_pmapped: Any,
    writer: metric_writers.MetricWriter,
    lead_host: Any,
    global_metrics_fn: Any,
    global_unc_metrics_fn: Any,
    workdir : str,
) -> Dict[str, Any]:
  """Evaluate Ade20k-C dataset.

  Args:
    train_state: train state.
    config: experiment configuration.
    rng: jax rng.
    eval_step_pmapped: eval state
    writer: CLU metrics writer instance.
    lead_host: Evaluate global metrics on one of the hosts (lead_host) given
      intermediate values collected from all hosts.
    global_metrics_fn: global metrics to evaluate.
    global_unc_metrics_fn: global uncertainty metrics to evaluate.
  Returns:
    eval_summary: summary evaluation
  """
  # Load Ade20k-C dataset
  # set resource limit to debug in mac osx
  # (see https://github.com/tensorflow/datasets/issues/1441)
  if jax.process_index() == 0 and sys.platform == 'darwin':
   low, high = resource.getrlimit(resource.RLIMIT_NOFILE)
   resource.setrlimit(resource.RLIMIT_NOFILE, (low, high))

  # update config:
  ood_config = ml_collections.ConfigDict()
  ood_config.update(**config)
  ood_config.update({'dataset_name': 'robust_segvit_variants'})

  # Calculate metrics per corruption.
  accuracy_per_corruption = {}
  prefix = 'ade20k_ind_c'
  for corruption in datasets_info.ADE20K_C_CORRUPTIONS:
    local_list = []  # list to compute macro average per corruption
    for severity in range(1, 6):

      with ood_config.unlocked():
        ood_config.dataset_configs.name = f'ade20k_ind_c_{corruption}_{severity}'

      data_rng, rng = jax.random.split(rng)
      dataset = train_utils.get_dataset(ood_config, data_rng)
      dataset.meta_data['prefix'] = prefix + f'_{corruption}_{severity}'

      eval_summary = evaluate(
          train_state=train_state,
          dataset=dataset,
          config=ood_config,
          step=0,
          eval_step_pmapped=eval_step_pmapped,
          writer=writer,
          lead_host=lead_host,
          global_metrics_fn=global_metrics_fn,
          global_unc_metrics_fn=global_unc_metrics_fn,
          prefix=dataset.meta_data['prefix'],
          workdir=workdir,
      )

      local_list.append(eval_summary)

    accuracy_per_corruption[corruption] = eval_utils.average_list_of_dicts(
        local_list)

  ade20k_c_metrics = eval_utils.average_list_of_dicts(
      accuracy_per_corruption.values())

  # append name to metrics
  key_separator = '_'
  avg_corrupted_metrics = {
      key_separator.join((prefix, key)): val
      for key, val in ade20k_c_metrics.items()
  }
  # update metrics
  eval_summary.update(avg_corrupted_metrics)
  writer.write_scalars(0, avg_corrupted_metrics)
  writer.flush()
  return eval_summary


def evaluate_street_hazards_ood_open(
    train_state: train_utils.TrainState,
    config: ml_collections.ConfigDict,
    rng: Any,
    eval_step_pmapped: Any,
    writer: metric_writers.MetricWriter,
    lead_host: Any,
    workdir: str,
) -> Dict[str, Any]:
  """Evaluate StreetHazards OOD dataset.

  Args:
    train_state: train state.
    config: experiment configuration.
    rng: jax rng.
    eval_step_pmapped: eval state
    writer: CLU metrics writer instance.
    lead_host: Evaluate global metrics on one of the hosts (lead_host) given
      intermediate values collected from all hosts.

  Returns:
    eval_summary: summary evaluation
  """
  # set resource limit to debug in mac osx
  # (see https://github.com/tensorflow/datasets/issues/1441)
  if jax.process_index() == 0 and sys.platform == 'darwin':
   low, high = resource.getrlimit(resource.RLIMIT_NOFILE)
   resource.setrlimit(resource.RLIMIT_NOFILE, (low, high))

  # update config:
  ood_config = ml_collections.ConfigDict()
  ood_config.update(**config)
  ood_config.update({'dataset_name': 'robust_segvit_segmentation'})

  device_count = jax.device_count()
  prefix = 'street_hazards_open'

  with ood_config.unlocked():
    ood_config.dataset_configs.name = 'street_hazards_open'
    ood_config.batch_size = device_count

  data_rng, rng = jax.random.split(rng)
  dataset = train_utils.get_dataset(ood_config, data_rng)
  dataset.meta_data['prefix'] = prefix

  eval_summary = evaluate_ood(
      train_state=train_state,
      dataset=dataset,
      config=ood_config,
      step=0,
      eval_step_pmapped=eval_step_pmapped,
      writer=writer,
      lead_host=lead_host,
      prefix=dataset.meta_data['prefix'],
      workdir=workdir,
      **config.get('eval_robustness_configs', {}),
  )

  # append name to metrics
  key_separator = '_'
  avg_open_set_metrics = {
      key_separator.join((prefix, key)): val
      for key, val in eval_summary.items()
  }
  # update metrics
  eval_summary.update(avg_open_set_metrics)
  writer.write_scalars(0, avg_open_set_metrics)
  writer.flush()

  return eval_summary


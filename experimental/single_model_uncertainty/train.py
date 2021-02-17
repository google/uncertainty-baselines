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

# Lint as: python3
"""Basic training loop example for Uncertainty Baselines."""

import os.path

from typing import Any, Callable, Dict, Iterator, Optional
from absl import logging
import tensorflow.compat.v2 as tf
import uncertainty_baselines as ub
import eval as eval_lib  # local file import
import loss_util as loss_lib  # local file import
from tensorboard.plugins.hparams import api as hp

_TensorDict = Dict[str, tf.Tensor]
_TrainStepFn = Callable[[Iterator[_TensorDict]], _TensorDict]


def _train_step_fn(model: tf.keras.Model,
                   optimizer: tf.keras.optimizers.Optimizer,
                   strategy: tf.distribute.Strategy,
                   metrics: Dict[str, tf.keras.metrics.Metric],
                   iterations_per_loop: int,
                   focal_loss_gamma: float) -> _TrainStepFn:
  """Return a function to run `iterations_per_loop` train steps."""

  # Note that train_iterator should return batches with the global batch size
  # (num_devices * per_core_batch_size).
  @tf.function
  def train_step(train_iterator: Iterator[_TensorDict]) -> _TensorDict:
    def step(per_replica_inputs: _TensorDict) -> None:
      """The function defining a single training step."""
      features = per_replica_inputs['features']
      labels = per_replica_inputs['labels']

      with tf.GradientTape() as tape:
        logits = model(features, training=True)
        if isinstance(logits, (tuple, list)):
          # If model returns a tuple of (logits, covmat), extract logits
          logits, _ = logits
        if focal_loss_gamma > 0.0:
          loss = loss_lib.compute_focal_loss(
              labels, logits, gamma=focal_loss_gamma)
        else:
          loss = tf.reduce_mean(
              tf.keras.losses.sparse_categorical_crossentropy(
                  y_true=labels, y_pred=logits, from_logits=True))

        regularization_losses = model.get_losses_for(inputs=None)
        if regularization_losses:
          loss += tf.reduce_sum(regularization_losses)
        # Even though features/labels are the per-core batch size, we divide the
        # loss by the number of replicas here because we average the loss and
        # gradients within each batch but then thew ill be SUMMED across
        # replicas, so this division turns the second sum into a mean. See
        # https://www.tensorflow.org/tutorials/distribute/custom_training#define_the_loss_function.
        # This assumes that the per-core batch size is the same for each replica
        # and step, which will be the case because we use padding.
        scaled_loss = loss / strategy.num_replicas_in_sync

      predictions = tf.nn.softmax(logits, axis=-1)
      for metric in metrics.values():
        metric.update_state(labels, predictions)
      grads = tape.gradient(scaled_loss, model.trainable_variables)
      optimizer.apply_gradients(list(zip(grads, model.trainable_variables)))
      return

    for metric in metrics.values():
      metric.reset_states()
    # Following the recommendation here, run multiple steps inside this training
    # function wrapped in tf.function for better TPU utilization:
    # https://www.kaggle.com/c/flower-classification-with-tpus/discussion/135443.
    for _ in tf.range(iterations_per_loop):  # Note the use of tf.range.
      ub.utils.call_step_fn(strategy, step, next(train_iterator))
    return {name: metric.result() for name, metric in metrics.items()}

  return train_step


def _write_summaries(
    train_step_outputs: Dict[str, Any],
    current_step: int,
    train_summary_writer: tf.summary.SummaryWriter,
    hparams: Optional[Dict[str, Any]] = None) -> None:
  """Log metrics every using tf.summary."""
  with train_summary_writer.as_default():
    if hparams:
      hp.hparams(hparams)
    for name, result in train_step_outputs.items():
      tf.summary.scalar(name, result, step=current_step)


def run_train_loop(
    train_dataset_builder: ub.datasets.BaseDataset,
    validation_dataset_builder: Optional[ub.datasets.BaseDataset],
    test_dataset_builder: ub.datasets.BaseDataset,
    batch_size: int,
    eval_batch_size: int,
    model: tf.keras.Model,
    optimizer: tf.keras.optimizers.Optimizer,
    eval_frequency: int,
    log_frequency: int,
    trial_dir: str,
    train_steps: int,
    mode: str,
    strategy: tf.distribute.Strategy,
    metrics: Dict[str, tf.keras.metrics.Metric],
    hparams: Dict[str, Any],
    ood_dataset_builder: ub.datasets.BaseDataset = None,
    ood_metrics: Dict[str, tf.keras.metrics.Metric] = None,
    focal_loss_gamma: float = 0.0,
    mean_field_factor: float = -1):
  """Train, possibly evaluate the model, and record metrics."""

  checkpoint_manager = None
  last_checkpoint_step = 0
  if trial_dir:
    # TODO(znado): add train_iterator to this once DistributedIterators are
    # checkpointable.
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
    checkpoint_manager = tf.train.CheckpointManager(
        checkpoint, trial_dir, max_to_keep=None)
    checkpoint_path = tf.train.latest_checkpoint(trial_dir)
    if checkpoint_path:
      last_checkpoint_step = int(checkpoint_path.split('-')[-1])
      if last_checkpoint_step >= train_steps:
        # If we have already finished training, exit.
        logging.info(
            'Training has already finished at step %d. Exiting.', train_steps)
        return
      elif last_checkpoint_step > 0:
        # Restore from where we previously finished.
        checkpoint.restore(checkpoint_manager.latest_checkpoint)
        logging.info('Resuming training from step %d.', last_checkpoint_step)

  train_dataset = train_dataset_builder.load(batch_size=batch_size)
  train_dataset = strategy.experimental_distribute_dataset(train_dataset)
  train_iterator = iter(train_dataset)

  iterations_per_loop = min(eval_frequency, log_frequency)
  # We can only run `iterations_per_loop` steps at a time, because we cannot
  # checkpoint the model inside a tf.function.
  train_step_fn = _train_step_fn(
      model,
      optimizer,
      strategy,
      metrics,
      iterations_per_loop=iterations_per_loop,
      focal_loss_gamma=focal_loss_gamma)
  train_summary_writer = tf.summary.create_file_writer(
      os.path.join(trial_dir, 'train'))

  (test_fn, test_dataset, test_summary_writer, val_fn, val_dataset,
   val_summary_writer, ood_fn, ood_dataset,
   ood_summary_writer) = eval_lib.setup_eval(
       validation_dataset_builder=validation_dataset_builder,
       test_dataset_builder=test_dataset_builder,
       batch_size=eval_batch_size,
       strategy=strategy,
       trial_dir=trial_dir,
       model=model,
       metrics=metrics,
       ood_dataset_builder=ood_dataset_builder,
       ood_metrics=ood_metrics,
       mean_field_factor=mean_field_factor)

  # Each call to train_step_fn will run iterations_per_loop steps.
  num_train_fn_steps = train_steps // iterations_per_loop
  # We are guaranteed that `last_checkpoint_step` will be divisible by
  # `iterations_per_loop` because that is how frequently we checkpoint.
  start_train_fn_step = last_checkpoint_step // iterations_per_loop
  for train_fn_step in range(start_train_fn_step, num_train_fn_steps):
    current_step = train_fn_step * iterations_per_loop
    # Checkpoint at the start of the step, before the training op is run.
    if checkpoint_manager and current_step % eval_frequency == 0:
      checkpoint_manager.save(checkpoint_number=current_step)
    if 'eval' in mode and current_step % eval_frequency == 0:
      eval_lib.run_eval_epoch(
          current_step,
          test_fn,
          test_dataset,
          test_summary_writer,
          val_fn,
          val_dataset,
          val_summary_writer,
          ood_fn,
          ood_dataset,
          ood_summary_writer)  # Only write hparams on the last step.
    train_step_outputs = train_step_fn(train_iterator)
    if current_step % log_frequency == 0:
      _write_summaries(train_step_outputs, current_step, train_summary_writer)
      train_step_outputs_np = {
          k: v.numpy() for k, v in train_step_outputs.items()
      }
      logging.info(
          'Training metrics for step %d: %s',
          current_step,
          train_step_outputs_np)

  if train_steps % iterations_per_loop != 0:
    remainder_train_step_fn = _train_step_fn(
        model,
        optimizer,
        strategy,
        metrics,
        iterations_per_loop=train_steps % iterations_per_loop,
        focal_loss_gamma=focal_loss_gamma)
    train_step_outputs = remainder_train_step_fn(train_iterator)

  # Always evaluate and record metrics at the end of training.
  _write_summaries(
      train_step_outputs, train_steps, train_summary_writer, hparams)
  train_step_outputs_np = {k: v.numpy() for k, v in train_step_outputs.items()}
  logging.info(
      'Training metrics for step %d: %s', current_step, train_step_outputs_np)
  if 'eval' in mode:
    eval_lib.run_eval_epoch(
        train_steps,
        test_fn,
        test_dataset,
        test_summary_writer,
        val_fn,
        val_dataset,
        val_summary_writer,
        ood_fn,
        ood_dataset,
        ood_summary_writer,
        hparams=hparams)
  # Save checkpoint at the end of training.
  if checkpoint_manager:
    checkpoint_manager.save(checkpoint_number=train_steps)

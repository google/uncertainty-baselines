# coding=utf-8
# Copyright 2020 The Uncertainty Baselines Authors.
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
"""Basic eval functions for Uncertainty Baselines."""

import os.path
import time

from typing import Callable, Dict, Iterator, List, Optional, Tuple
from absl import logging
import tensorflow.compat.v2 as tf
import uncertainty_baselines as ub


_EVAL_SLEEP_SECS = 5

_TensorDict = Dict[str, tf.Tensor]
EvalStepFn = Callable[[Iterator[_TensorDict]], _TensorDict]


def eval_step_fn(
    model: tf.keras.Model,
    strategy: tf.distribute.Strategy,
    metrics: Dict[str, tf.keras.metrics.Metric],
    iterations_per_loop: int) -> EvalStepFn:
  """Generator for a function to run iters_per_loop validation/test steps."""

  @tf.function
  def eval_step(train_iterator: Iterator[_TensorDict]) -> _TensorDict:
    def step(per_replica_inputs: _TensorDict):
      """The function defining a single validation/test step."""
      features = per_replica_inputs['features']
      labels = per_replica_inputs['labels']
      logits = model(features, training=False)
      loss = tf.reduce_mean(
          tf.keras.losses.sparse_categorical_crossentropy(
              y_true=labels, y_pred=logits, from_logits=True))
      # TODO(znado): refactor this to use uncertainty metrics.
      # Later when metric.result() is called, it will return the computed
      # result, averaged across replicas.
      accuracy = ub.utils.compute_accuracy(labels=labels, logits=logits)
      metrics['accuracy'].update_state(accuracy)
      metrics['loss'].update_state(loss)

    for metric in metrics.values():
      metric.reset_states()
    for _ in tf.range(iterations_per_loop):  # Note the use of tf.range.
      ub.utils.call_step_fn(strategy, step, next(train_iterator))
    return {name: value.result() for name, value in metrics.items()}

  return eval_step


def run_eval_epoch(
    val_fn: EvalStepFn,
    val_dataset: tf.data.Dataset,
    val_summary_writer: tf.summary.SummaryWriter,
    test_fn: EvalStepFn,
    test_dataset: tf.data.Dataset,
    test_summary_writer: tf.summary.SummaryWriter,
    current_step: int):
  """Run one evaluation epoch on the test and optionally validation splits."""
  if val_dataset:
    val_iterator = iter(val_dataset)
    val_outputs = val_fn(val_iterator)
    with val_summary_writer.as_default():
      for name, metric in val_outputs.items():
        tf.summary.scalar(name, metric, step=current_step)
    val_outputs_np = {k: v.numpy() for k, v in val_outputs.items()}
    logging.info(
        'Validation metrics for step %d: %s', current_step, val_outputs_np)
  test_iterator = iter(test_dataset)
  test_outputs = test_fn(test_iterator)
  with test_summary_writer.as_default():
    for name, metric in test_outputs.items():
      tf.summary.scalar(name, metric, step=current_step)


_EvalSetupResult = Tuple[
    Optional[EvalStepFn],
    Optional[tf.data.Dataset],
    Optional[tf.summary.SummaryWriter],
    EvalStepFn,
    tf.data.Dataset,
    tf.summary.SummaryWriter]


def setup_eval(
    dataset_builder: ub.datasets.BaseDataset,
    strategy,
    trial_dir: str,
    model: tf.keras.Model,
    metric_names: List[str]) -> _EvalSetupResult:
  """Setup the test and optionally validation loggers, step fns and datasets."""
  test_dataset = ub.utils.build_dataset(dataset_builder, strategy, 'test')
  test_summary_writer = tf.summary.create_file_writer(
      os.path.join(trial_dir, 'summaries/test'))
  num_test_steps = (
      dataset_builder.info['num_test_examples'] //
      dataset_builder.eval_batch_size)
  test_metrics = {
      name: tf.keras.metrics.Mean(name, dtype=tf.float32)
      for name in metric_names
  }
  test_fn = eval_step_fn(
      model,
      strategy,
      test_metrics,
      iterations_per_loop=num_test_steps)

  # Have to have separate val_fn and test_fn because otherwise tf.function
  # retraces the function each time, which is very slow, because we are passing
  # in a Python dict of metrics and int for iterations_per_loop.
  val_fn = None
  val_dataset = None
  val_summary_writer = None
  if dataset_builder.info['num_validation_examples'] > 0:
    num_val_steps = (
        dataset_builder.info['num_validation_examples'] //
        dataset_builder.eval_batch_size)
    val_dataset = ub.utils.build_dataset(
        dataset_builder, strategy, 'validation')
    val_summary_writer = tf.summary.create_file_writer(
        os.path.join(trial_dir, 'summaries/val'))
    if num_val_steps == num_test_steps:
      val_fn = test_fn
    else:
      # The metrics are reset at the start of each call to {val,test}_fn, so
      # reusing them is safe.
      val_fn = eval_step_fn(
          model,
          strategy,
          test_metrics,
          iterations_per_loop=num_val_steps)
  return (
      val_fn, val_dataset, val_summary_writer, test_fn, test_dataset,
      test_summary_writer)


def run_eval_loop(
    dataset_builder: ub.datasets.BaseDataset,
    model: tf.keras.Model,
    trial_dir: str,
    train_steps: int,
    strategy: tf.distribute.Strategy,
    metric_names: List[str],
    checkpoint_step: int = -1):
  """Evaluate the model on the validation and test splits and record metrics."""
  (val_fn,
   val_dataset,
   val_summary_writer,
   test_fn,
   test_dataset,
   test_summary_writer) = setup_eval(
       dataset_builder, strategy, trial_dir, model, metric_names)

  checkpoint = tf.train.Checkpoint(model=model)
  last_eval_step = -1
  # Note that this will only grab the latest checkpoint, so if multiple
  # checkpoints are saved while this is sleeping, it will skip the ones in
  # between.
  while True:
    # Check for a new checkpoint, and if there is not one, sleep for several
    # seconds.
    if checkpoint_step >= 0:
      checkpoint_path = os.path.join(
          trial_dir, 'model.ckpt-{}'.format(checkpoint_step))
    else:
      checkpoint_path = tf.train.latest_checkpoint(trial_dir)
    if not checkpoint_path:
      last_checkpoint_step = last_eval_step
    else:
      last_checkpoint_step = int(checkpoint_path.split('-')[-1])
    if last_checkpoint_step == last_eval_step:
      logging.info(
          'No new checkpoints since step %d (latest path is %s). Sleeping '
          'for %d seconds...',
          last_eval_step,
          checkpoint_path,
          _EVAL_SLEEP_SECS)
      time.sleep(_EVAL_SLEEP_SECS)
      continue

    # Restore from the latest checkpoint and evalutate on the validation and
    # test splits.
    last_eval_step = last_checkpoint_step
    logging.info('Restoring model from checkpoint %s.', checkpoint_path)
    checkpoint.restore(checkpoint_path)
    run_eval_epoch(
        val_fn,
        val_dataset,
        val_summary_writer,
        test_fn,
        test_dataset,
        test_summary_writer,
        last_eval_step)
    if last_eval_step >= train_steps:
      break

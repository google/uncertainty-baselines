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
"""Collection of shared utility functions."""

from typing import Any, Callable, Dict, Optional
from absl import logging
import numpy as np

import tensorflow.compat.v2 as tf
from uncertainty_baselines.datasets import base


def assert_weights_loaded(model: tf.keras.Model, before_restore_vars):
  """Assert that all variables changed after restoring."""
  after_restore_vars = model.trainable_weights
  for wi, (before_var, after_var) in enumerate(
      zip(before_restore_vars, after_restore_vars)):
    weight_i = model.trainable_weights[wi]
    logging.warning(
        'Weight %s norm before restore %f and after %f.',
        weight_i.name,
        np.linalg.norm(before_var),
        np.linalg.norm(after_var))
    if ('kernel/linear_bias' not in weight_i.name and
        np.array_equal(before_var, after_var)):
      error_message = (
          'Weight #{} ({}) was not restored properly. Norm at '
          'initialization: {}, norm after restoration: {}').format(
              wi,
              weight_i,
              np.linalg.norm(before_var),
              np.linalg.norm(after_var))
      raise ValueError(error_message)


def build_dataset(
    dataset_builder: base.BaseDataset,
    strategy,
    split: str,
    as_tuple: bool = False) -> tf.data.Dataset:
  """Build a tf.data.Dataset iterator using a DistributionStrategy."""
  dataset = dataset_builder.build(split, as_tuple=as_tuple)
  # TOOD(znado): look into using experimental_distribute_datasets_from_function
  # and a wrapped version of dataset_builder.build (see
  # https://www.tensorflow.org/api_docs/python/tf/distribute/Strategy#experimental_distribute_datasets_from_function).
  # Can experimental_distribute_dataset properly shard TFDS datasets?
  if strategy.num_replicas_in_sync > 1:
    dataset = strategy.experimental_distribute_dataset(dataset)
  return dataset


_TensorDict = Dict[str, tf.Tensor]
_StepFn = Callable[[_TensorDict], Optional[_TensorDict]]


def call_step_fn(
    strategy,
    step_fn: _StepFn,
    global_inputs: Any) -> Dict[str, float]:
  """Call the step_fn on the iterator output using DistributionStrategy."""
  step_outputs = strategy.run(step_fn, args=(global_inputs,))
  if step_outputs is None:
    return step_outputs
  if strategy.num_replicas_in_sync > 1:
    step_outputs = {
        name: strategy.reduce(tf.distribute.ReduceOp.SUM, (output,), axis=0)
        for name, output in step_outputs.items()
    }
    step_outputs = {
        k: v / strategy.num_replicas_in_sync
        for k, v in step_outputs.items()
    }
  return step_outputs


# TODO(znado): remove when uncertainty metrics has this implemented.
def compute_accuracy(labels: tf.Tensor, logits: tf.Tensor) -> tf.Tensor:
  """Computes classification accuracy given logits and dense labels.

  Args:
    labels: Integer Tensor of dense labels, shape [batch_size].
    logits: Tensor of shape [batch_size, num_classes].
  Returns:
    A scalar for the classification accuracy.
  """
  correct_prediction = tf.equal(
      tf.argmax(logits, 1, output_type=tf.int32), labels)
  return tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

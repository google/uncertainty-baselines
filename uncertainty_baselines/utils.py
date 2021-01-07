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
"""Collection of shared utility functions."""

from typing import Any, Callable, Dict, Optional

from absl import logging
import numpy as np
import tensorflow.compat.v2 as tf


def apply_label_smoothing(one_hot_targets, label_smoothing):
  """Apply label smoothing to the one-hot targets.

  Applies label smoothing such that the on-values are transformed from 1.0 to
  `1.0 - label_smoothing + label_smoothing / num_classes`, and the off-values
  are transformed from 0.0 to `label_smoothing / num_classes`. This weighted
  mixture of `one_hot_targets` with the uniform distribution is the same as is
  done in [1] and the original label smoothing paper [2].

  #### References
  [1]: Müller, Rafael, Simon Kornblith, and Geoffrey E. Hinton.
  "When does label smoothing help?." Advances in Neural Information Processing
  Systems. 2019.
  https://arxiv.org/abs/1906.02629
  [2]:  Christian Szegedy, Vincent Vanhoucke, Sergey Ioffe, Jon Shlens, and
  Zbigniew Wojna. "Rethinking the inception architecture for computer vision."
  In Proceedings of the IEEE conference on computer vision and pattern
  recognition, pages 2818–2826, 2016.
  https://arxiv.org/abs/1512.00567

  Args:
    one_hot_targets: one-hot targets for an example, a [batch, ..., num_classes]
      float array.
    label_smoothing: a scalarin [0, 1] used to smooth the labels.

  Returns:
    A float array of the same shape as `one_hot_targets` with smoothed label
    values.
  """
  on_value = 1.0 - label_smoothing
  num_classes = one_hot_targets.shape[-1]
  off_value = label_smoothing / num_classes
  one_hot_targets = one_hot_targets * on_value + off_value
  return one_hot_targets


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


_TensorDict = Dict[str, tf.Tensor]
_StepFn = Callable[[_TensorDict], Optional[_TensorDict]]


def call_step_fn(strategy,
                 step_fn: _StepFn,
                 global_inputs: Any,
                 concatenate_outputs: bool = False) -> Dict[str, float]:
  """Call the step_fn on the iterator output using DistributionStrategy."""

  step_outputs = strategy.run(step_fn, args=(global_inputs,))
  if step_outputs is None:
    return step_outputs
  if strategy.num_replicas_in_sync > 1:
    if concatenate_outputs:
      step_outputs = {
          name: tf.concat(output.values, axis=0)
          for name, output in step_outputs.items()
      }
    else:
      step_outputs = {
          name: strategy.reduce(tf.distribute.ReduceOp.SUM, (output,), axis=0)
          for name, output in step_outputs.items()
      }
      step_outputs = {
          k: v / strategy.num_replicas_in_sync for k, v in step_outputs.items()
      }
  return step_outputs

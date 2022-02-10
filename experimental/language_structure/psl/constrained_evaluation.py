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

# Lint as: python3
"""Constrained gradient updates during inference.

File consists of:
- Gradient updates during evaluation
"""

from typing import List

import tensorflow as tf
import psl_model  # local file import from experimental.language_structure.psl


def satisfy_weights(model, data: tf.Tensor, labels: tf.Tensor,
                    weights: List[tf.Tensor], constraints: psl_model.PSLModel,
                    grad_steps: int, alpha: float):
  """Update weights by satisfing test constraints."""
  for _ in range(grad_steps):
    with tf.GradientTape() as tape:
      logits = model(data, training=False)
      constraint_loss = constraints.compute_loss(data, logits)
      weight_loss = tf.reduce_sum([
          tf.reduce_mean(tf.math.squared_difference(w, w_h))
          for w, w_h in zip(weights, model.weights)
      ])
      loss = constraint_loss + alpha * weight_loss

    trainable_vars = model.trainable_variables
    gradients = tape.gradient(loss, trainable_vars)

    model.optimizer.apply_gradients(zip(gradients, trainable_vars))
    model.compiled_metrics.update_state(labels, logits)

  return model


def copy_model_weights(weights: List[tf.Tensor]) -> List[tf.Tensor]:
  """Copies a list of model weights."""
  weights_copy = []
  for layer in weights:
    weights_copy.append(tf.identity(layer))

  return weights_copy


def test_step(model, data: tf.Tensor, labels: tf.Tensor,
              constraints: psl_model.PSLModel, grad_steps: int,
              alpha: float) -> tf.Tensor:
  """Test step for gradient based weight updates.

  Args:
    model: tensorflow model being run
    data: input features
    labels: ground truth labels
    constraints: differentable psl constraints
    grad_steps: number of gradient steps taken to try and satisfy the
      constraints
    alpha: parameter to determine how important it is to keep the constrained
      weights close to the trained unconstrained weights

  Returns:
    Logits after satisfiying constraints.
  """
  weights_copy = copy_model_weights(model.get_weights())
  model = satisfy_weights(
      model,
      data,
      labels,
      weights=weights_copy,
      constraints=constraints,
      grad_steps=grad_steps,
      alpha=alpha)

  logits = model(data, training=False)
  model.compiled_loss(labels, logits)
  model.compiled_metrics.update_state(labels, logits)

  model.set_weights(weights_copy)

  return logits


def evaluate_constrained_model(model,
                               dataset: tf.Tensor,
                               constraints: psl_model.PSLModel,
                               grad_steps: int = 10,
                               alpha: float = 0.1) -> List[tf.Tensor]:
  """Custom evaluation step."""
  logits = []
  for x_batch, y_batch in dataset:
    batch_logits = test_step(
        model,
        x_batch,
        y_batch,
        constraints=constraints,
        grad_steps=grad_steps,
        alpha=alpha)
    logits.append(batch_logits)

  for metric in model.metrics:
    tf.print(metric, metric.result())

  return logits

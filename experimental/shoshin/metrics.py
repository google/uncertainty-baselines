# coding=utf-8
# Copyright 2024 The Uncertainty Baselines Authors.
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

"""Custom evaluation metrics."""

import tensorflow as tf


@tf.keras.saving.register_keras_serializable('one_vs_rest')
class OneVsRest(tf.keras.metrics.Metric):
  """A wrapper that extends metrics from binary to multi-class setup.

  This extension is done using one-vs-rest strategy. In which one class
  will be treated as a positive and the rest classes will be negative,
  hence reducing the calculation to a binary setup.
  """

  def __init__(
      self, metric: tf.keras.metrics.Metric, positive_class_index: int, **kwargs
  ):
    """Constructor.

    Args:
      metric: A metric object that is need to be extended for multi-class setup.
      positive_class_index: The index of the positive class.
      **kwargs: Keyword arguments expected by the core metric.
    """
    super(OneVsRest, self).__init__(name=metric.name, **kwargs)
    self.metric = metric
    self.positive_class_index = positive_class_index

  def update_state(self, y_true, y_pred, **kwargs):
    """Accumulate metrics statistics.

    Args:
      y_true: The ground truth labels. An integer tensor of shape (num_examples,
        num_classes).
      y_pred: The predicted values. A float tensor of shape (num_examples,
        num_classes).
      **kwargs: Keyword arguments expected by the core metric.
    """
    y_pred = y_pred[..., self.positive_class_index]
    y_true = y_true[..., self.positive_class_index]
    self.metric.update_state(y_true, y_pred, **kwargs)

  def result(self):
    return self.metric.result()

  def reset_state(self):
    self.metric.reset_state()

  def get_config(self):
    return {
        'metric': self.metric,
        'positive_class_index': self.positive_class_index
    }

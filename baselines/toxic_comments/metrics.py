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

"""Measures for models interacting with an oracle.

Oracle Collaborative AUC measures the usefulness of model uncertainty scores in
facilitating human-computer collaboration (e.g., between a neural model and an
"oracle", e.g. a human moderator in moderating online toxic comments).

The idea is that given a large amount of testing examples, the model will first
generate predictions for all examples, and then send a certain percentage of
examples that it is not confident about to the oracle, which returns perfect
predictions for those examples.

The goal of this metric is to understand, under capacity constraints (e.g. if
the model is only allowed to send 0.1% of total examples to the oracle), how
well the model can collaborate with it to achieve the best overall performance.
In this way, these metrics attempt to quantify the behavior of the full
model-oracle system rather than of the model alone.

A model that collaborates with an oracle well should not be accurate, but also
capable of quantifying its uncertainty well (i.e., its uncertainty should be
calibrated such that uncertainty ≅ model accuracy).
"""
from typing import Optional, Sequence

from robustness_metrics.metrics import uncertainty
import tensorflow as tf


def _replace_first_and_last_elements(original_tensor: Sequence[float],
                                     new_first_elem: float,
                                     new_last_elem: float):
  """Return a copy of original_tensor replacing its first and last elements."""
  return tf.concat([[new_first_elem], original_tensor[1:-1], [new_last_elem]],
                   axis=0)


def _compute_correct_predictions(y_true: Sequence[float],
                                 y_pred: Sequence[float],
                                 dtype: tf.DType = tf.float32) -> tf.Tensor:
  """Computes binary 'labels' of prediction correctness.

  Args:
    y_true: The ground truth labels. Shape (batch_size, ).
    y_pred: The predicted labels. Must be integer valued predictions for label
      index rather than the predictive probability. For multi-label
      classification problems, y_pred is typically obtained as
      `tf.math.reduce_max(logits)`. Shape (batch_size, ).
    dtype: (Optional) data type of the metric result.

  Returns:
    A Tensor of dtype and shape (batch_size, ).
  """
  y_true = tf.cast(tf.convert_to_tensor(y_true), dtype=dtype)
  y_pred = tf.cast(tf.convert_to_tensor(y_pred), dtype=dtype)

  # Ranks of both y_pred and y_true should be 1.
  if len(y_true.shape) != 1 or len(y_pred.shape) != 1:
    raise ValueError("Ranks of y_true and y_pred must both be 1. "
                     f"Got {len(y_true.shape)} and {len(y_pred.shape)}")

  # Creates binary 'label' of correct prediction, shape (batch_size, ).
  correct_preds = tf.math.equal(y_true, y_pred)
  return tf.cast(correct_preds, dtype=dtype)


# pylint: disable=protected-access
OracleCollaborativeAUC = uncertainty._KerasOracleCollaborativeAUCMetric
CalibrationAUC = uncertainty._KerasCalibrationAUCMetric
# pylint: enable=protected-access


class AbstainPrecision(tf.keras.metrics.Metric):
  """Implements the abstention precision metric.

  `AbstainPrecision` measures a model's uncertainty quantification ability
  by assuming the model has the ability to abstain (i.e., refuse to predict

  for an example due to low confidence). The abstention process can be done
  either per example, or globally over the dataset. In the latter case, the
  rejection decision is made by rejecting a pre-specified percentage of examples
  according to prediction confidence. This metric computes the percentage of
  correctly rejected examples, which is the percentage of incorrect predictions
  among all the abstained examples.

  The abstention decision is made under a budget, i.e., the model is only
  allowed to abstain a small amount of examples. For `AbstainPrecision`, this
  budget can be specified to be either a fixed number (`max_abstain_count`), or
  the fraction of the total dataset (`abstain_fraction`).

  It can be understood as the uncertainty analogy of 'Precision@TopK', where
  the ranking signal is the uncertainty score and the "label" is the prediction
  correctness. "TopK" is specified as the fraction of the total examples.

  For a AUC-style metric of the abstention policy, see `CalibrationAUC`.

  Attributes:
    abstain_fraction: The fraction of total examples to abstain.
    num_approx_bins: Number of histogram bins to use to approximate the
      distribution of the uncertainty score.
    max_abstain_count: The maximum number of total examples to abstain. If set,
      then the number of example to abstain is limited to be not larger than
      this value.
    binned_total_counts: The number of total examples in each bins of the
      uncertainty score historgram, shape (num_approx_bins, ).
    binned_correct_counts: The number of correct predictions in each bins of the
      uncertainty score historgram, shape (num_approx_bins, ).
    return_abstain_count: Whether to return the number of abstained examples
      rather than the precision.
  """

  # TODO(jereliu): Implement threshold-based abstention policy.

  def __init__(self,
               abstain_fraction: float = 0.01,
               num_approx_bins: int = 1000,
               max_abstain_count: Optional[int] = None,
               name: Optional[str] = None,
               dtype: Optional[tf.DType] = None,
               return_abstain_count: bool = False):
    """Constructs the abstention precision metric.

    Notice that `abstain_fraction` and `max_abstain_count` interact
    (i.e. the number abstained is the minimum of the two numbers defined
    by `abstain_fraction` and `max_abstain_count`).

    Args:
      abstain_fraction: The fraction of total examples to abstain. A float value
        between [0, 1].
      num_approx_bins: (Optional) Number of histogram bins to use to approximate
        the distribution of the uncertainty score.
      max_abstain_count: The maximum number of total examples to abstain. If
        set, then the number of example to abstain is limited to be not larger
        than this value.
      name: (Optional) Name of this metric.
      dtype: (Optional) Data type. Must be floating-point.
      return_abstain_count: (Optional) Whether to return the number of abstained
        examples rather than the precision. For debugging purpose only, default
        to False.
    """
    super().__init__(name=name, dtype=dtype)

    if max_abstain_count is not None:
      max_abstain_count = tf.cast(max_abstain_count, dtype=self.dtype)

    self.abstain_fraction = tf.cast(abstain_fraction, dtype=self.dtype)
    self.num_approx_bins = num_approx_bins
    self.max_abstain_count = max_abstain_count
    self.return_abstain_count = return_abstain_count

    # Initializes histogram for confidence score distributions.
    self.binned_total_counts = self.add_weight(
        "binned_total_counts",
        shape=(num_approx_bins,),
        initializer=tf.zeros_initializer,
        dtype=self.dtype)
    self.binned_correct_counts = self.add_weight(
        "binned_correct_counts",
        shape=(num_approx_bins,),
        initializer=tf.zeros_initializer,
        dtype=self.dtype)

  def update_state(self,
                   y_true: Sequence[float],
                   y_pred: Sequence[float],
                   confidence: Sequence[float],
                   sample_weight: Optional[Sequence[float]] = None) -> None:
    """Updates confidence and accuracy statistics.

    Args:
      y_true: The ground truth labels. Shape (batch_size, ).
      y_pred: The predicted labels. Must be integer valued predictions for label
        index rather than the predictive probability. For multi-label
        classification problems, `y_pred` is typically obtained as
        `tf.math.reduce_max(logits)`. Shape (batch_size, ).
      confidence: The confidence score where higher value indicates lower
        uncertainty. Values should be within [0, 1].
      sample_weight: Optional weighting of each example. Defaults to 1. Can be a
        Tensor whose rank is either 0, or the same rank as `y_true`, and must be
        broadcastable to `y_true`.
    """
    batch_size = tf.shape(y_true)[0]

    # Preprocess `confidence` and `sample_weight` tensors.
    confidence = tf.cast(tf.convert_to_tensor(confidence), dtype=self.dtype)
    confidence = tf.reshape(confidence, shape=(batch_size,))

    if sample_weight is not None:
      sample_weight = tf.convert_to_tensor(sample_weight)
      sample_weight = tf.reshape(sample_weight, shape=(batch_size,))
      sample_weight = tf.cast(sample_weight, dtype=self.dtype)
    else:
      sample_weight = tf.ones((batch_size,), dtype=self.dtype)

    # Computes correct predictions.
    correct_preds = _compute_correct_predictions(
        y_true, y_pred, dtype=self.dtype)
    correct_preds_weighted = correct_preds * sample_weight

    # Computes batch-specific histogram statistics for confidence score.
    batch_bin_indices = tf.histogram_fixed_width_bins(
        confidence,
        tf.constant([0., 1.], self.dtype),
        nbins=self.num_approx_bins)
    batch_total_counts = tf.math.unsorted_segment_sum(
        data=sample_weight,
        segment_ids=batch_bin_indices,
        num_segments=self.num_approx_bins)
    batch_correct_counts = tf.math.unsorted_segment_sum(
        data=correct_preds_weighted,
        segment_ids=batch_bin_indices,
        num_segments=self.num_approx_bins)

    self.binned_total_counts.assign_add(batch_total_counts)
    self.binned_correct_counts.assign_add(batch_correct_counts)

  def result(self):
    """Computes the abstention precision."""
    # TODO(jereliu): Incorporate uncertainty threshold into the computation of
    # `total_count_abstained`.

    # Computes the number of examples to abstain.
    total_counts = tf.reduce_sum(self.binned_total_counts)
    total_count_abstained = tf.floor(total_counts * self.abstain_fraction)

    if self.max_abstain_count is not None:
      total_count_abstained = tf.reduce_min(
          [total_count_abstained, self.max_abstain_count])

    if self.return_abstain_count:
      return total_count_abstained

    # Computes the correct predictions among the examples to be abstained.
    correct_predictions_abstained = self._compute_correct_predictions_abstained(
        total_count_abstained)

    return tf.math.divide_no_nan(
        total_count_abstained - correct_predictions_abstained,
        total_count_abstained)

  def _compute_correct_predictions_abstained(
      self, total_count_abstained: int) -> tf.Tensor:
    """Approximates the number of correct predictions in abstained examples.

    Args:
      total_count_abstained: Maximum number of examples to abstain.

    Returns:
      A scalar Tensor of self.dtype.
    """
    # Computes unique cumulative counts for non-empty bins.
    non_empty_bin_mask = self.binned_total_counts > 0.
    binned_total_counts_masked = tf.boolean_mask(self.binned_total_counts,
                                                 non_empty_bin_mask)
    binned_correct_counts_masked = tf.boolean_mask(self.binned_correct_counts,
                                                   non_empty_bin_mask)
    cumulative_total_counts = tf.cumsum(binned_total_counts_masked)

    # Finds the index of the bin whose cumulative count first exceeds the
    # `total_count_abstained`.
    final_bin_index = tf.argmax(
        cumulative_total_counts >= total_count_abstained, output_type=tf.int32)

    # Computes the example counts before the final bin.
    total_count_before_final_bin = tf.cond(
        final_bin_index > 0,
        lambda: cumulative_total_counts[final_bin_index - 1], lambda: 0.)
    correct_count_before_final_bin = tf.cond(
        final_bin_index > 0,
        lambda: tf.reduce_sum(binned_correct_counts_masked[:final_bin_index]),
        lambda: 0.)

    # Approximates the correct count for the final bin.
    total_count_abstained_final_bin = (
        total_count_abstained - total_count_before_final_bin)
    accuracy_final_bin = (
        binned_correct_counts_masked[final_bin_index] /
        binned_total_counts_masked[final_bin_index])
    correct_count_final_bin = (
        accuracy_final_bin * total_count_abstained_final_bin)

    return correct_count_before_final_bin + correct_count_final_bin

  def reset_states(self):
    """Resets all of the metric state variables.

    This function is called between epochs/steps,
    when a metric is evaluated during training.
    """
    vars_to_reset = (self.binned_total_counts, self.binned_correct_counts)
    tf.keras.backend.batch_set_value([(v, [
        0.,
    ] * self.num_approx_bins) for v in vars_to_reset])


class AbstainRecall(AbstainPrecision):
  """Implements the abstention recall metric.

  Different from `AbstainPrecision`, `AbstainRecall` computes the percentage of
  correctly abstained examples among all the incorrect predictions that **could
  have been abstained**.

  As a result, assume the model abstains according to confidence and under the
  budget. The numerator is the total number of incorrect predictions among the
  abstained examples, and the denominator is the total incorrect predictions
  made by the model.
  """

  def result(self):
    """Computes the abstention recall."""
    # TODO(jereliu): Incorporate uncertainty threshold into the computation of
    # `total_count_abstained`.

    # Computes numerator: the number of successfully abstained examples.
    total_counts = tf.reduce_sum(self.binned_total_counts)
    total_count_abstained = tf.floor(total_counts * self.abstain_fraction)
    correct_predictions_abstained = self._compute_correct_predictions_abstained(
        total_count_abstained)
    incorrect_predictions_abstained = (
        total_count_abstained - correct_predictions_abstained)

    # Computes denominator: the total number of incorrect predictions.
    correct_counts = tf.reduce_sum(self.binned_correct_counts)
    incorrect_counts = total_counts - correct_counts

    if self.return_abstain_count:
      return incorrect_counts

    return incorrect_predictions_abstained / incorrect_counts

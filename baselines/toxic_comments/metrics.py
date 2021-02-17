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
from typing import Any, Dict, Mapping, Optional, Sequence
import tensorflow as tf


def _replace_first_and_last_elements(original_tensor: Sequence[float],
                                     new_first_elem: float,
                                     new_last_elem: float):
  """Return a copy of original_tensor replacing its first and last elements."""
  return tf.concat([[new_first_elem], original_tensor[1:-1], [new_last_elem]],
                   axis=0)


class OracleCollaborativeAUC(tf.keras.metrics.AUC):
  """Computes the approximate oracle-collaborative equivalent of the AUC.

  This metric computes four local variables: binned_true_positives,
  binned_true_negatives, binned_false_positives, and binned_false_negatives, as
  a function of a linearly spaced set of thresholds and score bins. These are
  then sent to the oracle in increasing bin order, and used to compute the
  Oracle-Collaborative ROC-AUC or Oracle-Collaborative PR-AUC.

  Note because the AUC must be computed online that the results are not exact,
  but rather are expected values, similar to the regular AUC computation.
  """

  def __init__(self,
               oracle_fraction: float = 0.01,
               num_bins: int = 15,
               num_thresholds: int = 200,
               curve: str = "ROC",
               summation_method: str = "interpolation",
               name: Optional[str] = None,
               dtype: Optional[tf.DType] = None):
    """Constructs an expected oracle-collaborative AUC metric.

    Args:
      oracle_fraction: the fraction of total examples to send to the oracle.
      num_bins: Number of bins for the uncertainty score to maintain over the
        interval [0, 1].
      num_thresholds: Number of thresholds to use in linearly interpolating the
        ROC curve.
      curve: Name of the curve to be computed, either ROC (default) or PR
        (Precision-Recall).
      summation_method: Specifies the Riemann summation method. 'interpolation'
        applies the mid-point summation scheme for ROC. For PR-AUC, interpolates
        (true/false) positives but not the ratio that is precision (see Davis &
        Goadrich 2006 for details); 'minoring' applies left summation for
        increasing intervals and right summation for decreasing intervals;
        'majoring' does the opposite.
      name: (Optional) Name of this metric.
      dtype: (Optional) Data type. Must be floating-point.  Currently only
        binary data is supported.
    """
    # Check inputs.
    if not 0 <= oracle_fraction <= 1:
      raise ValueError("oracle_fraction must be between 0 and 1.")
    if num_bins <= 1:
      raise ValueError("num_bins must be > 1.")
    if dtype and not dtype.is_floating:
      raise ValueError("dtype must be a float type.")

    self.oracle_fraction = oracle_fraction
    self.num_bins = num_bins

    super().__init__(
        num_thresholds=num_thresholds,
        curve=curve,
        summation_method=summation_method,
        name=name,
        dtype=dtype)

    self.binned_true_positives = self.add_weight(
        "binned_true_positives",
        shape=(num_thresholds, num_bins),
        initializer=tf.zeros_initializer)

    self.binned_true_negatives = self.add_weight(
        "binned_true_negatives",
        shape=(num_thresholds, num_bins),
        initializer=tf.zeros_initializer)

    self.binned_false_positives = self.add_weight(
        "binned_false_positives",
        shape=(num_thresholds, num_bins),
        initializer=tf.zeros_initializer)

    self.binned_false_negatives = self.add_weight(
        "binned_false_negatives",
        shape=(num_thresholds, num_bins),
        initializer=tf.zeros_initializer)

  def update_state(self,
                   labels: Sequence[float],
                   probabilities: Sequence[float],
                   custom_binning_score: Optional[Sequence[float]] = None,
                   **kwargs: Mapping[str, Any]) -> None:
    """Updates the confusion matrix for OracleCollaborativeAUC.

    This will flatten the labels, probabilities, and custom binning score, and
    then compute the confusion matrix over all predictions.

    Args:
      labels: Tensor of shape [N,] of class labels in [0, k-1], where N is the
        number of examples. Currently only binary labels (0 or 1) are supported.
      probabilities: Tensor of shape [N,] of normalized probabilities associated
        with the positive class.
      custom_binning_score: (Optional) Tensor of shape [N,] used for assigning
        predictions to uncertainty bins. If not set, the default is to bin by
        predicted probability. All elements of custom_binning_score must be in
        [0, 1].
      **kwargs: Other potential keywords, which will be ignored by this method.
    """
    del kwargs  # Unused
    labels = tf.convert_to_tensor(labels)
    probabilities = tf.cast(probabilities, self.dtype)

    # Reshape labels, probabilities, custom_binning_score to [1, num_examples].
    labels = tf.reshape(labels, [1, -1])
    probabilities = tf.reshape(probabilities, [1, -1])
    if custom_binning_score is not None:
      custom_binning_score = tf.cast(
          tf.reshape(custom_binning_score, [1, -1]), self.dtype)
    # Reshape thresholds to [num_thresholds, 1] for easy tiling.
    thresholds = tf.cast(tf.reshape(self._thresholds, [-1, 1]), self.dtype)

    # pred_labels and true_labels both have shape [num_examples, num_thresholds]
    pred_labels = probabilities > thresholds
    true_labels = tf.tile(tf.cast(labels, tf.bool), [self.num_thresholds, 1])

    # Bin by distance from threshold if a custom_binning_score was not set.
    if custom_binning_score is None:
      custom_binning_score = tf.abs(probabilities - thresholds)
    else:
      # Tile the provided custom_binning_score for each threshold.
      custom_binning_score = tf.tile(custom_binning_score,
                                     [self.num_thresholds, 1])

    # Bin thresholded predictions using custom_binning_score.
    batch_binned_confusion_matrix = self._bin_confusion_matrix_by_score(
        pred_labels, true_labels, custom_binning_score)

    self.binned_true_positives.assign_add(
        batch_binned_confusion_matrix["true_positives"])
    self.binned_true_negatives.assign_add(
        batch_binned_confusion_matrix["true_negatives"])
    self.binned_false_positives.assign_add(
        batch_binned_confusion_matrix["false_positives"])
    self.binned_false_negatives.assign_add(
        batch_binned_confusion_matrix["false_negatives"])

  def _bin_confusion_matrix_by_score(
      self, pred_labels: Sequence[Sequence[bool]],
      true_labels: Sequence[Sequence[bool]],
      binning_score: Sequence[Sequence[float]]) -> Dict[str, tf.Tensor]:
    """Compute the confusion matrix, binning predictions by a specified score.

    Computes the confusion matrix over matrices of predicted and true labels.
    Each element of the resultant confusion matrix is itself a matrix of the
    same shape as the original input labels.

    In the typical use of this function in OracleCollaborativeAUC, the variables
    T and N (in the args and returns sections below) are the number of
    thresholds and the number of examples, respectively.

    Args:
      pred_labels: Boolean tensor of shape [T, N] of predicted labels.
      true_labels: Boolean tensor of shape [T, N] of true labels.
      binning_score: Boolean tensor of shape [T, N] of scores to use in
        assigning labels to bins.

    Returns:
      Dictionary of strings to entries of the confusion matrix
      ('true_positives', 'true_negatives', 'false_positives',
      'false_negatives'). Each entry is a tensor of shape [T, nbins].
    """
    correct_preds = tf.math.equal(pred_labels, true_labels)

    # Elements of the confusion matrix have shape [M, N]
    pred_true_positives = tf.math.logical_and(correct_preds, pred_labels)
    pred_true_negatives = tf.math.logical_and(correct_preds,
                                              tf.math.logical_not(pred_labels))
    pred_false_positives = tf.math.logical_and(
        tf.math.logical_not(correct_preds), pred_labels)
    pred_false_negatives = tf.math.logical_and(
        tf.math.logical_not(correct_preds), tf.math.logical_not(pred_labels))

    # Cast confusion matrix elements from bool to self.dtype.
    pred_true_positives = tf.cast(pred_true_positives, self.dtype)
    pred_true_negatives = tf.cast(pred_true_negatives, self.dtype)
    pred_false_positives = tf.cast(pred_false_positives, self.dtype)
    pred_false_negatives = tf.cast(pred_false_negatives, self.dtype)

    bin_indices = tf.histogram_fixed_width_bins(
        binning_score, tf.constant([0.0, 1.0], self.dtype), nbins=self.num_bins)

    binned_true_positives_rows = []
    binned_true_negatives_rows = []
    binned_false_positives_rows = []
    binned_false_negatives_rows = []

    # tf.math.unsorted_segment_sum doesn't support different indices in
    # different rows; zip over rows of the confusion matrix and stack instead.
    for tp_row, tn_row, fp_row, fn_row, idx_row in (
        zip(pred_true_positives, pred_true_negatives, pred_false_positives,
            pred_false_negatives, bin_indices)):
      binned_true_positives_rows.append(
          tf.math.unsorted_segment_sum(
              data=tp_row, segment_ids=idx_row, num_segments=self.num_bins))
      binned_true_negatives_rows.append(
          tf.math.unsorted_segment_sum(
              data=tn_row, segment_ids=idx_row, num_segments=self.num_bins))
      binned_false_positives_rows.append(
          tf.math.unsorted_segment_sum(
              data=fp_row, segment_ids=idx_row, num_segments=self.num_bins))
      binned_false_negatives_rows.append(
          tf.math.unsorted_segment_sum(
              data=fn_row, segment_ids=idx_row, num_segments=self.num_bins))

    binned_true_positives = tf.stack(binned_true_positives_rows, axis=0)
    binned_true_negatives = tf.stack(binned_true_negatives_rows, axis=0)
    binned_false_positives = tf.stack(binned_false_positives_rows, axis=0)
    binned_false_negatives = tf.stack(binned_false_negatives_rows, axis=0)

    return {
        "true_positives": binned_true_positives,
        "true_negatives": binned_true_negatives,
        "false_positives": binned_false_positives,
        "false_negatives": binned_false_negatives
    }

  def result(self):
    """Returns the approximate Oracle-Collaborative AUC.

    true_positives, true_negatives, false_positives, and false_negatives contain
    the binned confusion matrix for each threshold. We thus compute the
    confusion matrix (after collaborating with the oracle) as a function of the
    threshold and then integrate over threshold to approximate the final AUC.
    """
    cum_examples = tf.cumsum(
        self.binned_true_positives + self.binned_true_negatives +
        self.binned_false_positives + self.binned_false_negatives,
        axis=1)
    # The number of examples in each row is the same; choose the first.
    num_total_examples = cum_examples[0, -1]
    num_oracle_examples = tf.cast(
        tf.floor(num_total_examples * self.oracle_fraction), self.dtype)

    expected_true_positives = tf.zeros_like(self.true_positives)
    expected_true_negatives = tf.zeros_like(self.true_negatives)
    expected_false_positives = tf.zeros_like(self.false_positives)
    expected_false_negatives = tf.zeros_like(self.false_negatives)

    # Add true positives and true negatives predicted by the oracle. All
    # incorrect predictions are corrected.
    expected_true_positives += tf.reduce_sum(
        tf.where(cum_examples <= num_oracle_examples,
                 self.binned_true_positives + self.binned_false_negatives, 0.0),
        axis=1)
    expected_true_negatives += tf.reduce_sum(
        tf.where(cum_examples <= num_oracle_examples,
                 self.binned_true_negatives + self.binned_false_positives, 0.0),
        axis=1)

    # Identify the final bin the oracle sees examples from, and the remaining
    # number of predictions it can make on that bin.
    last_oracle_bin = tf.argmax(cum_examples > num_oracle_examples, axis=1)
    last_oracle_bin_indices = tf.stack(
        [tf.range(self.num_thresholds, dtype=tf.int64), last_oracle_bin],
        axis=1)
    last_complete_bin = last_oracle_bin - 1
    # The indices for tf.gather_nd must be positive; use this list for selection
    error_guarded_last_complete_bin = tf.abs(last_complete_bin)
    last_complete_bin_indices = (
        tf.stack([
            tf.range(self.num_thresholds, dtype=tf.int64),
            error_guarded_last_complete_bin
        ],
                 axis=1))

    last_complete_bin_cum_examples = tf.gather_nd(cum_examples,
                                                  last_complete_bin_indices)
    last_oracle_bin_cum_examples = tf.gather_nd(cum_examples,
                                                last_oracle_bin_indices)
    oracle_predictions_used = tf.where(last_complete_bin >= 0,
                                       last_complete_bin_cum_examples, 0.0)
    remaining_oracle_predictions = tf.where(
        last_oracle_bin_cum_examples > num_oracle_examples,
        num_oracle_examples - oracle_predictions_used, 0.0)

    # Add the final oracle bin (where the oracle makes some predictions) to the
    # confusion matrix.
    tp_last_oracle_bin = tf.gather_nd(self.binned_true_positives,
                                      last_oracle_bin_indices)
    tn_last_oracle_bin = tf.gather_nd(self.binned_true_negatives,
                                      last_oracle_bin_indices)
    fp_last_oracle_bin = tf.gather_nd(self.binned_false_positives,
                                      last_oracle_bin_indices)
    fn_last_oracle_bin = tf.gather_nd(self.binned_false_negatives,
                                      last_oracle_bin_indices)
    last_bin_count = (
        tp_last_oracle_bin + tn_last_oracle_bin + fp_last_oracle_bin +
        fn_last_oracle_bin)

    corrected_fn_last_bin = tf.math.divide_no_nan(
        fn_last_oracle_bin * remaining_oracle_predictions, last_bin_count)
    corrected_fp_last_bin = tf.math.divide_no_nan(
        fp_last_oracle_bin * remaining_oracle_predictions, last_bin_count)

    expected_true_positives += corrected_fn_last_bin
    expected_true_negatives += corrected_fp_last_bin
    expected_false_positives -= corrected_fp_last_bin
    expected_false_negatives -= corrected_fn_last_bin

    # Add the section of the confusion matrix untouched by the oracle.
    expected_true_positives += tf.reduce_sum(
        tf.where(cum_examples > num_oracle_examples, self.binned_true_positives,
                 0.0),
        axis=1)
    expected_true_negatives += tf.reduce_sum(
        tf.where(cum_examples > num_oracle_examples, self.binned_true_negatives,
                 0.0),
        axis=1)
    expected_false_positives += tf.reduce_sum(
        tf.where(cum_examples > num_oracle_examples,
                 self.binned_false_positives, 0.0),
        axis=1)
    expected_false_negatives += tf.reduce_sum(
        tf.where(cum_examples > num_oracle_examples,
                 self.binned_false_negatives, 0.0),
        axis=1)

    # Reset the first and last elements of the expected confusion matrix to get
    # the final confusion matrix. Because the thresholds for these entries are
    # outside [0, 1], they should be left untouched and not sent to the oracle.
    expected_true_positives = _replace_first_and_last_elements(
        expected_true_positives, tf.reduce_sum(self.binned_true_positives[0]),
        tf.reduce_sum(self.binned_true_positives[-1]))
    expected_true_negatives = _replace_first_and_last_elements(
        expected_true_negatives, tf.reduce_sum(self.binned_true_negatives[0]),
        tf.reduce_sum(self.binned_true_negatives[-1]))
    expected_false_positives = _replace_first_and_last_elements(
        expected_false_positives, tf.reduce_sum(self.binned_false_positives[0]),
        tf.reduce_sum(self.binned_false_positives[-1]))
    expected_false_negatives = _replace_first_and_last_elements(
        expected_false_negatives, tf.reduce_sum(self.binned_false_negatives[0]),
        tf.reduce_sum(self.binned_false_negatives[-1]))

    self.true_positives.assign(expected_true_positives)
    self.true_negatives.assign(expected_true_negatives)
    self.false_positives.assign(expected_false_positives)
    self.false_negatives.assign(expected_false_negatives)

    return super().result()

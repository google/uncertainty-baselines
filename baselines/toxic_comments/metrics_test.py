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
"""Tests for uncertainty_baselines.baselines.toxic_comments.metrics."""

import numpy as np
import tensorflow as tf
import metrics  # local file import


class OracleCollaborativeAUCTest(tf.test.TestCase):

  def setUp(self):
    super().setUp()

    self.y_true = np.array([0., 1., 0., 1., 0., 1., 1., 0.])
    self.y_pred = np.array([0.31, 0.42, 0.33, 0.84, 0.75, 0.86, 0.57, 0.68])

  def test_no_examples_ROC(self):
    num_thresholds = 7
    num_bins = 14
    oracle_auc_roc = metrics.OracleCollaborativeAUC(
        oracle_fraction=0.5,
        num_thresholds=num_thresholds,
        num_bins=14,
        curve='ROC')
    result = oracle_auc_roc.result()

    self.assertAllClose(oracle_auc_roc.binned_true_positives,
                        tf.zeros([num_thresholds, num_bins]))
    self.assertAllClose(oracle_auc_roc.true_positives,
                        tf.zeros([num_thresholds]))
    self.assertEqual(result, 0.)

  def test_no_examples_PR(self):
    num_thresholds = 8
    num_bins = 23
    oracle_auc_pr = metrics.OracleCollaborativeAUC(
        oracle_fraction=0.5,
        num_thresholds=num_thresholds,
        curve='PR',
        num_bins=num_bins)
    result = oracle_auc_pr.result()

    self.assertAllClose(oracle_auc_pr.binned_true_positives,
                        tf.zeros([num_thresholds, num_bins]))
    self.assertAllClose(oracle_auc_pr.true_positives,
                        tf.zeros([num_thresholds]))
    self.assertEqual(result, 0.)

  def test_reduces_to_AUC_zero_oracle_fraction(self):
    num_thresholds = 11
    oracle_auc = metrics.OracleCollaborativeAUC(
        oracle_fraction=0., num_thresholds=num_thresholds, num_bins=7)
    regular_auc = tf.keras.metrics.AUC(num_thresholds=num_thresholds)

    oracle_auc.update_state(self.y_true, self.y_pred)
    regular_auc.update_state(self.y_true, self.y_pred)

    self.assertAllClose(
        tf.reduce_sum(oracle_auc.binned_true_positives, axis=1),
        regular_auc.true_positives)
    self.assertAllClose(
        tf.reduce_sum(oracle_auc.binned_true_negatives, axis=1),
        regular_auc.true_negatives)
    self.assertAllClose(
        tf.reduce_sum(oracle_auc.binned_false_positives, axis=1),
        regular_auc.false_positives)
    self.assertAllClose(
        tf.reduce_sum(oracle_auc.binned_false_negatives, axis=1),
        regular_auc.false_negatives)

    oracle_auc_result = oracle_auc.result()
    regular_auc_result = regular_auc.result()

    self.assertAllClose(oracle_auc.true_positives, regular_auc.true_positives)
    self.assertAllClose(oracle_auc.true_negatives, regular_auc.true_negatives)
    self.assertAllClose(oracle_auc.false_positives, regular_auc.false_positives)
    self.assertAllClose(oracle_auc.false_negatives, regular_auc.false_negatives)
    self.assertEqual(oracle_auc_result, regular_auc_result)

  def test_ROC_perfect_AUC_with_unit_oracle_fraction(self):
    num_thresholds = 11
    curve = 'ROC'
    oracle_auc = metrics.OracleCollaborativeAUC(
        oracle_fraction=1.,
        num_thresholds=num_thresholds,
        num_bins=7,
        curve=curve)

    result = oracle_auc(self.y_true, self.y_pred)
    self.assertAllClose(oracle_auc.true_positives,
                        [sum(self.y_true == 1)] * (num_thresholds - 1) + [0])
    self.assertAllClose(oracle_auc.true_negatives,
                        [0] + [sum(self.y_true == 0)] * (num_thresholds - 1))
    self.assertAllClose(oracle_auc.false_positives,
                        [sum(self.y_true == 0)] + [0] * (num_thresholds - 1))
    self.assertAllClose(oracle_auc.false_negatives,
                        [0] * (num_thresholds - 1) + [sum(self.y_true == 1)])

    self.assertEqual(result, 1.)

  def test_PR_perfect_AUC_with_unit_oracle_fraction(self):
    num_thresholds = 11
    curve = 'PR'
    oracle_auc = metrics.OracleCollaborativeAUC(
        oracle_fraction=1.,
        num_thresholds=num_thresholds,
        num_bins=7,
        curve=curve)

    result = oracle_auc(self.y_true, self.y_pred)
    self.assertAllClose(oracle_auc.true_positives,
                        [sum(self.y_true == 1)] * (num_thresholds - 1) + [0])
    self.assertAllClose(oracle_auc.true_negatives,
                        [0] + [sum(self.y_true == 0)] * (num_thresholds - 1))
    self.assertAllClose(oracle_auc.false_positives,
                        [sum(self.y_true == 0)] + [0] * (num_thresholds - 1))
    self.assertAllClose(oracle_auc.false_negatives,
                        [0] * (num_thresholds - 1) + [sum(self.y_true == 1)])

    self.assertEqual(result, 1.)

  def test_PR_oracle_fraction_two_thirds(self):
    y_true = np.array([0., 0., 1., 1., 0., 1., 1., 0.])
    y_pred = np.array([0.31, 0.33, 0.42, 0.58, 0.69, 0.76, 0.84, 0.87])

    num_thresholds = 5  # -1e-7, 0.25, 0.5, 0.75, 1.0000001
    num_bins = 3
    curve = 'PR'
    oracle_auc = metrics.OracleCollaborativeAUC(
        oracle_fraction=0.67,  # floor(0.67 * 8) = 5 examples sent to oracle
        num_thresholds=num_thresholds,
        num_bins=num_bins,
        curve=curve)

    result = oracle_auc(y_true, y_pred)
    self.assertAllClose(
        oracle_auc.binned_true_positives,
        # y_true's positives are 0.42, 0.58, 0.76, and 0.84 in y_pred.
        np.array([
            [0., 2., 2.],  # Threshold -1e-7; bins are unmodified
            [2., 2., 0.],  # Threshold 0.25; bins [0, 0.58), [0.58, 0.91)
            [2., 1., 0.],  # Threshold 0.5: 0.42 is now a false positive.
            [2., 0., 0.],  # Threshold 0.75: only 0.76 and 0.84 are positive.
            [0., 0., 0.],  # Threshold 1.0000001: no positives.
        ]))
    self.assertAllClose(
        oracle_auc.binned_true_negatives,
        # The possible true negatives are 0.31, 0.33, 0.69, and 0.87.
        np.array([
            [0., 0., 0.],  # There are no negatives for threshold -1e-7.
            [0., 0., 0.],  # Threshold 0.25: still no negatives.
            [2., 0., 0.],  # Threshold 0.5: 0.31 and 0.33 are negative.
            [1., 2., 0.],  # Threshold 0.75: only 0.69 in first bin.
            [2., 0., 2.],  # Threshold 1.0000001: 0.76 and 0.84 in first bin.
        ]))
    self.assertAllClose(
        oracle_auc.binned_false_positives,
        # Compare these values with oracle_auc.binned_true_negatives.
        # For example, the total across their rows must always be 4.
        np.array([
            [2., 0., 2.],  # 0.76 and 0.84 in bin 3 (greater than -1e-7 + 0.66).
            [2., 2., 0.],  # Threshold 0.25: 0.76 and 0.84 move to second bin.
            [1., 1., 0.],  # Threshold 0.5: 0.76 (0.84) in first (second) bin.
            [1., 0., 0.],  # Threshold 0.75: only 0.87 remains in first bin.
            [0., 0., 0.],  # Threshold 1.0000001: no more positives.
        ]))
    self.assertAllClose(
        oracle_auc.binned_false_negatives,
        # Compare these values with oracle_auc.binned_true_positives.
        np.array([
            [0., 0., 0.],  # No negatives
            [0., 0., 0.],  # No negatives
            [1., 0., 0.],  # Threshold 0.5: only 0.42 is below threshold.
            [2., 0., 0.],  # Threshold 0.75: 0.42 still in bin 1; 0.58 joins it.
            [2., 2., 0.],  # Threshold 1.0000001: 0.42 and 0.58 in second bin.
        ]))

    # The first and last threshold are outside [0, 1] and are never corrected.
    # Second threshold: 0.5 corrected from fp to tn
    # Third threshold: 0.83 corrected from fp and fn each to tp and tn
    # Fourth threshold: 0.83 corrected from fp->tn, 1.67 corrected from fn->tp
    self.assertAllClose(oracle_auc.true_positives,
                        np.array([4., 4., 3. + 5 / 6, 2. + 5 / 3, 0.]))
    self.assertAllClose(oracle_auc.true_negatives,
                        np.array([0., 2. + 0.5, 2. + 5 / 6, 3. + 5 / 6, 4.]))
    self.assertAllClose(oracle_auc.false_positives,
                        np.array([4., 2. - 0.5, 2. - 5 / 6, 1. - 5 / 6, 0.]))
    self.assertAllClose(oracle_auc.false_negatives,
                        np.array([0., 0., 1. - 5 / 6, 2. - 5 / 3, 4.]))

    self.assertEqual(result, 0.9434595)

  def test_custom_binning_score(self):
    y_true = np.array([1., 0., 0., 1.])
    y_pred = np.array([0.31, 0.32, 0.83, 0.64])

    oracle_auc = metrics.OracleCollaborativeAUC(
        oracle_fraction=0.5,  # 2 examples sent to oracle
        num_bins=4,  # (-inf, 0.25), [0.25, 0.5), [0.5, 0.75), [0.75, inf)
        num_thresholds=4,  # -1e-7, 0.33, 0.67, 1.0000001
    )

    # This custom_binning_score means 0.31 and 0.32 are always sent to oracle.
    result = oracle_auc(y_true, y_pred, custom_binning_score=y_pred)

    self.assertAllClose(
        oracle_auc.binned_true_positives,
        # y_true's positives are 0.31 and 0.64 in y_pred.
        np.array([
            [0., 1., 1., 0.],
            [0., 0., 1., 0.],  # 0.31 is no longer above threshold 0.33
            [0., 0., 0., 0.],  # 0.64 is below threshold 0.67
            [0., 0., 0., 0.],
        ]))
    self.assertAllClose(
        oracle_auc.binned_true_negatives,
        # The possible true negatives are 0.32 and 0.83.
        np.array([
            [0., 0., 0., 0.],
            [0., 1., 0., 0.],  # 0.32 is below threshold 0.33
            [0., 1., 0., 0.],  # 0.84 is still above threshold 0.67
            [0., 1., 0., 1.],
        ]))
    self.assertAllClose(
        oracle_auc.binned_false_positives,
        # Compare these values with oracle_auc.binned_true_negatives.
        # For example, the total across their rows must always be 2.
        np.array([
            [0., 1., 0., 1.],  # 0.32 and 0.84 are both above threshold -1e-7
            [0., 0., 0., 1.],  # 0.32 moves to true_negatives
            [0., 0., 0., 1.],  # 0.84 still above threshold
            [0., 0., 0., 0.],  # all examples moved to true_negatives
        ]))
    self.assertAllClose(
        oracle_auc.binned_false_negatives,
        # Compare these values with oracle_auc.binned_true_positives.
        np.array([
            [0., 0., 0., 0.],
            [0., 1., 0., 0.],  # 0.31 becomes a false negative at threshold 0.33
            [0., 1., 1., 0.],  # 0.64 becomes a false negative at threshold 0.67
            [0., 1., 1., 0.],
        ]))

    # 0.31 is always corrected from false_positives to true_negatives.
    self.assertAllClose(oracle_auc.true_positives, np.array([2., 2., 1., 0.]))
    self.assertAllClose(oracle_auc.true_negatives, np.array([0., 1., 1., 2.]))
    self.assertAllClose(oracle_auc.false_positives, np.array([2., 1., 1., 0.]))
    self.assertAllClose(oracle_auc.false_negatives, np.array([0., 0., 1., 2.]))

    self.assertEqual(result, 0.625)

  def test_monotonic_with_increasing_oracle_fraction_and_dtype(self):
    y_true = np.array([1., 0., 0., 1., 1., 0., 1., 0., 1.])
    y_pred = np.array([0.11, 0.62, 0.33, 0.74, 0.35, 0.26, 0.67, 0.58, 0.89])
    tf_dtype = tf.float16
    np_dtype = np.float16

    auc00, auc03, auc06, auc09 = [
        metrics.OracleCollaborativeAUC(
            oracle_fraction=frac, num_thresholds=11, dtype=tf_dtype)
        for frac in np.array([0.0, 0.3, 0.6, 0.9])
    ]

    result00, result03, result06, result09 = [
        auc(y_true, y_pred) for auc in (auc00, auc03, auc06, auc09)
    ]

    self.assertDTypeEqual(auc00.binned_true_positives, np_dtype)
    self.assertDTypeEqual(auc00.true_positives, np_dtype)
    self.assertDTypeEqual(result00, np_dtype)
    self.assertBetween(result00, minv=0., maxv=result03)
    self.assertBetween(result06, minv=result03, maxv=result09)
    self.assertLessEqual(result09, 1.)


if __name__ == '__main__':
  tf.test.main()

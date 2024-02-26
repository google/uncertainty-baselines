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

"""Tests for metrics."""

import tensorflow as tf
import metrics  # local file import from experimental.shoshin

from google3.testing.pybase import googletest


class MetricsTest(tf.test.TestCase):

  def test_one_vs_rest_auc(self):
    auc = tf.keras.metrics.AUC()
    one_vs_rest_auc = metrics.OneVsRest(tf.keras.metrics.AUC(), 1)

    y_true = tf.constant([1, 0, 1, 1], dtype=tf.int32)
    one_hot_y_true = tf.one_hot(y_true, depth=2)
    y_pred = tf.constant([0.9, 0.3, 0.7, 0.2], dtype=tf.float32)
    one_hot_y_pred = tf.constant(
        [[0.1, 0.9], [0.7, 0.3], [0.3, 0.7], [0.8, 0.2]], dtype=tf.float32
    )

    auc.update_state(y_true, y_pred)
    expected_result = auc.result()

    one_vs_rest_auc.update_state(one_hot_y_true, one_hot_y_pred)
    result = one_vs_rest_auc.result()
    self.assertAllClose(result, expected_result)

  def test_one_vs_rest_aucpr(self):
    auc = tf.keras.metrics.AUC(curve="PR")
    one_vs_rest_auc = metrics.OneVsRest(tf.keras.metrics.AUC(curve="PR"), 1)

    y_true = tf.constant([1, 0, 1, 1], dtype=tf.int32)
    one_hot_y_true = tf.one_hot(y_true, depth=2)
    y_pred = tf.constant([0.9, 0.3, 0.7, 0.2], dtype=tf.float32)
    one_hot_y_pred = tf.constant(
        [[0.1, 0.9], [0.7, 0.3], [0.3, 0.7], [0.8, 0.2]], dtype=tf.float32
    )

    auc.update_state(y_true, y_pred)
    expected_result = auc.result()

    one_vs_rest_auc.update_state(one_hot_y_true, one_hot_y_pred)
    result = one_vs_rest_auc.result()
    self.assertAllClose(result, expected_result)

if __name__ == "__main__":
  googletest.main()

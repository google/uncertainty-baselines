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

"""Tests for utils."""

import os

from absl.testing import parameterized
import numpy as np
import robustness_metrics as rm
import tensorflow as tf
import tensorflow_datasets as tfds
import utils  # local file import from baselines.drug_cardiotoxicity


class UtilsTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.parameters(('ece', 0.1), ('brier', 0.02), ('accuracy', 1.0))
  def test_get_metric_result_value(self, metric_name, expected_result):
    probs = np.array([[0.1, 0.9]])
    labels = np.array([[0, 1]])

    if metric_name == 'ece':
      metric = rm.metrics.ExpectedCalibrationError(num_bins=10)
      metric.add_batch(probs[:, 1], label=labels[:, 1])
    elif metric_name == 'brier':
      metric = rm.metrics.Brier()
      metric.add_batch(probs, label=labels[:, 1])
    elif metric_name == 'accuracy':
      metric = tf.keras.metrics.CategoricalAccuracy()
      metric.update_state(labels, probs)

    result = utils.get_metric_result_value(metric)
    self.assertAllClose(result, expected_result)


  def test_write_params(self):
    test_output_dir = self.create_tempdir().full_path
    filename = os.path.join(test_output_dir, 'test_params.json')
    utils.write_params({'a': 1.0}, filename)
    self.assertTrue(tf.io.gfile.exists(filename))


if __name__ == '__main__':
  tf.test.main()

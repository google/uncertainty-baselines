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

"""Tests for subpopl_utils."""

import collections

import tensorflow as tf

import subpopl_utils  # local file import from baselines.jft
from google3.testing.pybase import googletest


class SubpoplUtilsTest(googletest.TestCase):

  def test_eval_subpopl_metrics(self):

    def addition_map_fn_builder(increment):

      def addition_map_fn(element):
        return element + increment

      return addition_map_fn

    def dict_map_fn(element):
      return collections.OrderedDict(
          image=element, labels=element, batch=element, mask=element)

    # Construct simple but slightly different datasets for 10 subpopulations.
    splits = {
        i: tf.data.Dataset.range(5).map(
            addition_map_fn_builder(i)).batch(3).map(dict_map_fn)
        for i in range(10)
    }

    # Create a predictable evaluation function that treats examples above "10"
    # correct.
    def evaluation_fn(dummy_params, image, labels, mask):
      del dummy_params, labels, mask
      n_correct = tf.reduce_sum(tf.cast(tf.greater(image, 10), tf.int32))
      n = tf.size(image)
      return [n_correct], None, [n], None

    measurements = subpopl_utils.eval_subpopl_metrics(splits, evaluation_fn,
                                                      None, 0)

    # Subpops 0-6 will have 0 correct, Subpops 7, 8, 9 will have 1, 2, 3,
    # respectively. There are 50 examples total. Percentiles are calculated
    # via interpolation.
    expected_measurements = {}
    expected_measurements['subpopl_avg_prec@1'] = 0.12
    expected_measurements['subpopl_med_prec@1'] = 0.0
    expected_measurements['subpopl_var_prec@1'] = 0.0416
    expected_measurements['subpopl_p95_prec@1'] = 0.51
    expected_measurements['subpopl_p75_prec@1'] = 0.15
    expected_measurements['subpopl_p25_prec@1'] = 0.0
    expected_measurements['subpopl_p05_prec@1'] = 0.0

    self.assertCountEqual(expected_measurements.keys(), measurements.keys())
    for key in expected_measurements:
      self.assertAlmostEqual(expected_measurements[key], measurements[key])


if __name__ == '__main__':
  googletest.main()

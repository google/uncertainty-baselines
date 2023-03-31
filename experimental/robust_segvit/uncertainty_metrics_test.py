# coding=utf-8
# Copyright 2023 The Uncertainty Baselines Authors.
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

"""Tests for uncertainty_metrics."""

from absl.testing import absltest
from absl.testing import parameterized
import jax.numpy as jnp
from uncertainty_metrics import reduce_2dmap  # local file import from experimental.robust_segvit
from uncertainty_metrics import SegmentationUncertaintyMetrics  # local file import from experimental.robust_segvit


class UncertaintyMetricsTest(parameterized.TestCase):

  def setUp(self):
    super(UncertaintyMetricsTest, self).setUp()
    self.targets = jnp.asarray([[[1, 2, 5, 7], [6, 4, 3, 3], [10, 9, 5, 0],
                                 [8, 6, 4, 4]]])

    self.preds = jnp.asarray([[[1, 2, 4, 7], [5, 6, 3, 3], [10, 9, 4, 0],
                               [8, 7, 3, 4]]])

    self.unc_map = jnp.asarray([[[0.1, 0.3, 0.6, 0.3], [0.7, 0.6, 0.2, 0.1],
                                 [0.2, 0.4, 0.5, 0.3], [0.1, 0.7, 0.6, 0.2]]])

    # create logit map from unc_map by mapping entropy vals (0.1,0.7)
    # to a feasible range of logit vals:(4.1, 6.2)
    self.logit_map = 4.1 + (0.7 - self.unc_map) * (6.2 - 4.1) / (0.7 - 0.1)

    self.window_size = 2
    self.accuracy_th = 0.5
    self.uncertainty_th = 0.4

    # true values
    self.true_binary_acc_map = jnp.asarray([[[0, 1], [1, 0]]])
    self.true_binary_unc_map = jnp.asarray([[[1, 0], [0, 0]]])
    self.true_p_accurate_certain = jnp.asarray([0.67])
    self.true_p_uncertain_innacurate = jnp.asarray([0.5])
    self.true_pavpu = jnp.asarray([0.75])

    # construct logits passed as input from unc_map
    self.num_classes = 11
    self.img_size = 4

    true_mask = jnp.arange(self.img_size * self.img_size
                          ) * self.num_classes + self.preds.flatten()
    logits = jnp.zeros((self.img_size * self.img_size * self.num_classes))
    logits = logits.at[true_mask].set(self.logit_map.flatten())
    self.logits = jnp.expand_dims(
        logits.reshape((self.img_size, self.img_size, self.num_classes)), 0)

  def test_setup(self):
    preds_logits = jnp.argmax(self.logits, -1)
    self.assertTrue(jnp.array_equal(self.preds, preds_logits))

  def test_calculate_pacc_cert(self):
    segment_unc = SegmentationUncertaintyMetrics(
        logits=self.logits,
        labels=self.targets,
        window_size=self.window_size,
        accuracy_th=self.accuracy_th,
        uncertainty_th=self.uncertainty_th)

    self.assertEqual(self.true_pavpu, segment_unc.pavpu)
    self.assertAlmostEqual(self.true_p_accurate_certain, segment_unc.pacc_cert,
                           2)
    self.assertAlmostEqual(self.true_p_uncertain_innacurate,
                           segment_unc.puncert_inacc, 2)

  @parameterized.parameters((1), (2), (3))
  def test_reduce_2dmap(self, batch_size):
    array_map = jnp.repeat(jnp.ones((1, 4, 4)), batch_size, axis=0)
    true_binary_map = jnp.repeat(jnp.ones((1, 2, 2)), batch_size, axis=0)
    binary_map = reduce_2dmap(array_map, self.window_size, self.accuracy_th)

    self.assertTrue(jnp.array_equal(true_binary_map, binary_map))


if __name__ == '__main__':
  absltest.main()

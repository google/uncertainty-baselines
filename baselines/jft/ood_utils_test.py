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

"""Tests for ood_utils."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np
import ood_utils  # local file import from baselines.jft


class OodUtilsTest(parameterized.TestCase):

  def setUp(self):
    super(OodUtilsTest, self).setUp()
    self.means = np.array([[-1, 0], [1, 0]])
    self.cov = np.identity(2)

  def test_ood_metric(self):
    dataset_name = "cifar10"
    method_name = "msp"
    metric_name = f"{dataset_name}_{method_name}"
    ood_metric = ood_utils.OODMetric(dataset_name, method_name)
    scores = [0.2, 0.4, 0.35, 0.1]
    labels = [0, 0, 1, 1]
    ood_metric.update(scores, labels)

    self.assertEqual(metric_name, ood_metric.get_metric_name())
    self.assertEqual((scores, labels), ood_metric.get_scores_and_labels())
    self.assertDictEqual(
        ood_metric.compute_metrics(),
        {
            "auroc": 0.25,
            "auprc": 0.5,
            "fprn": 1.0
        },
    )

  def test_compute_mean_and_cov(self):
    n_sample = 1000
    embeds_list, labels_list = [], []
    for class_id in range(2):
      embeds_list.append(
          np.random.multivariate_normal(
              self.means[class_id], self.cov, n_sample
          )
      )
      labels_list += [class_id] * n_sample
    embeds = np.vstack(embeds_list)
    labels = np.array(labels_list)
    class_ids = jnp.unique(labels)
    means, cov = ood_utils.compute_mean_and_cov(embeds, labels, class_ids)
    for a, b in zip(means, self.means):
      np.testing.assert_allclose(a, b, atol=0.1)
    np.testing.assert_allclose(cov, self.cov, atol=0.1)

  def test_compute_mahalanobis_distance(self):
    embeds = np.array([[-1, 0], [1, 0], [10, 0]])
    dists = jax.jit(ood_utils.compute_mahalanobis_distance)(
        embeds, self.means, self.cov
    )
    np.testing.assert_array_equal(np.array([0, 0, 81]), np.min(dists, axis=-1))


if __name__ == "__main__":
  absltest.main()

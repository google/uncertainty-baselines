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

"""Tests for init2winit.halton."""

from absl.testing import absltest
import numpy as np
from scipy import stats
from uncertainty_baselines import halton



class HaltonTest(absltest.TestCase):
  """Tests for generating Halton sequences."""

  def setUp(self):
    super().setUp()
    # While the Halton sequence generation is deterministic for any given
    # inputs, we need to set the numpy seed for stats.kstest and stats.ks_2samp.
    np.random.seed(1337)

  def testPrimes(self):
    primes = halton.generate_primes(100)
    # For each prime, make sure it is not evenly divisible by any number less
    # than itself.
    for p in primes:
      for n in range(2, int(p ** 0.5) + 1):
        self.assertNotEqual(0, p % n)

  def testHalton(self):
    sequence = halton.generate_sequence(num_samples=100, num_dims=4)
    self.assertLess(max(max(sequence)), 1)
    self.assertGreater(min(min(sequence)), 0)
    self.assertLen(sequence, 100)
    self.assertLen(sequence[0], 4)

  def testUniformness(self):
    """Perform a Kolmogorov-Smirnov test to check for sufficient uniformness."""
    alpha = 0.1
    num_tests = 100
    sequence = halton.generate_sequence(num_samples=10000, num_dims=num_tests)
    sequence = np.array(sequence)

    for num_subsamples in [2, 10, 50, 100, 250, 1000]:
      p_values = []
      for d in range(num_tests):
        _, p_value = stats.kstest(sequence[:num_subsamples, d], cdf='uniform')
        p_values.append(p_value)
      # Apply the Bonferroni correction (if any p-value is less than alpha / N,
      # we reject the null hypothesis that the sampes are uniform).
      is_not_uniform = min(p_values) < alpha / num_tests
      self.assertFalse(is_not_uniform)



if __name__ == '__main__':
  absltest.main()

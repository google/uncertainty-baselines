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

"""Tests for fewshot_utils."""

from absl.testing import absltest
from absl.testing import parameterized
import fewshot_utils  # local file import from baselines.jft


class FewshotUtilsTest(parameterized.TestCase):

  def test_select_best_l2_reg(self):
    shots_list = [5]
    l2_regs = [1.0, 2.0, 3.0]
    results = {
        'birds': {
            (5, 1.0): {'test_prec@1': 0.5},
            (5, 2.0): {'test_prec@1': 0.6},
            (5, 3.0): {'test_prec@1': 0.3},
        },
        'cars': {
            (5, 1.0): {'test_prec@1': 0.6},
            (5, 2.0): {'test_prec@1': 0.7},
            (5, 3.0): {'test_prec@1': 0.65},
        },
        'cifar100': {
            (5, 1.0): {'test_prec@1': 0.8},
            (5, 2.0): {'test_prec@1': 0.7},
            (5, 3.0): {'test_prec@1': 0.6},
        }
    }

    # Selects best l2 based on all datasets.
    best_l2 = fewshot_utils.select_best_l2_reg(
        results, shots_list, l2_regs, how='all')
    expected_best_l2 = {
        ('birds', 5): 2.0,
        ('cars', 5): 2.0,
        ('cifar100', 5): 2.0,
    }
    self.assertEqual(best_l2, expected_best_l2)

    # Selects best l2 based on all datasets except itself.
    best_l2 = fewshot_utils.select_best_l2_reg(
        results, shots_list, l2_regs, how='leave-self-out')
    expected_best_l2 = {
        ('birds', 5): 2.0,
        ('cars', 5): 1.0,
        ('cifar100', 5): 2.0,
    }
    self.assertEqual(best_l2, expected_best_l2)

    # Tests a non-supported l2 selection scheme.
    with self.assertRaises(ValueError):
      fewshot_utils.select_best_l2_reg(
          results, shots_list, l2_regs, how='bad-scheme')


if __name__ == '__main__':
  absltest.main()

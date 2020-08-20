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
"""Tests for CIFAR{10,100}."""

import tensorflow as tf
import uncertainty_baselines as ub


class CifarDatasetTest(ub.datasets.DatasetTest):

  def testCifar10DatasetSize(self):
    super(CifarDatasetTest, self)._testDatasetSize(
        ub.datasets.Cifar10Dataset, (32, 32, 3), validation_percent=0.1)

  def testCifar100DatasetSize(self):
    super(CifarDatasetTest, self)._testDatasetSize(
        ub.datasets.Cifar100Dataset, (32, 32, 3), validation_percent=0.1)


if __name__ == '__main__':
  tf.test.main()

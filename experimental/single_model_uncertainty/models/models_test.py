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
"""Tests for single_model_uncertainty.models.get()."""

import tensorflow as tf
import models  # local file import


class ModelsTest(tf.test.TestCase):

  def testGetModel(self):
    model = models.get(
        'genomics_cnn',
        batch_size=8,
        len_seqs=250,
        num_motifs=16,
        len_motifs=20,
        num_denses=32)
    self.assertEqual(12, len(model.layers))


if __name__ == '__main__':
  tf.test.main()

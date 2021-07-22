# coding=utf-8
# Copyright 2021 The Uncertainty Baselines Authors.
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

"""Tests for uncertainty_baselines.models.genomicscnn."""

from absl.testing import parameterized
import tensorflow as tf
import uncertainty_baselines as ub


class GenomicsCNNTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(('one_hot', True), ('embedding', False))
  def testCreateModel(self, one_hot):
    batch_size = 31
    num_classes = 10
    num_motifs = 16
    len_motifs = 20
    num_denses = 6
    seq_len = 250

    # generate random input data composed by {0, 1, 2, 3}
    rand_nums = tf.random.categorical(
        tf.math.log([[0.2, 0.3, 0.3, 0.2]]), batch_size * seq_len)
    rand_data = tf.reshape(rand_nums, (batch_size, seq_len))

    model = ub.models.genomics_cnn(
        batch_size=batch_size,
        num_classes=num_classes,
        num_motifs=num_motifs,
        len_motifs=len_motifs,
        num_denses=num_denses,
        one_hot=one_hot)
    logits = model(rand_data)
    self.assertEqual(logits.shape, (batch_size, num_classes))


if __name__ == '__main__':
  tf.test.main()

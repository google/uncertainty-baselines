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

"""Tests for mpnn."""
import tensorflow as tf

import uncertainty_baselines as ub


class MpnnTest(tf.test.TestCase):

  def setUp(self):
    super().setUp()
    self.random_seed = 42

    self.num_classes = 2
    self.batch_size = 4
    self.max_nodes = 30
    self.node_dim = 10
    self.edge_dim = 12
    self.num_layers = 2
    self.message_layer_size = 20
    self.readout_layer_size = 20

  def test_mpnn_base_model(self):
    """Tests if MPNN can be compiled successfully."""

    # Compiles classifier model.
    model = ub.models.mpnn(
        node_feature_dim=self.node_dim,
        num_heads=self.num_classes,
        num_layers=self.num_layers,
        message_layer_size=self.message_layer_size,
        readout_layer_size=self.readout_layer_size)

    # Computes output.
    tf.random.set_seed(self.random_seed)
    inputs = {
        "atoms":
            tf.random.normal((self.batch_size, self.max_nodes, self.node_dim)),
        "pairs":
            tf.random.normal((self.batch_size, self.max_nodes, self.max_nodes,
                              self.edge_dim))
    }
    logits = model(inputs, training=False)

    # Check if output tensors have correct shapes.
    logits_shape_observed = logits.shape.as_list()
    logits_shape_expected = [self.batch_size, self.num_classes]

    self.assertEqual(logits_shape_observed, logits_shape_expected)

if __name__ == "__main__":
  tf.test.main()

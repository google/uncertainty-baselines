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

"""Tests for augmentation_utils."""

import numpy as np
import tensorflow as tf
import augmentation_utils  # local file import from baselines.drug_cardiotoxicity
from uncertainty_baselines.datasets import drug_cardiotoxicity

from google3.testing.pybase import googletest


NUM_MOLECULES = 3


class AugmentationUtilsTest(googletest.TestCase):

  def setUp(self):
    super().setUp()
    # Prepares features of a single molecule.
    single_molecule_atom_mask = np.zeros(
        drug_cardiotoxicity._MAX_NODES, dtype=np.float32)
    # Set first half of nodes to valid (== 30 nodes).
    num_valid_nodes = int(drug_cardiotoxicity._MAX_NODES * 0.5)
    single_molecule_atom_mask[:num_valid_nodes] = 1.

    single_molecule_atoms = np.zeros((drug_cardiotoxicity._MAX_NODES,
                                      drug_cardiotoxicity._NODE_FEATURE_LENGTH),
                                     dtype=np.float32)
    # Set first half of features of valid nodes to 1 (== 13 features).
    num_valid_node_features = int(drug_cardiotoxicity._NODE_FEATURE_LENGTH *
                                  0.5)
    single_molecule_atoms[:num_valid_nodes, :num_valid_node_features] = 1

    single_molecule_pair_mask = np.zeros(
        (drug_cardiotoxicity._MAX_NODES, drug_cardiotoxicity._MAX_NODES),
        dtype=np.float32)
    # Add 8 bidirectional edges between: nodes 0 and 1, 2 and 4, 2 and 5,
    # 5 and 6, 8 and 9, 8 and 10, 8 and 13, 13 and 14.
    source_nodes = [0, 1, 2, 4, 2, 5, 5, 6, 8, 9, 8, 10, 8, 13, 13, 14]
    target_nodes = [1, 0, 4, 2, 5, 2, 6, 5, 9, 8, 10, 8, 13, 8, 14, 13]
    single_molecule_pair_mask[source_nodes, target_nodes] = 1.

    single_molecule_pairs = np.zeros(
        (drug_cardiotoxicity._MAX_NODES, drug_cardiotoxicity._MAX_NODES,
         drug_cardiotoxicity._EDGE_FEATURE_LENGTH),
        dtype=np.float32)
    # For each valid edge, set first half of edge features to 1 (6 features).
    num_valid_edge_features = int(drug_cardiotoxicity._EDGE_FEATURE_LENGTH *
                                  0.5)
    single_molecule_pairs[source_nodes,
                          target_nodes, :num_valid_edge_features] = 1.

    single_molecule_id = b'testMolecule'

    # Creates graphs.
    self.valid_graph = {
        'atom_mask': tf.constant([single_molecule_atom_mask] * NUM_MOLECULES),
        'atoms': tf.constant([single_molecule_atoms] * NUM_MOLECULES),
        'pair_mask': tf.constant([single_molecule_pair_mask] * NUM_MOLECULES),
        'pairs': tf.constant([single_molecule_pairs] * NUM_MOLECULES),
        'molecule_id': tf.constant([single_molecule_id] * NUM_MOLECULES),
    }

    self.empty_graph = {
        'atom_mask': tf.constant([]),
        'atoms': tf.constant([]),
        'pair_mask': tf.constant([]),
        'pairs': tf.constant([]),
        'molecule_id': tf.constant([]),
    }
    self.graph_augmenter_ratio_0 = augmentation_utils.GraphAugment(
        ['drop_nodes'], aug_ratio=0.0)
    self.graph_augmenter_empty_aug = augmentation_utils.GraphAugment(
        [], aug_ratio=0.5)
    self.graph_augmenter = augmentation_utils.GraphAugment(['drop_nodes'],
                                                           aug_ratio=0.5)

  def test_drop_ratio_0(self):
    # Drop no nodes.
    augmented_graph, idx_dropped_nodes = self.graph_augmenter_ratio_0.augment(
        self.valid_graph)
    # Check that none of the features have changed.
    for feature in augmented_graph:
      np.testing.assert_array_equal(augmented_graph[feature],
                                    self.valid_graph[feature])
      # Check that returned arrays of dropped node indices are empty.
      self.assertEmpty(idx_dropped_nodes)

  def test_drop_augmentations_empty(self):
    # Drop no nodes.
    augmented_graph, idx_dropped_nodes = self.graph_augmenter_empty_aug.augment(
        self.valid_graph)
    # Check that none of the features have changed.
    for feature in augmented_graph:
      np.testing.assert_array_equal(augmented_graph[feature],
                                    self.valid_graph[feature])
      # Check that returned arrays of dropped node indices are empty.
      self.assertEmpty(idx_dropped_nodes)

  def _check_no_nonzero_features(self,
                                 features: np.ndarray,
                                 dropped_nodes: np.ndarray,
                                 axis: int = 0):
    """Check that the features have no nonzero values.

    Args:
      features: Numpy array containing features to check.
      dropped_nodes: Numpy array of indices of dropped nodes.
      axis: Axis of features to check. When edges, axis==0 means passed nodes
        are considered source nodes. When axis==1, they are target nodes.
    """
    features_of_dropped_nodes = np.take(features, dropped_nodes, axis=axis)
    self.assertEqual(0, np.nonzero(features_of_dropped_nodes)[0].shape[0])
    self.assertEqual(0, np.nonzero(features_of_dropped_nodes)[1].shape[0])

  def test_drop_nodes(self):
    # Drop 50% of nodes == 15 nodes.
    augmented_graph, idx_dropped_nodes = self.graph_augmenter.drop_nodes(
        self.valid_graph)
    expected_num_dropped = int(drug_cardiotoxicity._MAX_NODES * 0.5 * 0.5)
    for molecule_idx in range(NUM_MOLECULES):
      nodes_left = np.nonzero(
          augmented_graph['atom_mask'][molecule_idx].numpy())[0]
      # Check that only half of nodes remain.
      np.testing.assert_equal(idx_dropped_nodes[molecule_idx].shape[0],
                              nodes_left.shape[0])
      np.testing.assert_equal(idx_dropped_nodes[molecule_idx].shape[0],
                              expected_num_dropped)
      np.testing.assert_equal(nodes_left.shape[0], expected_num_dropped)

      # Check that all removed node features are 0.
      self._check_no_nonzero_features(
          augmented_graph['atoms'][molecule_idx],
          idx_dropped_nodes[molecule_idx],
          axis=0)

      # Check that dropped nodes are not source nodes.
      self._check_no_nonzero_features(
          augmented_graph['pair_mask'][molecule_idx],
          idx_dropped_nodes[molecule_idx],
          axis=0)

      # Check that the edge features are 0.
      self._check_no_nonzero_features(
          augmented_graph['pairs'][molecule_idx],
          idx_dropped_nodes[molecule_idx],
          axis=0)

      # Check that dropped nodes are not target nodes.
      self._check_no_nonzero_features(
          augmented_graph['pair_mask'][molecule_idx],
          idx_dropped_nodes[molecule_idx],
          axis=1)

      # Check that the edge features are 0.
      self._check_no_nonzero_features(
          augmented_graph['pairs'][molecule_idx],
          idx_dropped_nodes[molecule_idx],
          axis=1)

  def test_empty_graph(self):
    _, idx_dropped_nodes = self.graph_augmenter.drop_nodes(self.empty_graph)
    # Check that returned arrays of dropped node indices are empty.
    self.assertEmpty(idx_dropped_nodes)


if __name__ == '__main__':
  googletest.main()

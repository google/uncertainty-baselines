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

from typing import Optional

import numpy as np
import tensorflow as tf
import augmentation_utils  # local file import from baselines.drug_cardiotoxicity
from uncertainty_baselines.datasets import drug_cardiotoxicity

from google3.testing.pybase import googletest


NUM_MOLECULES = 3
AUG_RATIO = 0.5
AUG_PROB = 1.
AUGMENTATIONS = [
    'drop_nodes', 'perturb_edges', 'permute_edges', 'mask_node_features',
    'subgraph'
]
VALID_NODES = [0, 2, 3, 5]


class AugmentationUtilsTest(googletest.TestCase):

  def setUp(self):
    super().setUp()
    # Case 1: Valid nodes only. Prepares nodes of a single molecule with valid
    # nodes stacked at the top. This the expected state of the atom_mask feature
    # before any augmentations are applied.
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
    single_molecule_atoms[:num_valid_nodes, :num_valid_node_features] = 1.

    # Prepare the edge features.
    single_molecule_pair_mask = np.zeros(
        (drug_cardiotoxicity._MAX_NODES, drug_cardiotoxicity._MAX_NODES),
        dtype=np.float32)
    # Add 8 bidirectional edges between: nodes 0 and 1, 2 and 4, 2 and 5,
    # 5 and 6, 8 and 9, 8 and 10, 8 and 13, 13 and 14.
    self.total_num_edges = 16
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

    # Case 2: Invalid nodes mixed in. Prepares nodes of a single molecule with
    # invalid nodes mixed in between valid nodes at the top of the atom_mask
    # feature. This case simulates the state of the nodes after augmentation(s)
    # have been applied, such as drop_nodes.
    single_molecule_atom_mask_with_invalid_nodes = np.zeros(
        drug_cardiotoxicity._MAX_NODES, dtype=np.float32)
    # Set nodes 0, 2, 3, and 5 to valid. Rest are invalid.
    single_molecule_atom_mask_with_invalid_nodes[VALID_NODES] = 1.
    # Set node features, keeping invalid node features 0.
    single_molecule_atoms_with_invalid_nodes = np.zeros(
        (drug_cardiotoxicity._MAX_NODES,
         drug_cardiotoxicity._NODE_FEATURE_LENGTH),
        dtype=np.float32)
    single_molecule_atoms_with_invalid_nodes[
        VALID_NODES, :num_valid_node_features] = 1.

    single_molecule_pair_mask_with_invalid_nodes = np.zeros(
        (drug_cardiotoxicity._MAX_NODES, drug_cardiotoxicity._MAX_NODES),
        dtype=np.float32)
    # Add 2 bidirectional edges between: nodes 0 and 2, 2 and 5.
    self.total_num_edges = 4
    source_nodes = [0, 2, 2, 5]
    target_nodes = [2, 0, 5, 2]
    single_molecule_pair_mask_with_invalid_nodes[source_nodes,
                                                 target_nodes] = 1.

    # Creates graphs.
    self.valid_graph = {
        'atom_mask': tf.constant([single_molecule_atom_mask] * NUM_MOLECULES),
        'atoms': tf.constant([single_molecule_atoms] * NUM_MOLECULES),
        'pair_mask': tf.constant([single_molecule_pair_mask] * NUM_MOLECULES),
        'pairs': tf.constant([single_molecule_pairs] * NUM_MOLECULES),
        'molecule_id': tf.constant([single_molecule_id] * NUM_MOLECULES),
    }

    self.graph_with_invalid_nodes = {
        'atom_mask':
            tf.constant([single_molecule_atom_mask_with_invalid_nodes] *
                        NUM_MOLECULES),
        'atoms':
            tf.constant([single_molecule_atoms_with_invalid_nodes] *
                        NUM_MOLECULES),
        'pair_mask':
            tf.constant([single_molecule_pair_mask_with_invalid_nodes] *
                        NUM_MOLECULES),
        'pairs':
            tf.constant([single_molecule_pairs] * NUM_MOLECULES),
        'molecule_id':
            tf.constant([single_molecule_id] * NUM_MOLECULES),
    }

    self.empty_graph = {
        'atom_mask': tf.constant([]),
        'atoms': tf.constant([]),
        'pair_mask': tf.constant([]),
        'pairs': tf.constant([]),
        'molecule_id': tf.constant([]),
    }
    self.graph_augmenter_ratio_0 = augmentation_utils.GraphAugment(
        AUGMENTATIONS, aug_ratio=0.0, aug_prob=AUG_PROB,
        perturb_edge_features=True)
    self.graph_augmenter_prob_0 = augmentation_utils.GraphAugment(
        AUGMENTATIONS, aug_ratio=AUG_RATIO, aug_prob=0.)
    self.graph_augmenter_empty_aug = augmentation_utils.GraphAugment(
        [], aug_ratio=AUG_RATIO, aug_prob=AUG_PROB)
    self.graph_augmenter_perturb_no_features = augmentation_utils.GraphAugment(
        AUGMENTATIONS,
        aug_ratio=AUG_RATIO,
        aug_prob=AUG_PROB,
        perturb_node_features=False,
        perturb_edge_features=False)
    self.graph_augmenter_perturb_features = augmentation_utils.GraphAugment(
        AUGMENTATIONS,
        aug_ratio=AUG_RATIO,
        aug_prob=AUG_PROB,
        perturb_node_features=True,
        perturb_edge_features=True)

  def test_drop_ratio_0(self):
    # Drop no nodes.
    augmented_graph = self.graph_augmenter_ratio_0.augment(
        self.valid_graph)
    # Check that none of the features have changed.
    for feature in augmented_graph:
      np.testing.assert_array_equal(augmented_graph[feature],
                                    self.valid_graph[feature])

  def test_drop_prob_0(self):
    # Drop no nodes.
    augmented_graph = self.graph_augmenter_prob_0.augment(
        self.valid_graph)
    # Check that none of the features have changed.
    for feature in augmented_graph:
      np.testing.assert_array_equal(augmented_graph[feature],
                                    self.valid_graph[feature])

  def test_drop_augmentations_empty(self):
    # Drop no nodes.
    augmented_graph = self.graph_augmenter_empty_aug.augment(
        self.valid_graph)
    # Check that none of the features have changed.
    for feature in augmented_graph:
      np.testing.assert_array_equal(augmented_graph[feature],
                                    self.valid_graph[feature])

  def _check_no_nonzero_features(self,
                                 features: np.ndarray,
                                 dropped_idx: np.ndarray,
                                 axis: Optional[int] = None):
    """Check that the features have no nonzero values.

    Args:
      features: Numpy array containing features to check.
      dropped_idx: Numpy array of indices of dropped features.
      axis: Axis of features to check. When edges, axis==0 means passed nodes
        are considered source nodes. When axis==1, they are target nodes.
        When axis is None, dropped_idx is specifying an edge, not a node.
    """
    if axis is not None:
      # Dropped node.
      features_of_dropped_nodes = np.take(features, dropped_idx, axis=axis)
      self.assertEqual(0, np.nonzero(features_of_dropped_nodes)[0].shape[0])
      self.assertEqual(0, np.nonzero(features_of_dropped_nodes)[1].shape[0])
    else:
      # Dropped edge.
      features_of_dropped_edge = features[dropped_idx[0], dropped_idx[1]]
      self.assertEqual(0, np.nonzero(features_of_dropped_edge)[0].shape[0])

  def _check_nonzero_features_exist_for_edge(self, features: np.ndarray,
                                             added_edge_idx: np.ndarray):
    """Check that the features have no nonzero values.

    Args:
      features: Numpy array containing features to check.
      added_edge_idx: Numpy array representing index of edge, e.g. [0, 1].
    """
    features_of_added_edges = features[added_edge_idx[0], added_edge_idx[1]]
    self.assertGreater(np.nonzero(features_of_added_edges)[0].shape[0], 0)

  def test_update_features_of_dropped_nodes(self):
    idx_drop = np.asarray([[0], [2], [3], [5], [7], [8]])
    expected_num_nodes_left = int(drug_cardiotoxicity._MAX_NODES * 0.5 -
                                  idx_drop.shape[0])
    atom_mask = self.valid_graph['atom_mask'][0]
    atoms = self.valid_graph['atoms'][0]
    (aug_atom_mask, aug_atoms
    ) = self.graph_augmenter_perturb_features._update_features_of_dropped_nodes(
        idx_drop, atom_mask, atoms)

    nodes_left = np.nonzero(aug_atom_mask.numpy())[0]
    np.testing.assert_equal(expected_num_nodes_left, nodes_left.shape[0])

    # Check that all removed node features are 0.
    self._check_no_nonzero_features(aug_atoms, idx_drop, axis=0)

  def test_remove_edges_of_dropped_nodes(self):
    idx_drop = np.asarray([[0], [2], [3], [5], [7], [8]])
    pair_mask = self.valid_graph['pair_mask'][0]
    pairs = self.valid_graph['pairs'][0]
    (aug_pair_mask, aug_pairs
    ) = self.graph_augmenter_perturb_features._remove_edges_of_dropped_nodes(
        idx_drop, pair_mask, pairs)
    # Check that dropped nodes are not source or target nodes.
    self._check_no_nonzero_features(aug_pair_mask, idx_drop, axis=0)
    self._check_no_nonzero_features(aug_pair_mask, idx_drop, axis=1)

    # Check that the edge features are 0.
    self._check_no_nonzero_features(aug_pairs, idx_drop, axis=0)
    self._check_no_nonzero_features(aug_pairs, idx_drop, axis=1)

  def test_drop_nodes(self):
    # Set random seed so that the following calls of drop_nodes will drop
    # the same nodes to test for equality. Test does NOT rely on seed value.
    tf.random.set_seed(1)

    # Drop 50% of nodes == 15 nodes.
    augmented_graph, idx_dropped_nodes = self.graph_augmenter_perturb_features.drop_nodes(
        self.valid_graph)

    # Only difference between augmented_graph and
    # augmented_graph_with_unperturbed_features should be that `atoms`
    # features are untouched in the latter.
    tf.random.set_seed(1)  # Reset counter of internal tf.random kernel.
    (augmented_graph_with_unperturbed_features,
     idx_dropped_nodes_with_unperturbed_features
    ) = self.graph_augmenter_perturb_no_features.drop_nodes(self.valid_graph)

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
      np.testing.assert_array_equal(
          idx_dropped_nodes[molecule_idx],
          idx_dropped_nodes_with_unperturbed_features[molecule_idx])

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

      # Check that for augmented_graph_with_unperturbed_features has the same
      # node features as the original graph.
      np.testing.assert_array_equal(
          self.valid_graph['atoms'][molecule_idx],
          augmented_graph_with_unperturbed_features['atoms'][molecule_idx])

  def test_mask_nodes(self):
    # Mask 50% of nodes == 15 nodes.
    augmented_graph, idx_masked_nodes = self.graph_augmenter_perturb_features.mask_node_features(
        self.valid_graph)
    for molecule_idx in range(NUM_MOLECULES):
      masked_nodes = idx_masked_nodes[molecule_idx]
      for node in masked_nodes:
        augmented_features = augmented_graph['atoms'][molecule_idx][node]
        original_features = self.valid_graph['atoms'][molecule_idx][node]
        self.assertFalse(np.array_equal(augmented_features, original_features))

  def test_perturb_edges(self):
    # Set random seed so that the following calls of drop_nodes will drop
    # the same nodes to test for equality. Test does NOT rely on seed value.
    tf.random.set_seed(1)

    # Perturb 50% of edges == 4 nodes will be randomly dropped, and
    # 4 bidirectional edges will be re-added with original features from
    # the dropped nodes.
    augmented_graph, (
        idx_dropped_edges,
        idx_added_edges) = self.graph_augmenter_perturb_features.perturb_edges(
            self.valid_graph)

    # Only difference between augmented_graph and
    # augmented_graph_with_unperturbed_features should be that `pairs`
    # features are untouched in the latter.
    tf.random.set_seed(1)  # Reset counter of internal tf.random kernel.
    augmented_graph_with_unperturbed_features, (
        idx_dropped_edges_with_unperturbed_features,
        idx_added_edges_with_unperturbed_features
    ) = self.graph_augmenter_perturb_no_features.perturb_edges(self.valid_graph)

    for molecule_idx in range(NUM_MOLECULES):
      dropped_edges = idx_dropped_edges[molecule_idx].numpy()
      added_edges = idx_added_edges[molecule_idx].numpy()

      np.testing.assert_array_equal(
          dropped_edges,
          idx_dropped_edges_with_unperturbed_features[molecule_idx].numpy())
      np.testing.assert_array_equal(
          added_edges,
          idx_added_edges_with_unperturbed_features[molecule_idx].numpy())

      # Check that dropped edges have been removed.
      for edge in dropped_edges:
        dropped_edge_was_readded = np.isin(edge, added_edges)
        if not dropped_edge_was_readded[0] and not dropped_edge_was_readded[1]:
          self._check_no_nonzero_features(
              augmented_graph['pairs'][molecule_idx], edge)
          self._check_no_nonzero_features(
              augmented_graph['pair_mask'][molecule_idx], edge)

      # Check that added edges exist and have features from the dropped edges.
      orig_pairs = self.valid_graph['pairs'][molecule_idx]
      for edge_idx, edge in enumerate(added_edges):
        # Note there are twice as many added edges as dropped edges because the
        # added edges are bidirectional.
        dropped_edge = dropped_edges[edge_idx % dropped_edges.shape[0]]
        dropped_edge_feature = orig_pairs[dropped_edge[0], dropped_edge[1]]
        added_edge_feature = augmented_graph['pairs'][molecule_idx][edge[0],
                                                                    edge[1]]
        np.testing.assert_array_equal(dropped_edge_feature, added_edge_feature)

      # Check that for augmented_graph_with_unperturbed_features has the same
      # edge features as the original graph.
      np.testing.assert_array_equal(
          orig_pairs,
          augmented_graph_with_unperturbed_features['pairs'][molecule_idx])

  def test_perturb_edges_on_graph_with_invalid_nodes(self):
    graph_augmenter = augmentation_utils.GraphAugment(
        augmentations_to_use=['perturb_edges'],
        aug_ratio=AUG_RATIO,
        aug_prob=AUG_PROB,
        drop_edges_only=False,
        perturb_edge_features=True)
    augmented_graph, (idx_dropped_edges,
                      idx_added_edges) = graph_augmenter.perturb_edges(
                          self.graph_with_invalid_nodes)

    for molecule_idx in range(NUM_MOLECULES):
      dropped_edges = idx_dropped_edges[molecule_idx].numpy()
      added_edges = idx_added_edges[molecule_idx].numpy()
      pair_mask = augmented_graph['pair_mask'][molecule_idx].numpy()

      # Check that dropped edges were affecting valid nodes and their edges.
      for edge in dropped_edges:
        self.assertIn(edge[0], VALID_NODES)
        self.assertIn(edge[1], VALID_NODES)
        dropped_edge_was_readded = np.isin(edge, added_edges)
        if not dropped_edge_was_readded[0] and not dropped_edge_was_readded[1]:
          self.assertEqual(0., pair_mask[edge[0], edge[1]])

      for edge in added_edges:
        self.assertIn(edge[0], VALID_NODES)
        self.assertIn(edge[1], VALID_NODES)
        self.assertEqual(1., pair_mask[edge[0], edge[1]])

  def test_perturb_edges_and_drop_edges_only(self):
    graph_augmenter = augmentation_utils.GraphAugment(
        AUGMENTATIONS,
        aug_ratio=AUG_RATIO,
        aug_prob=AUG_PROB,
        drop_edges_only=True,
        perturb_edge_features=True)
    # Perturb 50% of edges == 4 nodes will be randomly dropped, and
    # 0 will be added.
    augmented_graph, (idx_dropped_edges,
                      idx_added_edges) = graph_augmenter.perturb_edges(
                          self.valid_graph)
    self.assertEmpty(idx_added_edges)

    # Check that edges not dropped have stayed the same.
    for molecule_idx in range(NUM_MOLECULES):
      orig_pair_mask = self.valid_graph['pair_mask'][molecule_idx].numpy()
      orig_pairs = self.valid_graph['pairs'][molecule_idx].numpy()
      dropped_edges = idx_dropped_edges[molecule_idx].numpy()
      pairs = augmented_graph['pairs'][molecule_idx].numpy()
      pair_mask = augmented_graph['pair_mask'][molecule_idx].numpy()
      edges_of_valid_nodes = np.squeeze(np.dstack(np.where(pair_mask == 1)))
      for edge in edges_of_valid_nodes:
        if edge not in dropped_edges:
          np.testing.assert_array_equal(pair_mask[edge[0], edge[1]],
                                        orig_pair_mask[edge[0], edge[1]])
          np.testing.assert_array_equal(pairs[edge[0], edge[1]],
                                        orig_pairs[edge[0], edge[1]])

  def test_perturb_edges_and_initialize_edge_features_randomly(self):
    graph_augmenter = augmentation_utils.GraphAugment(
        augmentations_to_use=['perturb_edges'],
        aug_ratio=AUG_RATIO,
        aug_prob=AUG_PROB,
        drop_edges_only=False,
        perturb_edge_features=True,
        initialize_edge_features_randomly=True)

    # Perturb 50% of edges == 4 nodes will be randomly dropped, and
    # 4 will be added with randomly initialized features.
    augmented_graph, (idx_dropped_edges,
                      idx_added_edges) = graph_augmenter.perturb_edges(
                          self.valid_graph)
    for molecule_idx in range(NUM_MOLECULES):
      dropped_edges = idx_dropped_edges[molecule_idx].numpy()
      added_edges = idx_added_edges[molecule_idx].numpy()

      # Check that dropped edges have been removed.
      for edge in dropped_edges:
        dropped_edge_was_readded = np.isin(edge, added_edges)
        if not dropped_edge_was_readded[0] and not dropped_edge_was_readded[1]:
          self._check_no_nonzero_features(
              augmented_graph['pairs'][molecule_idx], edge)
          self._check_no_nonzero_features(
              augmented_graph['pair_mask'][molecule_idx], edge)

      # Check that added edges exist, i.e. have non-zero features, and that
      # their features are different from the original ones.
      for edge in added_edges:
        self._check_nonzero_features_exist_for_edge(
            augmented_graph['pairs'][molecule_idx], edge)
        self.assertNotEqual(
            augmented_graph['pairs'][molecule_idx].numpy().tolist(),
            self.valid_graph['pairs'][molecule_idx].numpy().tolist())
        self._check_nonzero_features_exist_for_edge(
            augmented_graph['pair_mask'][molecule_idx], edge)

  def test_permute_edges(self):
    augmented_graph, edge_permutations = self.graph_augmenter_perturb_features.permute_edges(
        self.valid_graph)

    for molecule_idx in range(NUM_MOLECULES):
      edge_permutation = edge_permutations[molecule_idx].numpy()
      aug_pair_mask = augmented_graph['pair_mask'][molecule_idx].numpy()
      orig_pair_mask = self.valid_graph['pair_mask'][molecule_idx].numpy()
      aug_pairs = augmented_graph['pairs'][molecule_idx].numpy()
      orig_pairs = self.valid_graph['pairs'][molecule_idx].numpy()

      for row_idx, row in enumerate(aug_pair_mask):
        if row_idx < edge_permutation.shape[0]:
          np.testing.assert_array_equal(
              row, orig_pair_mask[edge_permutation[row_idx]])
          np.testing.assert_array_equal(
              aug_pairs[row_idx], orig_pairs[edge_permutation[row_idx]])
        else:
          nonzero_elems_mask = np.nonzero(row)[0]
          self.assertEmpty(nonzero_elems_mask)
          nonzero_elems = np.nonzero(aug_pairs[row_idx])[0]
          self.assertEmpty(nonzero_elems)

  def test_subgraph(self):
    (augmented_graph,
     idx_kept_nodes) = self.graph_augmenter_perturb_features.subgraph(
         self.valid_graph)

    for molecule_idx in range(NUM_MOLECULES):
      idx_nodes_in_subgraph = idx_kept_nodes[molecule_idx].numpy()
      self.assertLen(idx_nodes_in_subgraph,
                     int(drug_cardiotoxicity._MAX_NODES * 0.5 * AUG_RATIO))
      aug_atom_mask = augmented_graph['atom_mask'][molecule_idx].numpy()
      aug_atoms = augmented_graph['atoms'][molecule_idx].numpy()
      aug_pair_mask = augmented_graph['pair_mask'][molecule_idx].numpy()
      aug_pairs = augmented_graph['pairs'][molecule_idx].numpy()
      for node_idx, node in enumerate(aug_atom_mask):
        if node == 1.:
          self.assertIn(node_idx, idx_nodes_in_subgraph)
        else:
          self.assertNotIn(node_idx, idx_nodes_in_subgraph)
          self._check_no_nonzero_features(aug_atoms, [node_idx], axis=0)
          self._check_no_nonzero_features(aug_pair_mask, [node_idx], axis=0)
          self._check_no_nonzero_features(aug_pair_mask, [node_idx], axis=1)
          self._check_no_nonzero_features(aug_pairs, [node_idx], axis=0)
          self._check_no_nonzero_features(aug_pairs, [node_idx], axis=1)

  def test_subgraph_single_node_graph(self):
    # Only one valid node.
    single_node_graph = self.valid_graph.copy()
    single_node_atom_mask = np.zeros(
        drug_cardiotoxicity._MAX_NODES, dtype=np.float32)
    single_node_atom_mask[0] = 1.
    single_node_pair_mask = np.zeros(
        (drug_cardiotoxicity._MAX_NODES, drug_cardiotoxicity._MAX_NODES),
        dtype=np.float32)
    single_node_graph.update({
        'atom_mask': tf.constant([single_node_atom_mask] * NUM_MOLECULES),
        'pair_mask': tf.constant([single_node_pair_mask] * NUM_MOLECULES),
    })
    _, idx_kept_nodes = self.graph_augmenter_perturb_features.subgraph(
        single_node_graph)
    for molecule_idx in range(NUM_MOLECULES):
      idx_nodes_in_subgraph = idx_kept_nodes[molecule_idx].numpy()
      np.testing.assert_array_equal(idx_nodes_in_subgraph,
                                    np.asarray([0]))

  def test_subgraph_cycle(self):
    # Graph with cycle.
    cycle_graph = self.valid_graph.copy()
    cycle_atom_mask = np.zeros(
        drug_cardiotoxicity._MAX_NODES, dtype=np.float32)
    cycle_atom_mask[0:3] = 1.
    cycle_pair_mask = np.zeros(
        (drug_cardiotoxicity._MAX_NODES, drug_cardiotoxicity._MAX_NODES),
        dtype=np.float32)
    # Create edges 0 --> 1 --> 2 --> 0.
    source_nodes = [0, 1, 2]
    target_nodes = [1, 2, 0]
    cycle_pair_mask[source_nodes, target_nodes] = 1.
    cycle_graph.update({
        'atom_mask': tf.constant([cycle_atom_mask] * NUM_MOLECULES),
        'pair_mask': tf.constant([cycle_pair_mask] * NUM_MOLECULES),
    })

    self.graph_augmenter = augmentation_utils.GraphAugment(
        AUGMENTATIONS,
        aug_ratio=1.,
        aug_prob=1.,
        perturb_node_features=True,
        perturb_edge_features=True)
    _, idx_kept_nodes = self.graph_augmenter.subgraph(
        cycle_graph)
    for molecule_idx in range(NUM_MOLECULES):
      idx_nodes_in_subgraph = idx_kept_nodes[molecule_idx].numpy()
      np.testing.assert_array_equal(np.sort(idx_nodes_in_subgraph),
                                    np.asarray([0, 1, 2]))

  def test_empty_graph(self):
    _, idx_dropped_nodes = self.graph_augmenter_perturb_features.drop_nodes(
        self.empty_graph)
    # Check that returned arrays of dropped node indices are empty.
    self.assertEmpty(idx_dropped_nodes)

    _, (idx_dropped_edges,
        idx_added_edges) = self.graph_augmenter_perturb_features.perturb_edges(
            self.empty_graph)
    # Check that returned arrays of perturbed edge indices are empty.
    self.assertEmpty(idx_dropped_edges)
    self.assertEmpty(idx_added_edges)

    _, edge_permutations = self.graph_augmenter_perturb_features.permute_edges(
        self.empty_graph)
    # Check that returned arrays of dropped node indices are empty.
    self.assertEmpty(edge_permutations)

    _, idx_masked_nodes = self.graph_augmenter_perturb_features.mask_node_features(
        self.empty_graph)
    # Check that returned arrays of dropped node indices are empty.
    self.assertEmpty(idx_masked_nodes)

    _, idx_kept_nodes = self.graph_augmenter_perturb_features.subgraph(
        self.empty_graph)
    # Check that returned arrays of dropped node indices are empty.
    self.assertEmpty(idx_kept_nodes)

if __name__ == '__main__':
  googletest.main()

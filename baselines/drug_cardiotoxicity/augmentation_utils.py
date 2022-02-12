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

"""Library of augmentation functions to transform graphs during training.

We support the following types of augmentations:
- drop_nodes: Drops a random proportion of nodes and their edges.
"""

import random
from typing import Dict, List, Tuple

import tensorflow as tf


class GraphAugment:
  """Graph augmentation module that defines possible transformations.

  #### References
  [1]: Yuning You, et. al. Graph Contrastive Learning with Augmentations.
       In _Neural Information Processing Systems_, 2020.
       https://arxiv.org/pdf/2010.13902.pdf

  Attributes:
    augmentations_available: Dict mapping str name of augmentation function to
    function.
    augmentations_to_use: List of str representing names of augmentation
      functions to use selected by user.
    aug_ratio: Float of proportion of nodes or edges to augment.
    aug_prob: Float of probability of applying augmentation to a given graph.
    perturb_node_features: Boolean indicating to zero out features of dropped
      nodes when True. Does nothing to `atoms` when False.
    drop_edges_only: Boolean indicating to drop edges only during the edge
      perturbation augmentation when True. Re-adds the edges when False.
    perturb_edge_features: Boolean indicating to affect `pairs` by zeroing out
      dropped edge featuers and setting features of newly added features when
      True. Does nothing to `pairs` when False.
    initialize_edge_features_randomly: Boolean indicating to re-initialize the
      edge features based on a uniform normal distribution during the edge
      perturbation augmentation when True. Uses original dropped edge features
      when False. Ignored when perturb_edge_features=False or
      drop_edges_only=True.
  """

  # TODO(jihyeonlee): Allow user to specify different aug_ratios for different
  # types of augmentation.
  def __init__(self,
               augmentations_to_use: List[str],
               aug_ratio: float = 0.2,
               aug_prob: float = 0.2,
               perturb_node_features: bool = False,
               drop_edges_only: bool = False,
               perturb_edge_features: bool = False,
               initialize_edge_features_randomly: bool = False):
    self.augmentations_available = {
        'drop_nodes': self.drop_nodes,
        'perturb_edges': self.perturb_edges
    }
    for augmentation in augmentations_to_use:
      if augmentation not in self.augmentations_available:
        raise ValueError(f'Not a valid augmentation: {augmentation}')
    if 'drop_nodes' in augmentations_to_use:
      self.perturb_node_features = perturb_node_features
    if 'perturb_edges' in augmentations_to_use:
      self.drop_edges_only = drop_edges_only
      self.perturb_edge_features = perturb_edge_features
      self.initialize_edge_features_randomly = initialize_edge_features_randomly
    self.augmentations_to_use = augmentations_to_use
    self.aug_ratio = aug_ratio
    self.aug_prob = aug_prob

  def augment(self, input_graph: Dict[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
    """Randomly selects and performs an augmentation from all possible.

    Args:
      input_graph: Graph features to be augmented.

    Returns:
      Augmented graph.
    """
    # TODO(jihyeonlee): Allow user to specify number of augmentations to perform
    # per graph. Consider allowing different aug_ratio by function.
    if self.aug_ratio == 0. or not self.augmentations_to_use or random.random(
    ) < self.aug_prob or self.aug_prob == 0.:
      return input_graph
    aug_function_name = random.choice(self.augmentations_to_use)
    augmented_graph, _ = self.augmentations_available[aug_function_name](
        input_graph)
    return augmented_graph

  def drop_nodes(
      self, input_graph: Dict[str, tf.Tensor]
  ) -> Tuple[Dict[str, tf.Tensor], List[tf.Tensor]]:
    """Randomly selects nodes to drop and removes them and their edges.

    Node dropping. Given the graph G, node dropping will randomly discard
    certain portion of vertices along with their connections. The underlying
    prior enforced by it is that missing part of vertices does not affect the
    semantic meaning of G. Each node’s dropping probability follows a default
    i.i.d. uniform distribution (or any other distribution).

    Args:
      input_graph: Graph to be augmented.

    Returns:
      Augmented graph with dropped nodes.
    """
    idx_dropped_nodes = []

    def _drop_nodes_helper(
        features: Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
      """Performs batch-wise operation to drop nodes and corresponding edges."""
      atoms, atom_mask, pairs, pair_mask = features

      # Select nodes to drop.
      nodes = tf.where(tf.equal(atom_mask, 1.))
      total_num_nodes = tf.shape(nodes)[0]
      drop_num = tf.math.ceil(
          tf.math.multiply(self.aug_ratio,
                           tf.cast(total_num_nodes, dtype=tf.float32)))
      if tf.math.equal(drop_num, 0):
        return features
      # idx_drop shape: [[idx1], [idx2], ...].
      idx_drop = tf.gather(
          tf.random.shuffle(nodes), tf.range(drop_num, dtype=tf.int64))
      idx_dropped_nodes.append(tf.squeeze(idx_drop))

      # Drop selected nodes.
      aug_atom_mask = tf.tensor_scatter_nd_update(
          atom_mask, idx_drop, tf.zeros(tf.shape(idx_drop)[0]))
      if self.perturb_node_features:
        aug_atoms = tf.tensor_scatter_nd_update(
            atoms, idx_drop, tf.zeros(
                (tf.shape(idx_drop)[0], tf.shape(atoms)[1])))
      else:
        aug_atoms = atoms

      # Remove edges attached to dropped nodes.
      # First, remove edges where source node has been dropped (i.e. set rows
      # of dropped node indices to 0).
      if self.perturb_edge_features:
        aug_pairs = tf.tensor_scatter_nd_update(
            pairs, idx_drop,
            tf.zeros((tf.shape(idx_drop)[0], tf.shape(pairs)[1],
                      tf.shape(pairs)[-1])))
      else:
        aug_pairs = pairs
      aug_pair_mask = tf.tensor_scatter_nd_update(
          pair_mask, idx_drop,
          tf.zeros((tf.shape(idx_drop)[0], tf.shape(pair_mask)[-1])))
      # Second, remove edges where target node has been dropped (i.e. set
      # columns of dropped node indices to 0).
      columns = idx_drop
      rows = tf.range(tf.shape(pair_mask)[0], dtype=tf.int64)
      ii, jj = tf.meshgrid(rows, columns, indexing='ij')
      idx_to_update = tf.stack([ii, jj], axis=-1)
      updated_values_pair_mask = tf.broadcast_to(0., tf.shape(ii))
      aug_pair_mask = tf.tensor_scatter_nd_update(aug_pair_mask, idx_to_update,
                                                  updated_values_pair_mask)
      if self.perturb_edge_features:
        updated_values_pairs = tf.zeros(
            (tf.shape(ii)[0], tf.shape(ii)[1], pairs.shape[-1]))
        aug_pairs = tf.tensor_scatter_nd_update(aug_pairs, idx_to_update,
                                                updated_values_pairs)

      return aug_atoms, aug_atom_mask, aug_pairs, aug_pair_mask

    # Replace features with augmented versions.
    aug_atoms, aug_atom_mask, aug_pairs, aug_pair_mask = tf.map_fn(
        _drop_nodes_helper,
        (input_graph['atoms'], input_graph['atom_mask'], input_graph['pairs'],
         input_graph['pair_mask']))
    augmented_graph = {
        'atoms': aug_atoms,
        'atom_mask': aug_atom_mask,
        'pairs': aug_pairs,
        'pair_mask': aug_pair_mask,
        'molecule_id': input_graph['molecule_id']
    }
    return augmented_graph, idx_dropped_nodes

  # TODO(jihyeonlee): Add permute_edges function that performs simpler version
  # of simply permuting the edge features, consistent with paper in
  # GraphAugment references..
  def perturb_edges(
      self, input_graph: Dict[str, tf.Tensor]
  ) -> Tuple[Dict[str, tf.Tensor], Tuple[List[tf.Tensor], List[tf.Tensor]]]:
    """Randomly drops edges and adds edges with randomly initialized features.

    Edge perturbation. It will perturb the connectivities in G through randomly
    adding or dropping certain ratio of edges. It implies that the semantic
    meaning of G has certain robustness to the edge connectivity pattern
    variances. We follow an i.i.d. uniform distribution to add/drop each edge.
    Note that the added edges are bidirectional.

    Args:
      input_graph: Graph to be augmented.

    Returns:
      Augmented graph and tuple of lists containing indices of removed edges and
      indices of newly added edges.
    """
    idx_dropped_edges = []
    idx_added_edges = []

    def _perturb_edges_helper(
        features: Tuple[tf.Tensor, tf.Tensor, tf.Tensor]
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
      """Performs batch-wise operation to remove edges and add new ones."""
      atom_mask, pairs, pair_mask = features

      # Select edges to drop.
      nodes = tf.where(tf.equal(atom_mask, 1.))
      total_num_nodes = tf.cast(tf.shape(nodes)[0], dtype=tf.int64)

      edges_of_valid_nodes = tf.where(tf.equal(pair_mask, 1.))
      num_edges_of_valid_nodes = tf.shape(edges_of_valid_nodes)[0]
      num_edges_to_drop = tf.math.ceil(
          tf.math.multiply(self.aug_ratio,
                           tf.cast(num_edges_of_valid_nodes, dtype=tf.float32)))
      num_edges_to_drop = tf.cast(num_edges_to_drop, dtype=tf.int64)
      if tf.math.equal(num_edges_to_drop, 0):
        return features
      idx_edges_to_drop = tf.gather(
          tf.random.shuffle(edges_of_valid_nodes),
          tf.range(num_edges_to_drop, dtype=tf.int64))

      if not self.initialize_edge_features_randomly:
        features_of_edges_to_drop = tf.gather_nd(pairs, idx_edges_to_drop)

      idx_dropped_edges.append(idx_edges_to_drop)

      if not self.drop_edges_only:
        # Select edges to add. Determine number to add from number of existing
        # edges in original graph, not all potential edges, which is a much
        # larger total.
        # Added edges are bidirectional.
        num_edges_to_add = num_edges_to_drop
        idx_edges_to_add = tf.random.uniform([num_edges_to_add, 2],
                                             minval=0,
                                             maxval=total_num_nodes - 1,
                                             dtype=tf.int64)
        idx_edges_to_add = tf.map_fn(
            fn=lambda edge: tf.squeeze(tf.gather(nodes, edge, axis=0)),
            elems=idx_edges_to_add)

        idx_bidirectional_edges_to_add = tf.concat(
            (idx_edges_to_add, tf.reverse(idx_edges_to_add, axis=[1])), axis=0)

        idx_added_edges.append(idx_bidirectional_edges_to_add)

      # If self.drop_edges_only is False, re-add edges. Initialize the features
      # to values [0,1) sampled from a uniform distribution when
      # self.perturb_edge_features is True. We use the original dropped edge
      # features when self.perturb_edge_features is False.

      # Makes masks for perturbed edges.
      # <int>[num_edges_to_drop + num_edges_to_add * 2]
      drop_values_for_pair_mask = tf.zeros(num_edges_to_drop)
      if self.drop_edges_only:
        edges_to_perturb = idx_edges_to_drop
        perturbation_values_for_pair_mask = drop_values_for_pair_mask
      else:
        edges_to_perturb = tf.concat(
            (idx_edges_to_drop, idx_bidirectional_edges_to_add), axis=0)
        add_values_for_pair_mask = tf.ones(
            tf.shape(idx_bidirectional_edges_to_add)[0])
        perturbation_values_for_pair_mask = tf.concat(
            (drop_values_for_pair_mask, add_values_for_pair_mask), axis=0)

      # Makes feature values for perturbed edges.
      # <float>[num_edges_to_drop + num_edges_to_add * 2, _EDGE_FEATURE_LENGTH]
      drop_values_for_pairs = tf.zeros((num_edges_to_drop, tf.shape(pairs)[-1]))
      if self.drop_edges_only:
        perturbation_values_for_pairs = drop_values_for_pairs
      else:
        if self.initialize_edge_features_randomly:
          add_values_for_pairs = tf.random.uniform(
              (tf.shape(idx_bidirectional_edges_to_add)[0],
               tf.shape(pairs)[-1]))
        else:
          add_values_for_pairs = tf.concat(
              (features_of_edges_to_drop, features_of_edges_to_drop), axis=0)
        perturbation_values_for_pairs = tf.concat(
            (drop_values_for_pairs, add_values_for_pairs), axis=0)

      aug_pair_mask = tf.tensor_scatter_nd_update(
          pair_mask, edges_to_perturb, perturbation_values_for_pair_mask)
      if self.perturb_edge_features:
        aug_pairs = tf.tensor_scatter_nd_update(pairs, edges_to_perturb,
                                                perturbation_values_for_pairs)
      else:
        aug_pairs = pairs

      return atom_mask, aug_pairs, aug_pair_mask

    _, aug_pairs, aug_pair_mask = tf.map_fn(
        _perturb_edges_helper,
        (input_graph['atom_mask'], input_graph['pairs'],
         input_graph['pair_mask']))
    augmented_graph = {
        'atoms': input_graph['atoms'],
        'atom_mask': input_graph['atom_mask'],
        'pairs': aug_pairs,
        'pair_mask': aug_pair_mask,
        'molecule_id': input_graph['molecule_id']
    }
    return augmented_graph, (idx_dropped_edges, idx_added_edges)


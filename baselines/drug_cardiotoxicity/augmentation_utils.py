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

  Attributes:
    augmentations_available: Dict mapping str name of augmentation function to
    function.
    augmentations_to_use: List of str representing names of augmentation
      functions to use selected by user.
    aug_ratio: Float of proportion of nodes or edges to augment.
    aug_prob: Float of probability of applying augmentation to a given graph.
  """

  def __init__(self, augmentations_to_use: List[str], aug_ratio: float = 0.2,
               aug_prob: float = 0.2):
    self.augmentations_available = {
        'drop_nodes': self.drop_nodes
    }
    self.augmentations_to_use = augmentations_to_use
    self.aug_ratio = aug_ratio
    self.aug_prob = aug_prob

  def augment(
      self, input_graph: Dict[str, tf.Tensor]
  ) -> Tuple[Dict[str, tf.Tensor], List[tf.Tensor]]:
    """Randomly selects and performs an augmentation from all possible.

    Args:
      input_graph: Graph features to be augmented.

    Returns:
      Augmented graph and indices of affected nodes/edges.
    """
    # TODO(jihyeonlee): Allow user to specify number of augmentations to perform
    # per graph. Consider allowing different aug_ratio by function.
    if self.aug_ratio == 0. or not self.augmentations_to_use or random.random(
    ) < self.aug_prob:
      return input_graph, []
    aug_function_name = random.choice(self.augmentations_to_use)
    return self.augmentations_available[aug_function_name](input_graph)

  def drop_nodes(
      self, input_graph: Dict[str, tf.Tensor]
  ) -> Tuple[Dict[str, tf.Tensor], List[tf.Tensor]]:
    """Randomly selects nodes to drop and removes them and their edges.

    Node dropping. Given the graph G, node dropping will randomly discard
    certain portion of vertices along with their connections. The underlying
    prior enforced by it is that missing part of vertices does not affect the
    semantic meaning of G. Each node’s dropping probability follows a default
    i.i.d. uniform distribution (or any other distribution).

    #### References
    [1]: Yuning You, et. al. Graph Contrastive Learning with Augmentations.
         In _Neural Information Processing Systems_, 2020.
         https://arxiv.org/pdf/2010.13902.pdf

    Args:
      input_graph: Graph to be augmented.

    Returns:
      Augmented graph with dropped nodes.
    """
    idx_dropped_nodes = []

    def drop_nodes_helper(
        features: Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
      """Performs batch-wise operation to drop nodes and corresponding edges."""
      atoms, atom_mask, pairs, pair_mask, molecule_id = features

      # Select nodes to drop.
      nodes = tf.where(tf.equal(atom_mask, 1.))
      total_num_nodes = tf.shape(nodes)[0]
      drop_num = tf.math.ceil(
          tf.math.multiply(self.aug_ratio,
                           tf.cast(total_num_nodes, dtype=tf.float32)))
      # idx_drop shape: [[idx1], [idx2], ...].
      idx_drop = tf.gather(
          tf.random.shuffle(nodes), tf.range(drop_num, dtype=tf.int64))
      idx_dropped_nodes.append(tf.squeeze(idx_drop))

      # Drop selected nodes.
      aug_atoms = tf.tensor_scatter_nd_update(
          atoms, idx_drop, tf.zeros(
              (tf.shape(idx_drop)[0], tf.shape(atoms)[1])))
      aug_atom_mask = tf.tensor_scatter_nd_update(
          atom_mask, idx_drop, tf.zeros(tf.shape(idx_drop)[0]))

      # Remove edges attached to dropped nodes.
      # First, remove edges where source node has been dropped (i.e. set rows
      # of dropped node indices to 0).
      aug_pairs = tf.tensor_scatter_nd_update(
          pairs, idx_drop,
          tf.zeros(
              (tf.shape(idx_drop)[0], tf.shape(pairs)[1], tf.shape(pairs)[-1])))
      aug_pair_mask = tf.tensor_scatter_nd_update(
          pair_mask, idx_drop,
          tf.zeros((tf.shape(idx_drop)[0], tf.shape(pair_mask)[-1])))
      # Second, remove edges where target node has been dropped (i.e. set
      # columns of dropped node indices to 0).
      columns = idx_drop
      rows = tf.cast(tf.range(tf.shape(pair_mask)[0]), tf.int64)
      ii, jj = tf.meshgrid(rows, columns, indexing='ij')
      idx_to_update = tf.stack([ii, jj], axis=-1)
      updated_values_pairs = tf.zeros(
          (tf.shape(ii)[0], tf.shape(ii)[1], pairs.shape[-1]))
      updated_values_pair_mask = tf.broadcast_to(0., tf.shape(ii))
      aug_pairs = tf.tensor_scatter_nd_update(aug_pairs, idx_to_update,
                                              updated_values_pairs)
      aug_pair_mask = tf.tensor_scatter_nd_update(aug_pair_mask, idx_to_update,
                                                  updated_values_pair_mask)

      return aug_atoms, aug_atom_mask, aug_pairs, aug_pair_mask, molecule_id

    # Replace features with augmented versions.
    aug_features = tf.map_fn(
        drop_nodes_helper,
        (input_graph['atoms'], input_graph['atom_mask'], input_graph['pairs'],
         input_graph['pair_mask'], input_graph['molecule_id']))
    augmented_graph = {
        'atoms': aug_features[0],
        'atom_mask': aug_features[1],
        'pairs': aug_features[2],
        'pair_mask': aug_features[3],
        'molecule_id': aug_features[4]
    }
    return augmented_graph, idx_dropped_nodes


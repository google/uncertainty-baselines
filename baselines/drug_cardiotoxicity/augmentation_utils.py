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

"""Library of augmentation functions to transform graphs during training.

We support the following types of augmentations:
- drop_nodes: Drops a random proportion of nodes and their edges.
- perturb_edges: Randomly drops and adds edges.
"""

from typing import Dict, List, Tuple

import tensorflow as tf


def drop_nodes(
    input_graph: Dict[str, tf.Tensor],
    aug_ratio: float = 0.2) -> Tuple[Dict[str, tf.Tensor], List[tf.Tensor]]:
  """Randomly selects nodes to drop and removes them and their edges.

  Node dropping. Given the graph G, node dropping will randomly discard certain
  portion of vertices along with their connections. The underlying prior
  enforced by it is that missing part of vertices does not affect the semantic
  meaning of G. Each node’s probability of being dropped follows a default
  i.i.d. uniform distribution (or any other distribution).

  #### References
  [1]: Yuning You, et. al. Graph Contrastive Learning with Augmentations.
       In _Neural Information Processing Systems_, 2020.
       https://arxiv.org/pdf/2010.13902.pdf

  Args:
    input_graph: Graph to be augmented.
    aug_ratio: Ratio of nodes to be dropped.

  Returns:
    Augmented graph with dropped nodes.
  """
  num_molecules = input_graph['atoms'].shape[0]
  if aug_ratio == 0. or num_molecules == 0.:
    return input_graph, []

  # Will stack each of these at the end to serve as features of augmented graph.
  augmented_atoms = []
  augmented_atom_mask = []
  augmented_pairs = []
  augmented_pair_mask = []
  idx_dropped_nodes = []

  for molecule_idx in range(num_molecules):
    atoms = input_graph['atoms'][molecule_idx]
    atom_mask = input_graph['atom_mask'][molecule_idx]

    # Select nodes to drop.
    nodes = tf.where(atom_mask == 1.)
    total_num_nodes = nodes.shape[0]
    drop_num = tf.math.ceil(aug_ratio * total_num_nodes)
    # idx_drop shape: [[idx1], [idx2], ...].
    idx_drop = tf.gather(
        tf.random.shuffle(nodes), tf.range(drop_num, dtype=tf.int64))
    idx_dropped_nodes.append(tf.squeeze(idx_drop))

    # Drop selected nodes.
    augmented_atoms.append(
        tf.tensor_scatter_nd_update(
            atoms, idx_drop, tf.zeros(
                (idx_drop.shape[0], atoms.shape[1]))))
    augmented_atom_mask.append(
        tf.tensor_scatter_nd_update(atom_mask, idx_drop,
                                    tf.zeros(idx_drop.shape[0])))

    # Remove edges attached to dropped nodes.
    pairs = input_graph['pairs'][molecule_idx]
    pair_mask = input_graph['pair_mask'][molecule_idx]

    # Identify where dropped nodes are source or target nodes in edges.
    all_edges_of_dropped_nodes = tf.where(
        tf.gather(pair_mask, tf.squeeze(idx_drop)) == 1.)
    # Convert to global indices.
    edge_drop_idx = []
    for x, y in all_edges_of_dropped_nodes:
      edge_drop_idx.append(tf.concat([tf.squeeze(idx_drop[x]), y], axis=0))
      edge_drop_idx.append(tf.concat([y, tf.squeeze(idx_drop[x])], axis=0))
    edge_drop_idx = tf.stack(edge_drop_idx)

    augmented_pairs.append(
        tf.tensor_scatter_nd_update(
            pairs, edge_drop_idx,
            tf.zeros((edge_drop_idx.shape[0], pairs.shape[-1]))))
    augmented_pair_mask.append(
        tf.tensor_scatter_nd_update(
            pair_mask, edge_drop_idx,
            tf.zeros(edge_drop_idx.shape[0])))

  # Replace features with augmented versions.
  augmented_graph = {
      'atoms': tf.stack(augmented_atoms, axis=0),
      'atom_mask': tf.stack(augmented_atom_mask, axis=0),
      'pairs': tf.stack(augmented_pairs, axis=0),
      'pair_mask': tf.stack(augmented_pair_mask, axis=0),
      'molecule_id': input_graph['molecule_id'],
  }
  return augmented_graph, idx_dropped_nodes

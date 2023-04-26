# coding=utf-8
# Copyright 2023 The Uncertainty Baselines Authors.
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

"""Library for Graph Attention model."""
from typing import Optional

import tensorflow as tf
from uncertainty_baselines.models.mpnn import get_adjacency_matrix


class GraphAttentionLayer(tf.keras.layers.Layer):
  """A layer that implements graph attention."""

  def __init__(self,
               node_feature_dim,
               out_node_feature_dim,
               constant_attention=False):
    """Construct a graph attention layer.

    Args:
     node_feature_dim: dimension (integer) of incoming node level features.
       An incoming tensor should have dimension of (batch_size,
       num_nodes, node_feature_dim).
     out_node_feature_dim: dimension (integer) of outcoming node level features.
       An outcoming tensor should have dimension of (batch_size,
       num_nodes, out_node_feature_dim).
     constant_attention: a boolean. If True, we directly use equal attention
       coefficients across neighbors without going through the network. Default
       is False.
    """
    super().__init__()
    self.constant_attention = constant_attention
    self.w = self.add_weight(
        name="w",
        shape=(node_feature_dim, out_node_feature_dim),
        initializer="glorot_uniform",
        regularizer="l2",
        trainable=True)
    self.a = self.add_weight(
        name="a",
        shape=(2 * out_node_feature_dim, 1),
        initializer="glorot_uniform",
        regularizer="l2",
        trainable=True)

    self.leakyrelu = tf.keras.layers.LeakyReLU(alpha=0.2)

  def call(self, h, adj):
    """Forward pass computation of the layer.

    Args:
      h: An incoming tensor contains node level features and it
        should have dimension of (batch_size, num_nodes, node_feature_dim).
      adj: An incoming tensor contains adjacency matrices. Each adjacency
        matrix has added diagonal ones before entering the layer.
        It should have dimension of (batch_size, num_nodes, num_nodes).

    Returns:
      new_h: The new node level features tensor. It is aggregated
        from neighbours with attention and it has dimension of
        (batch_size, out_node_feature_dim).
    """
    wh = tf.matmul(h, self.w)  # (batch_size, num_nodes, out_node_feature_dim)

    # Go through attention.
    if self.constant_attention:
      attention_scores = tf.ones([tf.shape(h)[0], tf.shape(h)[1], tf.shape(h)[1]
                                 ])  # (batch_size, num_nodes, num_nodes)
    else:
      attention_inputs = self._prepare_attention_inputs(
          wh)  # (batch_size, num_nodes, num_nodes, 2*out_node_feature_dim)
      attention_scores = self._calc_attention_scores(
          attention_inputs)  # (batch_size, num_nodes, num_nodes)

    attention_coeffs = self._calc_attention_coeffs(attention_scores, adj)

    new_h = tf.matmul(attention_coeffs,
                      wh)  # (batch_size, num_nodes, out_node_feature_dim)
    return new_h

  def _prepare_attention_inputs(self, wh):
    """Prepare inputs for downstream attention computation.

    Since our downstream attention score calculation takes a pair of
    nodes as input, this method really is to generate a list of all possible
    pairs and arrange them into adjacency-matrix-like shape.

    Args:
      wh: Incoming transformed node representation. It should have
        dimension of (batch_size, num_nodes, out_node_feature_dim).
    Returns:
      attention_inputs: Prepared tensor to feed into attention score
        calculation. It has dimension of (batch_size, num_nodes,
        num_nodes, 2*out_node_feature_dim).
    """
    batch_size, num_nodes = tf.shape(wh)[0], tf.shape(wh)[1]

    # Get wh repeats and tiles.
    wh_repeats = tf.repeat(
        wh, repeats=tf.ones(num_nodes, dtype=tf.int32) * num_nodes,
        axis=1)  # (batch_size, num_nodes^2, out_node_feature_dim)

    wh_tile = tf.tile(
        wh,
        [1, num_nodes, 1])  # (batch_size, num_nodes^2, out_node_feature_dim)

    # Generate a list of all possible pairs of nodes.
    attention_inputs = tf.concat(
        [wh_repeats, wh_tile],
        axis=2)  # (batch_size, num_nodes^2, 2*out_node_feature_dim)

    # Reshape the list into an adjacency-matrix-like tensor.
    attention_inputs = tf.reshape(
        attention_inputs,
        [batch_size, num_nodes, num_nodes, -1
        ])  # (batch_size, num_nodes, num_nodes, 2*out_node_feature_dim)

    return attention_inputs

  def _calc_attention_scores(self, attention_inputs):
    """Compute attention scores.

    Args:
      attention_inputs: The incoming tensor contains pair-wise
        node-level representations. It has dimension of (batch_size,
        num_nodes, num_nodes, 2*out_node_feature_dim)
    Returns:
      attention_scores: Computed attention scores tensor with dimension
        of (batch_size, num_nodes, num_nodes)
    """
    attention_scores = tf.squeeze(
        tf.matmul(attention_inputs, self.a),
        axis=[3])  # (batch_size, num_nodes, num_nodes)
    attention_scores = self.leakyrelu(
        attention_scores)  # (batch_size, num_nodes, num_nodes)
    return attention_scores

  def _calc_attention_coeffs(self, attention_scores, adj):
    """Compute attention coefficients.

    The attention coefficients, at high level, are computed based on attention
    scores and adjacency matrix. In this current implementation, we only allow
    direct neighbours to have positive attention coefficients; all the
    non-direct neighbours may have high attention scores, but we still assign
    zero coefficients to them. This can be dis-regulated in the future
    because long-distance intra-molecular interactions are common.

    Args:
      attention_scores: An incoming tensor contains pair-wise attention
        scores (logits). It has dimension of (batch_size, num_nodes, num_nodes).
      adj: An incoming tensor contains adjacency matrices. Each adjacency
        matrix has added diagonal ones before entering the layer.
        It should have dimension of (batch_size, num_nodes, num_nodes). It's
        used to decide how large of a neighbourhood to consider for
        attention contributions.
    Returns:
      attention_coeffs: Computed attention coefficients tensor with dimension
        of (batch_size, num_nodes, num_nodes)
    """
    # Create a tensor of (batch_size, num_nodes, num_nodes)
    # with default scores of negative infinities (-9e15). The reason is
    # some entries will be updated with computed attention scores
    # (should be much larger than negative infinities), and after going
    # through softmax, entries with default scores will give zero
    # attention coefficients.
    default_scores = -9e15 * tf.ones_like(
        attention_scores)  # (batch_size, num_nodes, num_nodes)

    # Final scores will be same as default scores, except that those
    # entries for direct-neighbours (adj > 0) will be updated with
    # computed attention scores.
    whether_neighboring = tf.math.greater(adj, 0)
    final_scores = tf.where(
        whether_neighboring, attention_scores,
        default_scores)  # (batch_size, num_nodes, num_nodes)
    # After softmax, those entries with default scores give zero
    # attention coefficients.
    attention_coeffs = tf.nn.softmax(
        final_scores, axis=2)  # (batch_size, num_nodes, num_nodes)

    return attention_coeffs


class GATModel(tf.keras.Model):
  """A model that implements molecule classification via graph attention.

  #### References
  [1]: Petar Veličković, et. al. Graph Attention Networks. ICLR 2018.
       https://arxiv.org/abs/1710.10903

  """

  def __init__(
      self,
      attention_heads,
      node_feature_dim,
      out_node_feature_dim,
      readout_layer_size,
      num_classes,
      constant_attention=False,
      kernel_regularizer: Optional[tf.keras.regularizers.Regularizer] = None,
      dropout_rate=0.1):
    """Construct a graph attention model.

    Args:
     attention_heads: number (integer) of attention heads.
     node_feature_dim: dimension (integer) of incoming node level features.
     out_node_feature_dim: dimension (integer) of node level features outcoming
       from the attention layer.
     readout_layer_size: dimension (integer) of graph level features after
       readout layer.
     num_classes: number (integer) of classes for classification.
     constant_attention: a boolean. If True, we directly use equal attention
       coefficients across neighbors without going through the network. Default
       is False.
     kernel_regularizer: Regularization function for Dense layer.
     dropout_rate: a float regulating percent of features to turn OFF.
    """
    super().__init__()
    self.dropout = tf.keras.layers.Dropout(dropout_rate)

    # We have three consecutive graph attention layers
    # before readout the graph level representations.
    self.attention_heads1 = [
        GraphAttentionLayer(node_feature_dim, out_node_feature_dim,
                            constant_attention) for _ in range(attention_heads)
    ]
    self.attention_heads2 = [
        GraphAttentionLayer(out_node_feature_dim * attention_heads,
                            out_node_feature_dim, constant_attention)
        for _ in range(attention_heads)
    ]
    self.attention_heads3 = [
        GraphAttentionLayer(out_node_feature_dim * attention_heads,
                            out_node_feature_dim, constant_attention)
        for _ in range(attention_heads)
    ]

    self.i_layer = tf.keras.layers.Dense(
        readout_layer_size,
        activation="sigmoid",
        kernel_regularizer=kernel_regularizer)

    self.j_layer = tf.keras.layers.Dense(
        readout_layer_size, kernel_regularizer=kernel_regularizer)

    self.classifier = tf.keras.layers.Dense(
        num_classes, kernel_regularizer=kernel_regularizer)

    self.softmax = tf.keras.layers.Softmax()

  def graph_representation(self, nodes, adj, training=False):
    """Forward pass to compute molecular graph level representation.

    Args:
      nodes: An incoming tensor contains node level features and it
        should have dimension of (batch_size, num_nodes, node_feature_dim).
      adj: An incoming tensor contains adjacency matrices. Each adjacency
        matrix has added diagonal ones before entering the layer.
        It should have dimension of (batch_size, num_nodes, num_nodes).
      training: A boolean indicating if the model is in training mode or not.
        This affects the behavior of dropout layers.

    Returns:
      x_g: The graph level features tensor. It is aggregated
        from neighbours with attention and it has dimension of
        (batch_size, out_node_feature_dim).
    """
    # Go through graph attention 1 & 2.
    attention_layers = [self.attention_heads1, self.attention_heads2]
    nodes_under_iter = nodes
    for attention_heads in attention_layers:
      nodes_under_iter = self.dropout(nodes_under_iter, training=training)
      nodes_under_iter = tf.concat(
          [a_head(nodes_under_iter, adj) for a_head in attention_heads],
          axis=2)  # (batch_size, num_nodes, heads * out_node_feature_dim)
      nodes_under_iter = tf.nn.elu(
          nodes_under_iter
      )  # (batch_size, num_nodes, heads * out_node_feature_dim)

    # Go through graph attention 3.
    nodes_under_iter = self.dropout(nodes_under_iter, training=training)

    if len(self.attention_heads3) > 1:
      nodes_under_iter = tf.keras.layers.Average()([
          a_head(nodes_under_iter, adj) for a_head in self.attention_heads3
      ])  # (batch_size, num_nodes, out_node_feature_dim)
    else:
      nodes_under_iter = self.attention_heads3[0](nodes_under_iter, adj)

    # Go though graph level aggregation.
    readout = tf.reduce_sum(
        tf.multiply(
            self.i_layer(
                tf.keras.layers.Concatenate()([nodes_under_iter, nodes])),
            self.j_layer(nodes_under_iter)),
        axis=1)  # (batch_size, graph_level_features)

    return readout

  def call(self, inputs, training=False):  # pytype: disable=signature-mismatch  # overriding-parameter-count-checks
    """Forward pass computation of the model.

    Args:
      inputs: An inputs dictionary fed to the model.
      training: A boolean indicating if the model is in training mode or not.

    Returns:
      output: Logits tensor with dimension of (batch_size, classes)
        for classification.
    """
    nodes, edges = inputs["atoms"], inputs["pairs"]
    adjacency_matrix = tf.cast(get_adjacency_matrix(edges), tf.int32)
    readout = self.graph_representation(nodes, adjacency_matrix, training)

    logits = self.classifier(
        readout, training=training)  # (batch_size, classes)
    return self.softmax(logits)


def gat(attention_heads,
        node_feature_dim,
        out_node_feature_dim,
        readout_layer_size,
        num_classes,
        constant_attention=False,
        kernel_regularizer: Optional[tf.keras.regularizers.Regularizer] = None,
        dropout_rate=0.1) -> tf.keras.Model:
  """Builds a GAT model.

  Args:
   attention_heads: number (integer) of attention heads.
   node_feature_dim: dimension (integer) of incoming node level features.
   out_node_feature_dim: dimension (integer) of node level features outcoming
     from the attention layer.
   readout_layer_size: dimension (integer) of graph level features after
     readout layer.
   num_classes: number (integer) of classes for classification.
   constant_attention: a boolean. If True, we directly use equal attention
     coefficients across neighbors without going through the network. Default
     is False.
   kernel_regularizer: Regularization function for Dense layer.
   dropout_rate: a float regulating percent of features to turn OFF.

  Returns:
    A Keras Model (not compiled).
  """
  return GATModel(
      attention_heads=attention_heads,
      node_feature_dim=node_feature_dim,
      out_node_feature_dim=out_node_feature_dim,
      readout_layer_size=readout_layer_size,
      num_classes=num_classes,
      constant_attention=constant_attention,
      kernel_regularizer=kernel_regularizer,
      dropout_rate=dropout_rate)

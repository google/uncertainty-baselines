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

"""Library for MPNN model."""
from typing import Any, Dict, Optional, Tuple

import edward2 as ed
import tensorflow as tf

from uncertainty_baselines.models import classifier_utils


class MpnnLayer(tf.keras.layers.Layer):
  """Message passing layer."""

  def __init__(
      self,
      num_node_features: int,
      message_layer_size: int,
      kernel_regularizer: Optional[tf.keras.regularizers.Regularizer] = None,
      use_spec_norm=False,
      spec_norm_multiplier=2.0):
    """Initializes the instance.

    Args:
      num_node_features: Number of node input features.
      message_layer_size: Number of hidden nodes in the message function.
      kernel_regularizer: Regularization function for inner layers of MpnnLayer.
      use_spec_norm: Whether to use Spectral normalization for the MP layer.
      spec_norm_multiplier: Multiplier used to control the magnitude of
        eigenvalue of the MP layer weight matrix.
    """
    super().__init__()
    self.num_node_features = num_node_features
    self.message_layer_size = message_layer_size
    self.use_spec_norm = use_spec_norm
    self.spec_norm_multiplier = spec_norm_multiplier
    # Follow the section of Gated Graph Neural Networks (GG-NN),
    # Li et al. (2016) to define message function: a simple
    # linear transformation of h_v, h_w and e_{vw}.
    self.message_function_dense = tf.keras.layers.Dense(
        self.message_layer_size,
        kernel_regularizer=kernel_regularizer)
    if self.use_spec_norm:
      self.message_function = ed.layers.SpectralNormalization(
          self.message_function_dense, inhere_layer_name=True,
          norm_multiplier=spec_norm_multiplier)
    else:
      self.message_function = self.message_function_dense
    self.update_function = tf.keras.layers.GRU(
        self.num_node_features, return_state=True,
        kernel_regularizer=kernel_regularizer)

  def prepare_message_input(self, nodes: tf.Tensor,
                            edges: tf.Tensor) -> tf.Tensor:
    """Prepares message input tensor for message_function.

    This is done by concatenating node-v features (node self), node-w features
    (other node) and edge-vw features (between the node-v self and the
    other node-w).
    Args:
      nodes: Float tensor with shape [batch_size, num_nodes, num_node_features].
      edges: Float tensor with shape [batch_size, num_nodes, num_nodes,
        num_edge_features].

    Returns:
      Message input tensor ready to feed into self.message_function.
    """
    batch_size, num_nodes = tf.shape(nodes)[0], tf.shape(nodes)[1]
    # Concatenate h_v, h_w and e_{vw}
    tiled_nodes = tf.reshape(
        tf.tile(nodes, [1, num_nodes, 1]),
        [batch_size, num_nodes, num_nodes, self.num_node_features])
    message_input = tf.concat(
        [tf.transpose(tiled_nodes, [0, 2, 1, 3]), tiled_nodes, edges], axis=-1)

    return message_input

  def aggregate(self, messages: tf.Tensor,
                adjacency_matrix: tf.Tensor) -> tf.Tensor:
    """Aggregates messages from node-v's neighbors.

    Args:
      messages: Float tensor with shape [batch_size, num_nodes, num_nodes,
        message_layer_size].
      adjacency_matrix: Boolean tensor with shape [batch_size, num_nodes,
        num_nodes].

    Returns:
      Message input tensor ready to feed into self.message_function. It has
        shape [batch_size, num_nodes, message_layer_size].
    """
    neighbor_messages = tf.multiply(
        messages,
        tf.expand_dims(tf.cast(adjacency_matrix, messages.dtype), axis=-1))
    return tf.reduce_sum(neighbor_messages, axis=2)

  def prepare_update_function_inputs(
      self, aggregated_messages: tf.Tensor,
      nodes: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    """Prepares inputs for update function.

    Args:
      aggregated_messages: Float tensor with shape [batch_size, num_nodes,
        message_layer_size]
      nodes: Float tensor with shape [batch_size, num_nodes, num_node_features].

    Returns:
      A tuple of reshaped messages tensor and nodes tensor.
    """
    messages_inputs = tf.reshape(aggregated_messages,
                                 [-1, 1, self.message_layer_size])
    nodes_inputs = tf.reshape(nodes, [-1, self.num_node_features])

    return messages_inputs, nodes_inputs

  def call(self, nodes: tf.Tensor, edges: tf.Tensor) -> tf.Tensor:
    """Applies the layer to the given inputs.

    Args:
      nodes: Float tensor with shape [batch_size, num_nodes, num_node_features].
      edges: Float tensor with shape [batch_size, num_nodes, num_nodes,
        num_edge_features].

    Returns:
      Updated nodes tensor.
    """
    # Generate messages from nodes and edges.
    message_input = self.prepare_message_input(nodes, edges)
    messages = self.message_function(message_input)

    # Aggregates messages from neighbors.
    adjacency_matrix = get_adjacency_matrix(edges)
    aggregated_messages = self.aggregate(messages, adjacency_matrix)

    # Update nodes features by feeding messages into original
    # nodes features.
    (update_input_messages,
     update_input_nodes) = self.prepare_update_function_inputs(
         aggregated_messages, nodes)
    _, updated_nodes = self.update_function(
        update_input_messages, initial_state=update_input_nodes)

    updated_nodes = tf.reshape(updated_nodes, tf.shape(nodes))
    if self.use_spec_norm:
      updated_nodes = updated_nodes + nodes

    return updated_nodes


def get_adjacency_matrix(pairs: tf.Tensor) -> tf.Tensor:
  """Extracts the adjacency matrix from the full pair features tensor.

  Args:
    pairs: Float edge feature tensor with shape [batch_size, max_nodes,
      max_nodes, num_pair_features].

  Returns:
    Boolean tensor with shape [batch_size, max_nodes, max_nodes] indicating
    which nodes are connected to one another.
  """
  num_edge_types = 4
  return tf.reduce_any(
      tf.cast(pairs[:, :, :, :num_edge_types], tf.bool), axis=-1)


class MpnnModel(tf.keras.Model):
  """Classifier model based on a MPNN encoder."""

  def __init__(
      self,
      nodes_shape: Tuple[int, int],
      edges_shape: Tuple[int, int, int],
      num_heads: int,
      num_layers: int,
      message_layer_size: int,
      readout_layer_size: int,
      gp_layer_kwargs: Optional[Dict[str, Any]] = None,
      use_gp_layer: bool = False,
      kernel_regularizer: Optional[tf.keras.regularizers.Regularizer] = None,
      use_spec_norm: bool = False,
      spec_norm_multiplier: float = 2.0,
      use_spec_norm_mp: bool = False,
      spec_norm_multiplier_mp: float = 2.0):
    """Constructor.

    Notes:
      * The readout is Eq. (4) from https://arxiv.org/pdf/1704.01212.pdf.

    Args:
      nodes_shape: Shape of the nodes tensor (excluding batch dimension).
      edges_shape: Shape of the edges tensor (excluding batch dimension).
      num_heads: Number of output classes.
      num_layers: Number of message passing layers.
      message_layer_size: Number of hidden units in message functions.
      readout_layer_size: Number of hidden units in the readout function.
      gp_layer_kwargs: Dict of parameters used in Gaussian Process layer.
      use_gp_layer: Bool, if set True, GP layer is used to build classifier.
      kernel_regularizer: Regularization function for Dense layer.
      use_spec_norm: Whether to use Spectral normalization for the dense layer.
      spec_norm_multiplier: Multiplier used to control the magnitude of
        eigenvalue of the dense layer weight matrix.
      use_spec_norm_mp: Whether to use Spectral normalization for the MP layer.
      spec_norm_multiplier_mp: Multiplier used to control the magnitude of
        eigenvalue of the MP layer weight matrix.

    """
    super().__init__()
    self.use_gp_layer = use_gp_layer

    self.mpnn_layers = []
    for _ in range(num_layers):
      self.mpnn_layers.append(
          MpnnLayer(
              nodes_shape[-1], message_layer_size, kernel_regularizer,
              use_spec_norm=use_spec_norm_mp,
              spec_norm_multiplier=spec_norm_multiplier_mp))

    self.i_layer = tf.keras.layers.Dense(
        readout_layer_size, activation='sigmoid',
        kernel_regularizer=kernel_regularizer)

    self.j_layer = tf.keras.layers.Dense(
        readout_layer_size, kernel_regularizer=kernel_regularizer)

    if use_spec_norm:
      self.i_layer_final = ed.layers.SpectralNormalization(
          self.i_layer, inhere_layer_name=True,
          norm_multiplier=spec_norm_multiplier)
      self.j_layer_final = ed.layers.SpectralNormalization(
          self.j_layer, inhere_layer_name=True,
          norm_multiplier=spec_norm_multiplier)
    else:
      self.i_layer_final = self.i_layer
      self.j_layer_final = self.j_layer

    self.classifier = classifier_utils.build_classifier(
        num_classes=num_heads,
        gp_layer_kwargs=gp_layer_kwargs,
        use_gp_layer=use_gp_layer,
        kernel_regularizer=kernel_regularizer)

    self.softmax = tf.keras.layers.Softmax()

  def call(self, inputs, training=False):

    nodes, edges = inputs['atoms'], inputs['pairs']
    nodes_under_iter = nodes
    for mpnn_layer in self.mpnn_layers:
      nodes_under_iter = mpnn_layer(nodes_under_iter, edges)

    readout = tf.reduce_sum(
        tf.multiply(
            self.i_layer_final(
                tf.keras.layers.Concatenate()([nodes_under_iter, nodes])),
            self.j_layer_final(nodes_under_iter)),
        axis=1)

    logits = self.classifier(readout, training=training)
    if self.use_gp_layer:
      # If model uses gp layer, the classifier returns a tuple of
      # (logits, covmat).
      logits, covmat = logits
      if not training:
        logits = ed.layers.utils.mean_field_logits(
            logits, covmat, mean_field_factor=0.1)
    return self.softmax(logits)


def mpnn(
    nodes_shape: Tuple[int, int],
    edges_shape: Tuple[int, int, int],
    num_heads: int,
    num_layers: int,
    message_layer_size: int,
    readout_layer_size: int,
    gp_layer_kwargs: Optional[Dict[str, Any]] = None,
    use_gp_layer: bool = False,
    kernel_regularizer: Optional[tf.keras.regularizers.Regularizer] = None,
    use_spec_norm: bool = False,
    spec_norm_multiplier: float = 2.0,
    use_spec_norm_mp: bool = False,
    spec_norm_multiplier_mp: float = 2.0
) -> tf.keras.Model:
  """Builds a MPNN model.

  Notes:
    * The readout is Eq. (4) from https://arxiv.org/pdf/1704.01212.pdf.

  Args:
    nodes_shape: Shape of the nodes tensor (excluding batch dimension).
    edges_shape: Shape of the edges tensor (excluding batch dimension).
    num_heads: Number of output classes.
    num_layers: Number of message passing layers.
    message_layer_size: Number of hidden units in message functions.
    readout_layer_size: Number of hidden units in the readout function.
    gp_layer_kwargs: Dict of parameters used in Gaussian Process layer.
    use_gp_layer: Bool, if set True, GP layer is used to build classifier.
    kernel_regularizer: Regularization function for Dense layer.
    use_spec_norm: Whether to use Spectral normalization for the dense layer.
    spec_norm_multiplier: Multiplier used to control the magnitude of
        eigenvalue of the dense layer weight matrix.
    use_spec_norm_mp: Whether to use Spectral normalization for the MP layer.
    spec_norm_multiplier_mp: Multiplier used to control the magnitude of
        eigenvalue of the MP layer weight matrix.

  Returns:
    A Keras Model (not compiled).
  """
  return MpnnModel(nodes_shape=nodes_shape,
                   edges_shape=edges_shape,
                   num_heads=num_heads,
                   num_layers=num_layers,
                   message_layer_size=message_layer_size,
                   readout_layer_size=readout_layer_size,
                   gp_layer_kwargs=gp_layer_kwargs,
                   use_gp_layer=use_gp_layer,
                   kernel_regularizer=kernel_regularizer,
                   use_spec_norm=use_spec_norm,
                   spec_norm_multiplier=spec_norm_multiplier,
                   use_spec_norm_mp=use_spec_norm_mp,
                   spec_norm_multiplier_mp=spec_norm_multiplier_mp)

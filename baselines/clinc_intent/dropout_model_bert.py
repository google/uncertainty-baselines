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
"""The BERT model with dropout layers."""
import functools
import math
from typing import Any, Dict, Union, Tuple

import tensorflow as tf

from official.modeling import tf_utils  # pylint: disable=unused-import
from official.nlp.modeling import layers as bert_layers
from official.nlp.modeling import networks as bert_encoder


def _monte_carlo_dropout(inputs: tf.Tensor, dropout_rate: float,
                         use_mc_dropout: bool,
                         channel_wise_dropout: bool) -> tf.Tensor:
  """Implements a Monte Carlo dropout layer callable for the Transformer model.

  Args:
    inputs: An input tensor in the BERT encoder. It can be either a 3D layer
      output with shape [batch_size, seq_len, hidden_dim], or a 4D attention
      mask with shape [batch_size, num_head, seq_len, seq_len].
    dropout_rate: Dropout rate.
    use_mc_dropout: Whether to enable Monte Carlo dropout at inference time.
    channel_wise_dropout: Whether to apply structured dropout along the
      dimension of the hidden channels or of the attention heads.

  Returns:
    (tf.Tensor) Output of the (structured) Monte Carlo dropout layer.
  """
  training = True if use_mc_dropout else None
  noise_shape = None
  input_size = len(inputs.shape)

  if input_size not in (3, 4):
    raise ValueError(f'"inputs" shape can only be 3 or 4, got {input_size}.')

  if channel_wise_dropout:
    # Produces structured dropout mask depending on input shape.
    if input_size == 3:
      # Input is a 3D layer output [batch_size, seq_len, hidden_dim]
      noise_shape = [inputs.shape[0], 1, inputs.shape[-1]]
    elif input_size == 4:
      # Input is a 4D attention mask [batch_size, num_head, seq_len, seq_len]
      noise_shape = [inputs.shape[0], inputs.shape[1], 1, 1]

  return tf.keras.layers.Dropout(dropout_rate, noise_shape=noise_shape)(
      inputs, training=training)


class DropoutMultiHeadAttention(bert_layers.MultiHeadAttention):
  """MultiHeadAttention layer with Monte Carlo dropout."""

  def __init__(self,
               use_mc_dropout: bool = False,
               channel_wise_dropout: bool = False,
               **kwargs: Dict[str, Any]):
    super().__init__(**kwargs)
    self._attention_scores_mc_dropout = functools.partial(
        _monte_carlo_dropout,
        dropout_rate=self._dropout,
        use_mc_dropout=use_mc_dropout,
        channel_wise_dropout=channel_wise_dropout)

  def compute_attention(
      self,
      query: tf.Tensor,
      key: tf.Tensor,
      value: tf.Tensor,
      attention_mask: Union[tf.Tensor,
                            None] = None) -> Tuple[tf.Tensor, tf.Tensor]:
    """Applies Dot-product attention with Monte Carlo dropout.

    This function defines the computation inside `call` with projected
    multi-head Q, K, V inputs. This function is the same as the original
    compute_attention function, except that the dropout layers were overwritten
    to enable inference-time dropouts.

    Args:
      query: Projected query `Tensor` of shape `[B, T, N, key_size]`.
      key: Projected key `Tensor` of shape `[B, T, N, key_size]`.
      value: Projected value `Tensor` of shape `[B, T, N, value_size]`.
      attention_mask: a boolean mask of shape `[B, T, S]`, that prevents
        attention to certain positions.

    Returns:
      attention_output: Multi-headed outputs of attention computation.
      attention_scores: Multi-headed attention weights.
    """
    query = tf.multiply(query, 1.0 / math.sqrt(float(self._key_size)))
    attention_scores = tf.einsum(self._dot_product_equation, key, query)
    attention_scores = self._masked_softmax(attention_scores, attention_mask)

    # Replace the default dropout layer with mc_dropout layer.
    attention_scores_dropout = self._attention_scores_mc_dropout(
        attention_scores)

    # `context_layer` = [B, T, N, H]
    attention_output = tf.einsum(self._combine_equation,
                                 attention_scores_dropout, value)
    return attention_output, attention_scores


class DropoutTransformer(bert_layers.TransformerScaffold):
  """Transformer encoder with Monte Carlo dropout."""

  def __init__(self,
               use_mc_dropout_mha: bool = False,
               use_mc_dropout_att: bool = False,
               use_mc_dropout_ffn: bool = False,
               channel_wise_dropout_mha: bool = False,
               channel_wise_dropout_att: bool = False,
               channel_wise_dropout_ffn: bool = False,
               **kwargs: Dict[str, Any]):
    """Initializer.

    Args:
      use_mc_dropout_mha: Whether to apply MC Dropout to the multi-head
        attention score layer.
      use_mc_dropout_att: Whether to apply MC Dropout to the attention
        output layer.
      use_mc_dropout_ffn: Whether to apply MC Dropout to the feedforward
        layer.
      channel_wise_dropout_mha: Whether to apply MC Dropout to the
        multi-head attention score layer.
      channel_wise_dropout_att: Whether to apply MC Dropout to the
        attention output layer.
      channel_wise_dropout_ffn: Whether to apply MC Dropout to the
        feedforward layer.
      **kwargs: Additional keyword arguments to TransformerScaffold.
    """
    attention_cls = functools.partial(
        DropoutMultiHeadAttention,
        use_mc_dropout=use_mc_dropout_mha,
        channel_wise_dropout=channel_wise_dropout_mha)

    super(DropoutTransformer, self).__init__(
        attention_cls=attention_cls, **kwargs)

    # Build custom _attention_dropout and _output_dropout layers.
    self._attention_mc_dropout = functools.partial(
        _monte_carlo_dropout,
        dropout_rate=self._dropout_rate,
        use_mc_dropout=use_mc_dropout_att,
        channel_wise_dropout=channel_wise_dropout_att)
    self._feedforward_mc_dropout = functools.partial(
        _monte_carlo_dropout,
        dropout_rate=self._dropout_rate,
        use_mc_dropout=use_mc_dropout_ffn,
        channel_wise_dropout=channel_wise_dropout_ffn)

  def call(self, inputs: tf.Tensor) -> tf.Tensor:
    if isinstance(inputs, (list, tuple)) and len(inputs) == 2:
      input_tensor, attention_mask = inputs
    else:
      input_tensor, attention_mask = (inputs, None)

    attention_output = self._attention_layer(
        query=input_tensor, value=input_tensor, attention_mask=attention_mask)
    # Replace the default dropout layer with mc_dropout layer.
    attention_output = self._attention_mc_dropout(attention_output)
    attention_output = self._attention_layer_norm(input_tensor +
                                                  attention_output)
    if self._feedforward_block is None:
      intermediate_output = self._intermediate_dense(attention_output)
      intermediate_output = self._intermediate_activation_layer(
          intermediate_output)
      layer_output = self._output_dense(intermediate_output)
      # Replace the default dropout layer with mc_dropout layer.
      layer_output = self._feedforward_mc_dropout(layer_output)
      layer_output = tf.cast(layer_output, tf.float32)
      layer_output = self._output_layer_norm(layer_output + attention_output)
    else:
      layer_output = self._feedforward_block(attention_output)

    return layer_output


class DropoutTransformerEncoder(bert_encoder.EncoderScaffold):
  """Transformer encoder network with Monte Carlo dropout."""

  def __init__(
      self,
      use_mc_dropout_mha: bool = False,
      use_mc_dropout_att: bool = False,
      use_mc_dropout_ffn: bool = False,
      channel_wise_dropout_mha: bool = False,
      channel_wise_dropout_att: bool = False,
      channel_wise_dropout_ffn: bool = False,
      # A dict of kwargs to pass to the transformer.
      hidden_cfg: Union[tf.Tensor, None] = None,
      **kwargs: Dict[str, Any]):
    hidden_cls = DropoutTransformer

    # Add MC Dropout arguments to default transformer config.
    mc_dropout_cfg = {
        'use_mc_dropout_mha': use_mc_dropout_mha,
        'use_mc_dropout_att': use_mc_dropout_att,
        'use_mc_dropout_ffn': use_mc_dropout_ffn,
        'channel_wise_dropout_mha': channel_wise_dropout_mha,
        'channel_wise_dropout_att': channel_wise_dropout_att,
        'channel_wise_dropout_ffn': channel_wise_dropout_ffn
    }

    if hidden_cfg:
      hidden_cfg.update(mc_dropout_cfg)
    else:
      hidden_cfg = mc_dropout_cfg

    super(DropoutTransformerEncoder, self).__init__(
        hidden_cls=hidden_cls, hidden_cfg=hidden_cfg, **kwargs)

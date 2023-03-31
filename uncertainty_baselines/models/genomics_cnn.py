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

"""GenomicsCNN model.

A 1D CNN model for genomic sequence classification.

## References
[1]: Ren, Jie, Peter J. Liu, Emily Fertig, Jasper Snoek, Ryan Poplin,
     Mark Depristo, Joshua Dillon, and Balaji Lakshminarayanan.
     "Likelihood ratios for out-of-distribution detection."
     In Advances in Neural Information Processing Systems, pp. 14707-14718.
     2019.
     http://papers.nips.cc/paper/9611-likelihood-ratios-for-out-of-distribution-detection
"""

import tensorflow as tf

VOCAB_SIZE = 4  # DNA sequences are composed of {A, C, G, T}


def _conv_pooled_block(inputs: tf.Tensor, num_motifs: int, len_motifs: int,
                       embed_size: int, l2_weight: float) -> tf.Tensor:
  """Creates an 1D convolutional layer with max pooling over sequence positions.

  Args:
    inputs: (tf.Tensor) Input tensor, shape (batch_size, seq_len, embed_size).
    num_motifs: (int) Number of motifs/filters to apply to input.
    len_motifs: (int) Length of the motif/convolutional filter.
    embed_size: (int) Static size of hidden dimension of the text embedding.
    l2_weight: (float) L2 regularization parameter.

  Returns:
    (tf.Tensor) shape (batch_size, 1, num_motifs).
  """
  filter_shape = (len_motifs, embed_size)

  conv_layer = tf.keras.layers.Conv2D(
      num_motifs,
      filter_shape,
      strides=(1, 1),
      padding='valid',
      data_format='channels_last',
      activation=tf.keras.activations.relu,
      kernel_regularizer=tf.keras.regularizers.l2(l2_weight),
      name='conv1d')

  # expend last dim to fit conv2d
  out = tf.expand_dims(inputs, axis=3)  # [batch_size, seq_len, embed_size, 1]
  out = conv_layer(out)  # [batch_size, seq_len-len_motifs+1, 1, num_motifs]
  out = tf.squeeze(
      out, axis=2)  # [batch_size, seq_len-len_motifs+1, num_motifs]

  # Max pooling over all positions for each motifs.
  out = tf.reduce_max(out, axis=1)  # [batch_size, num_motifs]

  return out


def genomics_cnn(
    batch_size: int,
    num_motifs: int,
    len_motifs: int,
    num_denses: int,
    num_classes: int = 10,
    embed_size: int = 4,
    dropout_rate: float = 0.1,
    l2_weight: float = 0.0,
    one_hot: bool = True,
) -> tf.keras.models.Model:
  """Builds TextCNN model.

  Args:
    batch_size: (int) Value of the static per_replica batch size.
    num_motifs: (int) Number of motifs (= number of filters) to apply to input.
    len_motifs: (int) Length of the motifs (= size of convolutional filters).
    num_denses: (int) Number of nodes in the dense layer.
    num_classes: (int) Number of output classes.
    embed_size: (int) Static size of hidden dimension of the embedding output.
    dropout_rate: (float) Fraction of the convolutional output units and dense.
      layer output units to drop.
    l2_weight: (float) Strength of L2 regularization for weights in the output
      dense layer.
    one_hot: (bool) If using one hot encoding to encode input sequences.

  Returns:
    (tf.keras.Model) The 1D convolutional model for genomic sequences.
  """
  inputs = tf.keras.Input(shape=[None], batch_size=batch_size, dtype=tf.int32)

  if one_hot:
    x = tf.one_hot(inputs, depth=VOCAB_SIZE)
  else:
    x = tf.keras.layers.Embedding(
        VOCAB_SIZE, embed_size, name='embedding')(
            inputs)
  x = _conv_pooled_block(x, num_motifs, len_motifs, embed_size, l2_weight)
  x = tf.keras.layers.Dropout(dropout_rate, name='dropout1')(x)
  x = tf.keras.layers.Dense(
      num_denses,
      activation=tf.keras.activations.relu,
      kernel_regularizer=tf.keras.regularizers.l2(l2_weight),
      name='dense')(x)
  x = tf.keras.layers.Dropout(dropout_rate, name='dropout2')(x)
  x = tf.keras.layers.Dense(
      num_classes,
      activation=None,
      kernel_regularizer=tf.keras.regularizers.l2(l2_weight),
      name='logits')(x)

  return tf.keras.Model(inputs=inputs, outputs=x)

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

"""Genomics CNN with SNGP model.

A 1D CNN model with spectral normalization and Gaussian process for genomic
sequence classification.

## References
[1]: Ren, Jie, Peter J. Liu, Emily Fertig, Jasper Snoek, Ryan Poplin,
     Mark Depristo, Joshua Dillon, and Balaji Lakshminarayanan.
     "Likelihood ratios for out-of-distribution detection."
     In Advances in Neural Information Processing Systems, pp. 14707-14718.
     2019.
     http://papers.nips.cc/paper/9611-likelihood-ratios-for-out-of-distribution-detection
[2]: Liu, Jeremiah Zhe, Zi Lin, Shreyas Padhy, Dustin Tran, Tania Bedrax-Weiss,
     and Balaji Lakshminarayanan. "Simple and Principled Uncertainty Estimation
     with Deterministic Deep Learning via Distance Awareness." arXiv preprint
     arXiv:2006.10108 (2020).
"""

from typing import Any, Dict, Optional
from absl import logging
import tensorflow as tf
from models import util as models_util  # local file import from experimental.single_model_uncertainty


VOCAB_SIZE = 4  # DNA sequences are composed of {A, C, G, T}


def _conv_pooled_block(
    inputs: tf.Tensor,
    conv_layer: tf.keras.layers.Layer,
) -> tf.Tensor:
  """Creates an 1D convolutional layer with max pooling over sequence positions.

  Args:
    inputs: (tf.Tensor) Input tensor, shape (batch_size, seq_len, embed_size).
    conv_layer: (tf.keras.layers.Layer) 2D convolutional layer.

  Returns:
    (tf.Tensor) shape (batch_size, 1, num_motifs).
  """
  logging.info('conv input shape %s', inputs.shape)
  # expend last dim to fit conv2d
  out = tf.expand_dims(inputs, axis=3)  # [batch_size, seq_len, embed_size, 1]
  out = conv_layer(out)  # [batch_size, seq_len, embed_size, num_motifs]
  # TODO(jjren) indices=1 only works for embed_size=4
  out = tf.gather(out, indices=1, axis=2)  # [batch_size, seq_len, num_motifs]

  # Max pooling over all positions for each motifs.
  out = tf.reduce_max(out, axis=1)  # [batch_size, num_motifs]

  return out


def _input_embedding(inputs, vocab_size, one_hot=True, embed_size=None):
  """Transform input integers into one-hot encodings or through embeddings."""
  if one_hot:
    out = tf.one_hot(inputs, depth=vocab_size)
    embed_size = vocab_size
  else:
    if embed_size:
      out = tf.keras.layers.Embedding(vocab_size, embed_size)(inputs)
    else:
      raise ValueError('Embed input integers but embedding size is not given.')
  return out


def genomics_cnn(batch_size: int,
                 len_seqs: int,
                 num_motifs: int,
                 len_motifs: int,
                 num_denses: int,
                 num_classes: int = 10,
                 embed_size: int = 4,
                 one_hot: bool = True,
                 l2_weight: float = 0.0,
                 dropout_rate: float = 0.1,
                 before_conv_dropout: bool = False,
                 use_mc_dropout: bool = False,
                 spec_norm_hparams: Optional[Dict[str, Any]] = None,
                 gp_layer_hparams: Optional[Dict[str, Any]] = None,
                 **unused_kwargs: Dict[str, Any]) -> tf.keras.models.Model:

  """Builds Genomics CNN model.

  Args:
    batch_size: (int) Value of the static per_replica batch size.
    len_seqs: (int) Sequence length.
    num_motifs: (int) Number of motifs (= number of filters) to apply to input.
    len_motifs: (int) Length of the motifs (= size of convolutional filters).
    num_denses: (int) Number of nodes in the dense layer.
    num_classes: (int) Number of output classes.
    embed_size: (int) Static size of hidden dimension of the embedding output.
    one_hot: (bool) If using one hot encoding to encode input sequences.
    l2_weight: (float) L2 regularization coefficient.
    dropout_rate: (float) Fraction of the convolutional output units and dense.
      layer output units to drop.
    before_conv_dropout: (bool) Whether to use filter wise dropout before the
      convolutional layer.
    use_mc_dropout: (bool) Whether to apply Monte Carlo dropout.
    spec_norm_hparams: (dict) Hyperparameters for spectral normalization.
    gp_layer_hparams: (dict) Hyperparameters for Gaussian Process output layer.
    **unused_kwargs: (dict) Unused keyword arguments that will be ignored by the
      model.

  Returns:
    (tf.keras.Model) The 1D convolutional model for genomic sequences.
  """
  # define layers
  if spec_norm_hparams:
    spec_norm_bound = spec_norm_hparams['spec_norm_bound']
    spec_norm_iteration = spec_norm_hparams['spec_norm_iteration']
  else:
    spec_norm_bound = None
    spec_norm_iteration = None

  conv_layer = models_util.make_conv2d_layer(
      use_spec_norm=(spec_norm_hparams is not None),
      spec_norm_bound=spec_norm_bound,
      spec_norm_iteration=spec_norm_iteration)

  dense_layer = models_util.make_dense_layer(
      use_spec_norm=(spec_norm_hparams is not None),
      spec_norm_bound=spec_norm_bound,
      spec_norm_iteration=spec_norm_iteration)

  output_layer = models_util.make_output_layer(
      gp_layer_hparams=gp_layer_hparams)

  # compute outputs given inputs
  inputs = tf.keras.Input(
      shape=[len_seqs], batch_size=batch_size, dtype=tf.int32)
  x = _input_embedding(
      inputs, VOCAB_SIZE, one_hot=one_hot, embed_size=embed_size)

  # filter-wise dropout before conv,
  # x.shape=[batch_size, len_seqs, vocab_size/embed_size]
  if before_conv_dropout:
    x = models_util.apply_dropout(
        x,
        dropout_rate,
        use_mc_dropout,
        filter_wise_dropout=True,
        name='conv_dropout')

  x = _conv_pooled_block(
      x,
      conv_layer=conv_layer(
          filters=num_motifs,
          kernel_size=(len_motifs, embed_size),
          strides=(1, 1),
          kernel_regularizer=tf.keras.regularizers.l2(l2_weight),
          name='conv'))
  x = models_util.apply_dropout(
      x, dropout_rate, use_mc_dropout, name='dropout1')
  x = dense_layer(
      units=num_denses,
      activation=tf.keras.activations.relu,
      kernel_regularizer=tf.keras.regularizers.l2(l2_weight),
      name='dense')(
          x)
  x = models_util.apply_dropout(
      x, dropout_rate, use_mc_dropout, name='dropout2')
  if gp_layer_hparams and gp_layer_hparams['gp_input_dim'] > 0:
    # Uses random projection to reduce the input dimension of the GP layer.
    x = tf.keras.layers.Dense(
        gp_layer_hparams['gp_input_dim'],
        kernel_initializer='random_normal',
        use_bias=False,
        trainable=False,
        name='gp_random_projection')(
            x)
  outputs = output_layer(num_classes, name='logits')(x)
  return tf.keras.Model(inputs=inputs, outputs=outputs)

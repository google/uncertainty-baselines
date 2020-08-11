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

import functools
from typing import Any, Dict
from absl import logging
from edward2.experimental import sngp
import tensorflow.compat.v2 as tf
import uncertainty_baselines.experiments.single_model_uncertainty.models.util as models_util


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


def create_model(
    batch_size: int,
    len_seqs: int,
    num_motifs: int,
    len_motifs: int,
    num_denses: int,
    num_classes: int = 10,
    embed_size: int = 4,
    one_hot: bool = True,
    dropout_rate: float = 0.1,
    use_mc_dropout: bool = False,
    spec_norm_hparams: Dict[str, Any] = None,
    gp_layer_hparams: Dict[str, Any] = None,
) -> tf.keras.models.Model:

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
    dropout_rate: (float) Fraction of the convolutional output units and dense.
      layer output units to drop.
    use_mc_dropout: (bool) Whether to apply Monte Carlo dropout.
    spec_norm_hparams: (dict) Hyperparameters for spectral normalization.
    gp_layer_hparams: (dict) Hyperparameters for Gaussian Process output layer.

  Returns:
    (tf.keras.Model) The 1D convolutional model for genomic sequences.
  """
  inputs = tf.keras.Input(
      shape=[len_seqs], batch_size=batch_size, dtype=tf.int32)

  if one_hot:
    x = tf.one_hot(inputs, depth=VOCAB_SIZE)
    embed_size = VOCAB_SIZE
  else:
    x = tf.keras.layers.Embedding(
        VOCAB_SIZE, embed_size, name='embedding')(
            inputs)
  # apply filter-wise dropout to x, x.shape=[batch_size, len_seqs, embed_size]
  x = models_util.apply_dropout(
      x, dropout_rate, use_mc_dropout, filter_wise_dropout=True)
  if spec_norm_hparams:
    spec_norm_bound = spec_norm_hparams['spec_norm_bound']
    spec_norm_iteration = spec_norm_hparams['spec_norm_iteration']
  else:
    spec_norm_bound = None
    spec_norm_iteration = None
  conv2d = models_util.make_conv2d_layer(
      num_filters=num_motifs,
      kernel_size=(len_motifs, embed_size),
      strides=(1, 1),
      activation='relu',
      use_spec_norm=(spec_norm_hparams is not None),
      spec_norm_bound=spec_norm_bound,
      spec_norm_iteration=spec_norm_iteration)
  x = _conv_pooled_block(x, conv_layer=conv2d())
  x = models_util.apply_dropout(
      x, dropout_rate, use_mc_dropout, filter_wise_dropout=False)
  x = tf.keras.layers.Dense(
      num_denses,
      activation=tf.keras.activations.relu,
      name='dense')(x)
  x = tf.keras.layers.Dropout(dropout_rate, name='dropout2')(x)
  x = tf.keras.layers.Dense(
      num_classes,
      activation=None,
      name='logits')(x)
  x = models_util.apply_dropout(
      x, dropout_rate, use_mc_dropout, filter_wise_dropout=False)
  if gp_layer_hparams:
    gp_output_layer = functools.partial(
        sngp.RandomFeatureGaussianProcess,
        num_inducing=gp_layer_hparams['gp_hidden_dim'],
        gp_kernel_scale=gp_layer_hparams['gp_scale'],
        gp_output_bias=gp_layer_hparams['gp_bias'],
        normalize_input=gp_layer_hparams['gp_input_normalization'],
        gp_cov_momentum=gp_layer_hparams['gp_cov_discount_factor'],
        gp_cov_ridge_penalty=gp_layer_hparams['gp_cov_ridge_penalty'])
    outputs = gp_output_layer(num_classes)(x)
  else:
    outputs = tf.keras.layers.Dense(num_classes, name='dense_to_logits')(x)
  return tf.keras.Model(inputs=inputs, outputs=outputs)

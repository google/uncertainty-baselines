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

"""TextCNN model.

A standard 1D CNN model for sentence classification.

## References
[1]: Yoon Kim. Convolutional Neural Networks for Sentence Classification.
     In _Empirical Methods in Natural Language Processing_, 2014.
     https://www.aclweb.org/anthology/D14-1181/
"""

import tensorflow as tf


def embedding_block(inputs,
                    vocab_size,
                    feature_size,
                    embed_size,
                    premade_embedding_arr=None):
  """Creates Embedding Layer.

  Args:
    inputs: (tf.Tensor) Input sentence in token indices format, shape
      (batch_size, feature_size).
    vocab_size: (int) Static size of vocabulary.
    feature_size: (int) Static size of input feature.
    embed_size: (int) Static size of hidden dimension of the embedding output.
    premade_embedding_arr: (np.ndarray) Pre-made word embedding in numpy array
      format, shape (vocab_size, embed_size).

  Raises:
    (ValueError): If shape of premade_embedding_arr is not
      (vocab_size, embed_size).

  Returns:
    (tf.Tensor) shape (batch_size, feature_size, embed_size).
  """
  # Make initializer.
  if premade_embedding_arr is not None:
    premade_vocab_size, premade_embed_size = premade_embedding_arr.shape
    if premade_vocab_size != vocab_size or premade_embed_size != embed_size:
      raise ValueError("'premade_embedding_arr' should have size ({}, {}). "
                       'Observed ({}, {})'.format(vocab_size, embed_size,
                                                  premade_vocab_size,
                                                  premade_embed_size))
    embed_init = tf.keras.initializers.Constant(premade_embedding_arr)
  else:
    embed_init = tf.keras.initializers.RandomUniform(minval=-1, maxval=1)

  # Define layer.
  embedding_layer = tf.keras.layers.Embedding(
      vocab_size,
      embed_size,
      input_length=feature_size,
      embeddings_initializer=embed_init,
      name='embedding')
  return embedding_layer(inputs)


def conv_pooled_block(inputs, num_filters, filter_size, feature_size,
                      embed_size):
  """Creates convolution layer with max pooling.

  Args:
    inputs: (tf.Tensor) Input tensor, shape (batch_size, feature_size,
      embed_size).
    num_filters: (int) Number of filters to apply to input.
    filter_size: (int) Static size of the convolutional filter.
    feature_size: (int) Static size of input feature.
    embed_size: (int) Static size of hidden dimension of the text embedding.

  Returns:
    (tf.Tensor) shape (batch_size, 1, num_filters).
  """
  filter_shape = (filter_size, embed_size)
  max_pool_shape = (feature_size - filter_size + 1, 1)

  conv_layer = tf.keras.layers.Conv2D(
      num_filters,
      filter_shape,
      strides=(1, 1),
      padding='valid',
      data_format='channels_last',
      activation='relu',
      kernel_initializer='glorot_normal',
      bias_initializer=tf.keras.initializers.constant(0.1),
      name='convolution_{:d}'.format(filter_size))

  # Max pooling over sentence positions for each filter.
  maxpool_layer = tf.keras.layers.MaxPool2D(
      pool_size=max_pool_shape,
      strides=(1, 1),
      padding='valid',
      data_format='channels_last',
      name='max_pooling_{:d}'.format(filter_size))

  conv = conv_layer(inputs)
  return maxpool_layer(conv)


def textcnn(filter_sizes,
            num_filters,
            num_classes,
            feature_size,
            vocab_size,
            embed_size,
            dropout_rate,
            l2,
            premade_embedding_arr=None):
  """Builds TextCNN model.

  Args:
    filter_sizes: (list) A list specifying the sizes of the convolutional
      filters.
    num_filters: (int) Number of filters to apply to input.
    num_classes: (int) Number of output classes.
    feature_size: (int) Static size of input feature.
    vocab_size: (int) Static size of vocabulary.
    embed_size: (int) Static size of hidden dimension of the embedding output.
    dropout_rate: (float) Fraction of the convolutional output units to drop.
    l2: (float) Strength of L2 regularization for wieghts in the output dense
      layer.
    premade_embedding_arr: (np.ndarray) Pre-made word embedding in numpy array
      format, shape (vocab_size, embed_size).

  Returns:
    (tf.keras.Model) The TextCNN model.
  """
  inputs = tf.keras.Input(shape=(feature_size,))

  # Prepare word embedding for convolutional layers.
  embed = embedding_block(
      inputs,
      vocab_size,
      feature_size,
      embed_size,
      premade_embedding_arr=premade_embedding_arr)
  embed = tf.keras.layers.Reshape((feature_size, embed_size, 1),
                                  name='add_channel')(
                                      embed)

  # Evaluate and gather conv layer output for each filter size.
  pool_outputs = []
  for filter_size in filter_sizes:
    pool = conv_pooled_block(embed, num_filters, filter_size, feature_size,
                             embed_size)
    pool_outputs.append(pool)

  pool_outputs = tf.keras.layers.concatenate(
      pool_outputs, axis=-1, name='concatenate')

  # Flatten and apply dropout.
  flat_outputs = tf.keras.layers.Flatten(
      data_format='channels_last', name='flatten')(
          pool_outputs)
  flat_outputs = tf.keras.layers.Dropout(
      dropout_rate, name='dropout')(
          flat_outputs)

  # Dense output.
  dense_output_layer = tf.keras.layers.Dense(
      num_classes,
      activation=None,
      kernel_initializer='glorot_normal',
      bias_initializer=tf.keras.initializers.constant(0.1),
      kernel_regularizer=tf.keras.regularizers.l2(l2),
      bias_regularizer=tf.keras.regularizers.l2(l2),
      name='dense')
  outputs = dense_output_layer(flat_outputs)

  return tf.keras.Model(inputs=inputs, outputs=outputs)

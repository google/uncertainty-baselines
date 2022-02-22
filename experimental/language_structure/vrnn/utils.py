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

"""utils methods/classes."""

from typing import Any, Dict, Optional, Sequence

import numpy as np
import sklearn.metrics
import tensorflow as tf
import tensorflow_hub as hub

PADDING_VALUE = 0


def state_is_tuple(cell_type):
  return cell_type == 'lstm'


def create_mask(inputs: tf.Tensor,
                masking_prob: Dict[Any, float],
                seed: Optional[int] = None) -> tf.Tensor:
  """Creates mask by the masking probability of each element in the inputs."""
  threshold = tf.zeros_like(inputs, dtype=tf.float32)
  for element, ratio in masking_prob.items():
    threshold += tf.where(tf.equal(inputs, element), ratio, 0.0)
  prob = tf.random.uniform(inputs.shape, minval=0, maxval=1, seed=seed)
  return tf.cast(prob < threshold, tf.int32)


def value_in_tensor(inputs: tf.Tensor, tensor: tf.Tensor) -> tf.Tensor:
  """Checks if each element in `inputs` is in `tensor`."""
  tile_multiples = tf.concat(
      [tf.ones(tf.rank(inputs), dtype=tf.int32),
       tf.shape(tensor)], axis=0)
  inputs = tf.tile(tf.expand_dims(inputs, -1), tile_multiples)
  return tf.reduce_any(tf.equal(inputs, tensor), -1)


def create_rebalanced_sample_weights(
    labels: tf.Tensor,
    dtype: Optional[tf.dtypes.DType] = tf.float32,
    mask_padding: Optional[bool] = True) -> tf.Tensor:
  """Creates the sample weights by inverse of label counts."""
  unique_label, _, count = tf.unique_with_counts(tf.reshape(labels, [-1]))
  weights = tf.reduce_min(count) / count
  sample_weights = tf.map_fn(
      fn=lambda t: tf.where(labels == tf.cast(t[0], dtype=labels.dtype), t[1], 0
                           ),
      elems=tf.stack([tf.cast(unique_label, dtype=weights.dtype), weights],
                     axis=1))
  sample_weights = tf.cast(tf.reduce_sum(sample_weights, axis=0), dtype=dtype)
  if mask_padding:
    sample_weights *= tf.cast(tf.sign(labels), dtype=dtype)
  sample_weights /= tf.reduce_mean(sample_weights)
  return sample_weights


def get_rnn_cls(cell_type: str):
  if cell_type == 'lstm':
    return tf.keras.layers.LSTM
  elif cell_type == 'gru':
    return tf.keras.layers.GRU
  else:
    return tf.keras.layers.SimpleRNN


def get_rnn_cell(cell_type: str):
  if cell_type == 'lstm':
    return tf.keras.layers.LSTMCell
  elif cell_type == 'gru':
    return tf.keras.layers.GRUCell
  else:
    return tf.keras.layers.SimpleRNNCell


def to_one_hot(x) -> tf.Tensor:
  """Returns the argmax of the input tensor in one-hot format."""
  indices = tf.math.argmax(x, axis=1)
  depth = x.shape.as_list()[-1]
  x_hard = tf.one_hot(indices, depth, dtype=x.dtype)
  return tf.stop_gradient(x_hard - x) + x


def get_last_step(inputs: tf.Tensor, seq_length: tf.Tensor) -> tf.Tensor:
  """Returns the last step of inputs by the sequence length.

  If the sequence length is zero, it will return the zero tensor.

  Args:
    inputs: tensor of [batch_size, max_seq_length, hidden_size].
    seq_length: tensor of [batch_size] recording the actual length of inputs.

  Returns:
    tensor of [batch_size, hidden_size], where tensor[i, :] = inputs[i,
    seq_length[i], :]
  """
  batch_range = tf.range(tf.shape(seq_length)[0])

  non_empty_seq = tf.sign(seq_length)
  safe_indices = tf.cast((seq_length - non_empty_seq), dtype=tf.int32)
  indices = tf.stack([batch_range, safe_indices], axis=1)
  result = tf.gather_nd(inputs, indices)
  # Expand axis to broadcast to the second dimension (hidden size).
  result *= tf.expand_dims(tf.cast(non_empty_seq, dtype=result.dtype), axis=1)
  return result


# thanks for the implementation at
# https://blog.evjang.com/2016/11/tutorial-categorical-variational.html
def sample_gumbel(shape, eps=1e-20):
  """Sample from Gumbel 0 to 1."""
  uniform = tf.random.uniform(shape, minval=0, maxval=1)
  return -tf.math.log(-tf.math.log(uniform + eps) + eps)


def gumbel_softmax_sample(logits, temperature):
  """Draw a sample from the Gumbel-Softmax distribution."""
  y = logits + sample_gumbel(tf.shape(logits))
  y_adjusted = y / temperature
  return tf.nn.softmax(y_adjusted), y_adjusted


class GumbelSoftmaxSampler(tf.keras.layers.Layer):
  """Gumbel-Softmax sampler.

  Sample from the Gumbel-Softmax distribution and optionally discretize.
  """

  def __init__(self,
               temperature,
               hard: bool = False,
               trainable_temperature: bool = True):
    """GumbelSoftmaxSampler constructor.

    Args:
      temperature: non-negative scalar
      hard: if True, take argmax, but differentiate w.r.t. soft sample y
      trainable_temperature: whether temperature is trainable
    """
    self._trainable_temperature = trainable_temperature
    self._initial_temperature = temperature
    self._hard = hard

    super(GumbelSoftmaxSampler, self).__init__()

  def build(self, input_shape):
    self._temperature = self.add_weight(
        'temperature',
        initializer=tf.keras.initializers.Constant(self._initial_temperature),
        trainable=self._trainable_temperature)
    super().build(input_shape)

  def call(self, logits: tf.Tensor, return_logits: bool = False):
    """Sample from the Gumbel-Softmax distribution and optionally discretize.

    Args:
      logits: [batch_size, n_class] unnormalized log-probs.
      return_logits: whether to also return logits tensor.

    Returns:
      A [batch_size, n_class] sample from the Gumbel-Softmax distribution.
      If self._hard=True, then the returned sample will be one-hot, otherwise it
      will be a probabilitiy distribution that sums to 1 across classes.
    """
    y, logits = gumbel_softmax_sample(logits, self._temperature)
    if self._hard:
      y = to_one_hot(y)
    if return_logits:
      return y, logits
    return y


class MLP(tf.keras.Model):
  """Multilayer perceptron."""

  def __init__(self,
               output_sizes: Sequence[int],
               use_bias: bool = True,
               dropout: float = 0.5,
               hidden_activation: Optional[Any] = None,
               final_activation: Optional[Any] = None):
    super(MLP, self).__init__()

    self._layers = []
    for output_size in output_sizes:
      self._layers.append(
          tf.keras.layers.Dense(
              output_size, activation=hidden_activation, use_bias=use_bias))
    if dropout not in (None, 0):
      self._layers.append(tf.keras.layers.Dropout(dropout))
    if final_activation:
      self._layers.append(final_activation)

  def call(self, inputs):
    outputs = inputs
    for layer in self._layers:
      outputs = layer(outputs)
    return outputs


class SequentialWordLoss(tf.keras.losses.SparseCategoricalCrossentropy):
  """Cross entropy loss of the word id sequences."""

  def __init__(self, *args, word_weights: Optional[Any] = None, **kwargs):
    """SequentialWordLoss constructor.

    Args:
      *args: optional arguments passed to
        tf.keras.losses.SparseCategoricalCrossentropy.
      word_weights: of shape [vocab_size], the weights of each token, used to
        rescale loss. word_weights[0] should be the weight of the padding token
        id 0.
      **kwargs: optional arguments passed to
        tf.keras.losses.SparseCategoricalCrossentropy.
    """
    # Disable reduction to be able to apply sequence mask and (optional) word
    # weights.
    super(SequentialWordLoss, self).__init__(
        reduction=tf.keras.losses.Reduction.NONE, *args, **kwargs)
    self._word_weights = word_weights

  def call(self, y_true, y_pred, sample_weight: Optional[tf.Tensor] = None):
    loss = super().call(y_true=y_true, y_pred=y_pred)
    if sample_weight:
      sample_weight = tf.cast(sample_weight, dtype=loss.dtype)
      loss *= sample_weight
    if self._word_weights is not None:
      word_idx = tf.cast(y_true, tf.int32)
      weights = tf.gather(self._word_weights, word_idx)
      loss *= tf.cast(weights, dtype=loss.dtype)
    return loss


class BowLoss(SequentialWordLoss):
  """Bag-of-word loss [1].

  Reference:
  [1]: Zhao et al. Learning Discourse-level Diversity for Neural Dialog Models
    using Conditional Variational Autoencoders. https://arxiv.org/abs/1703.10960
  """

  def __init__(self, *args, sequence_axis: Optional[int] = 1, **kwargs):
    """BowLoss Constructor.

    Args:
      *args: arguments passed to super class SequentialWordLoss.
      sequence_axis: the axis of the sequence dimension bow logits to be
        repeated.
      **kwargs: arguments passed to super class SequentialWordLoss.
    """
    super(BowLoss, self).__init__(*args, **kwargs)
    self._sequence_axis = sequence_axis

  def call(self, y_true, bow_pred, sample_weight: Optional[tf.Tensor] = None):
    """Computes bow loss.

    Args:
      y_true: the label tensor, of shape [d0, d1, ..., dN] where dN =
        self._sequence_axis.
      bow_pred: the bow prediction logits, of shape [d0, d1, ..., d_{N-1}, H].
        It will be repeated to [d0, d1, ..., d_{N-1}, dN, H] and compute
        SequentialWordLoss with y_true.
      sample_weight: the optional tensor of shape [d0, d1, ..., dN] specifying
        the weight to rescale the loss.

    Returns:
      loss: tensor of shape [d0, d1, ..., dN].
    """
    y_true_shape = tf.shape(y_true)
    y_true_rank = len(y_true.shape)
    axis = self._sequence_axis
    if y_true_rank <= axis:
      raise ValueError(
          'Expected sequence axis {}, but y_true has a lower rank {}: {}'
          .format(axis, y_true_rank, y_true_shape))

    # Step 1/2: construct the multiples for tf.tile; insert the max_seq_length
    # multiple in the sequence axis. It's equivalent to:
    #   multiples = [1] * y_true_rank
    #   multiples.insert(axis, y_true_shape[axis])
    multiples = tf.concat([[1] * axis, [y_true_shape[axis]], [1] *
                           (y_true_rank - axis)],
                          axis=0)
    # Step 2/2: repeat `bow_pred` to match `y_true` on the sequence axis.
    y_pred = tf.tile(tf.expand_dims(bow_pred, axis=axis), multiples)
    loss = super().call(y_true, y_pred, sample_weight)
    return loss


class KlLoss(tf.keras.losses.KLDivergence):
  """KL divergence with Batch Prior Regularization support [1].

  Reference:
  [1]: Zhao et al. Learning Discourse-level Diversity for Neural Dialog Models
    using Conditional Variational Autoencoders. https://arxiv.org/abs/1703.10960
  """

  def __init__(self,
               bpr: bool,
               *args,
               from_logits: Optional[bool] = False,
               **kwargs):
    super(KlLoss, self).__init__(*args, **kwargs)
    self._bpr = bpr
    self._from_logits = from_logits

  def call(self, p_z, q_z):
    if self._from_logits:
      p_z = tf.nn.softmax(p_z)
      q_z = tf.nn.softmax(q_z)

    if self._bpr:
      if not p_z.shape.is_compatible_with(q_z.shape):
        raise ValueError(
            'Inconsistent shape between p_z_logits {} and q_z_logits {}'.format(
                p_z.shape, q_z.shape))
      batch_size = tf.shape(p_z)[0]
      p_z = tf.reduce_mean(p_z, axis=0)
      q_z = tf.reduce_mean(q_z, axis=0)
      loss = super().call(q_z, p_z) * tf.cast(batch_size, p_z.dtype)
    else:
      loss = super().call(q_z, p_z)
    return loss


class BertPreprocessor(tf.keras.Model):
  """Preprocessor converting text into BERT input formats."""

  def __init__(self, tfhub_url: str, max_seq_length: int):
    super(BertPreprocessor, self).__init__()

    self._tfhub_url = tfhub_url
    self._max_seq_length = max_seq_length

    preprocess = hub.load(self._tfhub_url)
    self._special_tokens_dict = preprocess.tokenize.get_special_tokens_dict()

    self.tokenizer = hub.KerasLayer(preprocess.tokenize, name='tokenizer')
    self.packer = hub.KerasLayer(
        preprocess.bert_pack_inputs,
        arguments=dict(seq_length=self._max_seq_length),
        name='packer')

  def call(self, inputs: Sequence[tf.Tensor], concat: Optional[bool] = False):
    segments = [self.tokenizer(input) for input in inputs]
    truncated_segments = [
        segment[:, :self._max_seq_length] for segment in segments
    ]
    if concat:
      return self.packer(truncated_segments)
    return [self.packer([segment]) for segment in truncated_segments]

  @property
  def vocab_size(self) -> int:
    return self._special_tokens_dict['vocab_size'].numpy().item()


def _get_flatten_non_padding_value(
    tensors: Sequence[tf.Tensor],
    mask_gen_tensor: tf.Tensor) -> Sequence[tf.Tensor]:
  """Returns the flatten tensors with the padding filtered."""
  mask_gen_tensor = tf.reshape(mask_gen_tensor, [-1])
  padding_mask = mask_gen_tensor != PADDING_VALUE
  outputs = []
  for tensor in tensors:
    tensor = tf.reshape(tensor, [-1])
    outputs.append(tf.boolean_mask(tensor, padding_mask))
  return outputs


def adjusted_mutual_info(y_true: tf.Tensor, y_pred: tf.Tensor) -> float:
  """Computes adjusted mutual information of non-padded prediction and label."""
  # pylint: disable=unbalanced-tuple-unpacking
  y_pred, y_true = _get_flatten_non_padding_value([y_pred, y_true],
                                                  mask_gen_tensor=y_true)
  return sklearn.metrics.adjusted_mutual_info_score(y_true, y_pred)


def cluster_purity(y_true: tf.Tensor, y_pred: tf.Tensor) -> float:
  """Computes cluster purity of non-padded prediction and label."""
  # pylint: disable=unbalanced-tuple-unpacking
  y_pred, y_true = _get_flatten_non_padding_value([y_pred, y_true],
                                                  mask_gen_tensor=y_true)
  contingency_matrix = sklearn.metrics.cluster.contingency_matrix(
      y_true, y_pred)
  return np.sum(np.amax(contingency_matrix,
                        axis=0)) / np.sum(contingency_matrix)

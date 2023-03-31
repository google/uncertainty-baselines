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

"""Differentiable PSL constraints.

File consists of:
- Differentiable PSL constraints for dialog structure rules.
"""

from typing import List, Sequence

import tensorflow as tf
from psl import psl_model  # local file import from experimental.language_structure


class PSLModelDSTCSynthetic(psl_model.PSLModel):
  """Defining PSL rules for the DSTC Synthetic dataset."""

  def __init__(self,
               rule_weights: List[float],
               rule_names: List[str],
               logic: str = 'lukasiewicz',
               **kwargs) -> None:
    super().__init__(rule_weights, rule_names, logic=logic, **kwargs)

    for option in ['config']:
      if option not in kwargs:
        raise KeyError('Missing argument: %s' % (option,))

    self.config = kwargs['config']
    self.batch_size = self.config['batch_size']
    self.dialog_size = self.config['max_dialog_size']
    self.word_weights = self.config['word_weights']
    self.embed_layer = tf.keras.layers.Embedding(
        self.word_weights.shape[0],
        self.word_weights.shape[1],
        embeddings_initializer=tf.keras.initializers.Constant(
            self.word_weights),
        trainable=False)
    self.predicates = {}

  def _get_tensor_column(self, data, index):
    """Gathers a column in a tensor and reshapes."""
    return tf.reshape(tf.gather(data, index, axis=-1), [self.batch_size, -1])

  def _has_word(self, data, index):
    """Checks if the data contain a specific token."""
    word = self._get_tensor_column(data, index)
    word_mapping = tf.equal(word, self.config['includes_word'])
    return tf.cast(word_mapping, tf.float32)

  def _first_statement(self):
    """Creates a (batch_size, dialog_size) first statement mask."""
    return tf.constant([[1.0] + [0.0] * (self.dialog_size - 1)] *
                       self.batch_size)

  def _last_statement(self, data):
    """Creates a (batch_size, dialog_size) end statement mask."""
    end = self._get_tensor_column(data, self.config['mask_index'])
    end_mask = tf.equal(end, self.config['last_utterance_mask'])
    return tf.cast(end_mask, tf.float32)

  def _previous_statement(self, data):
    """Creates a cross product matrix representing the previous statements.

      Creates a cross product matrix masked to contain only the previous
      statement values. This matrix is max_dialog_size by max_dialog_size.

      For example, given a dialog with three utterances and a single padded
      utterance, the matrix would look like:

           Utt1 Utt2 Utt3 Pad1
      Utt1  0    1    0    0
      Utt2  0    0    1    0
      Utt3  0    0    0    0
      Pad1  0    0    0    0

      Here Utt1 is a previous statement to Utt2 and Utt2 is the previous
      statement to Utt3.

      To create this matrix, an off diagonal matrix is created:

           Utt1 Utt2 Utt3 Pad1
      Utt1  0    1    0    0
      Utt2  0    0    1    0
      Utt3  0    0    0    1
      Pad1  0    0    0    0

      And multiplied by a matrix that masks out the padding:

           Utt1 Utt2 Utt3 Pad1
      Utt1  1    1    1    0
      Utt2  1    1    1    0
      Utt3  1    1    1    0
      Pad1  0    0    0    0

    Args:
        data: input features used to produce the logits.

    Returns:
        A cross product matrix mask containing the previous statements.
    """
    off_diagonal_matrix = tf.linalg.diag([1.0] * (self.dialog_size - 1), k=1)

    # Creates a (batch_size, dialog_size, dialog_size) tensor indicating which
    # utterances have padding (see docstirng for details).
    padding = self._get_tensor_column(data, self.config['mask_index'])
    padding_mask = tf.equal(padding, self.config['utterance_mask'])
    padding_mask = tf.cast(padding_mask, tf.float32)
    padding_mask = tf.repeat(padding_mask, self.dialog_size, axis=-1)
    padding_mask = tf.reshape(padding_mask,
                              [-1, self.dialog_size, self.dialog_size])

    # Creates a (batch_size, dialog_size, dialog_size) tensor indicating what
    # the previous statements are in a dialog.
    return off_diagonal_matrix * padding_mask

  def generate_predicates(self, input_ids: Sequence[tf.Tensor]):
    """Generates potentials used throughout the rules."""
    hidden = self.embed_layer(input_ids)
    input_mask = tf.sign(input_ids)
    logits = tf.reduce_sum(
        hidden *
        tf.cast(tf.expand_dims(input_mask, axis=-1), dtype=hidden.dtype),
        axis=-2)
    probs = tf.nn.softmax(logits)

    if self.config['hard_pseudo_label']:
      predictions = tf.argmax(probs, axis=-1)
      self.predicates['predictions'] = tf.one_hot(
          predictions, depth=probs.shape[-1])
    else:
      self.predicates['predictions'] = probs

  def rule_1(self, logits, **unused_kwargs) -> float:
    labels = self.predicates['predictions']
    return self.template_rx_implies_sx(
        tf.cast(labels, dtype=logits.dtype), logits)

  def compute_loss_per_rule(self, data: tf.Tensor,
                            logits: tf.Tensor) -> List[float]:
    """Calculate the loss for each of the PSL rules."""
    rule_kwargs = dict(logits=logits, data=data)
    losses = []

    for rule_weight, rule_function in zip(self.rule_weights,
                                          self.rule_functions):
      losses.append(rule_weight * rule_function(**rule_kwargs))
    return losses

  def compute_loss(self, data: tf.Tensor, logits: tf.Tensor) -> float:
    """Calculate the total loss for all PSL rules."""
    return sum(self.compute_loss_per_rule(data, logits))

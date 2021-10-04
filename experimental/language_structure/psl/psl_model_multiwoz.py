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

# Lint as: python3
"""Differentable PSL constraints.

File consists of:
- Differentable PSL constraints for dialog structure rules.
"""

from typing import List

import tensorflow as tf
import psl_model  # local file import


class PSLModelMultiWoZ(psl_model.PSLModel):
  """Defining PSL rules for the MultiWoZ dataset."""

  def __init__(self, rule_weights: List[float], rule_names: List[str],
               **kwargs) -> None:
    super().__init__(rule_weights, rule_names, **kwargs)

    if 'config' not in kwargs:
      raise KeyError('Missing argument: config')
    self.config = kwargs['config']
    self.class_map = self.config['class_map']

  def _first_statement(self, batch_size, dialog_size):
    """Creates a (batch_size, dialog_size) first statement mask."""
    return tf.constant([[1.0] + [0.0] * (dialog_size - 1)] * batch_size)

  def _end_statement(self, data, batch_size):
    """Creates a (batch_size, dialog_size) end statement mask."""
    end = self._get_tensor_column(data, self.config['mask_index'], batch_size)
    end_mask = tf.equal(end, self.config['last_utterance_mask'])
    return tf.cast(end_mask, tf.float32)

  def _previous_statement(self, data, batch_size, dialog_size):
    """Creates a cross product matrix mask representing the previous statements.

    Creates a cross product matrix masked to contain only the previous statement
    values. This matrix is max_dialog_size by max_dialog_size.

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
      batch_size: the batch size
      dialog_size: the length of the dialog

    Returns:
      A cross product matrix mask containing the previoius statements.
    """
    off_diagonal_matrix = tf.linalg.diag([1.0] * (dialog_size - 1), k=1)

    # Creates a (batch_size, dialog_size, dialog_size) tensor indicating which
    # utterances have padding (see docstirng for details).
    padding = self._get_tensor_column(data, self.config['mask_index'],
                                      batch_size)
    padding_mask = tf.equal(padding, self.config['utterance_mask'])
    padding_mask = tf.cast(padding_mask, tf.float32)
    padding_mask = tf.repeat(padding_mask, dialog_size, axis=-1)
    padding_mask = tf.reshape(padding_mask, [-1, dialog_size, dialog_size])

    # Creates a (batch_size, dialog_size, dialog_size) tensor indicating what
    # the previous statements are in a dialog.
    return off_diagonal_matrix * padding_mask

  def _get_tensor_column(self, data, index, batch_size):
    """Gathers a column in a tensor and reshapes."""
    return tf.reshape(tf.gather(data, index, axis=-1), [batch_size, -1])

  def rule_1(self, logits, **unused_kwargs) -> float:
    """Dialog structure rule.

    Rule:
      !FirstStatement -> !State('greet')

    Meaning:
      IF: the utterance is not the first utterance in a dialog.
      THEN: the utterance should not be a greetings.

    Args:
      logits: logits outputed by a neural model.

    Returns:
      A loss incurred by this dialog structure rule.
    """

    batch_size, dialog_size, _ = logits.shape

    # Creates predicates for differentable potentals.
    first_statement = self._first_statement(batch_size, dialog_size)
    state_greet = self._get_tensor_column(logits, self.class_map['greet'],
                                          batch_size)

    return self.template_rx_implies_sx(
        self.soft_not(first_statement), self.soft_not(state_greet))

  def rule_2(self, logits, data, **unused_kwargs) -> float:
    """Dialog structure rule.

    Rule:
      FirstStatement(S) & HasGreetWord(S) -> State(S, 'greet')

    Meaning:
      IF: the first utterance does conatin a common greet word.
      THEN: the utterance should be a greetings.

    Args:
      logits: logits outputed by a neural model.
      data: input features used to produce the logits.

    Returns:
      A loss incurred by this dialog structure rule.
    """
    batch_size, dialog_size, _ = logits.shape

    # Creates predicates for differentable potentals.
    first_statement = self._first_statement(batch_size, dialog_size)
    greet_word = self._get_tensor_column(data, self.config['greet_index'],
                                         batch_size)
    greet_word_mapping = tf.equal(greet_word, self.config['includes_word'])
    has_greet_word = tf.cast(greet_word_mapping, tf.float32)
    state_greet = self._get_tensor_column(logits, self.class_map['greet'],
                                          batch_size)

    return self.template_rx_and_sx_implies_tx(first_statement, has_greet_word,
                                              state_greet)

  def rule_3(self, logits, data, **unused_kwargs):
    """Dialog structure rule.

    Rule:
      FirstStatement(S) & !HasGreetWord(S) -> State(S, 'init_request')

    Meaning:
      IF: the first utterance does not conatin a common greet word.
      THEN: the utterance should be an inital request.

    Args:
      logits: logits outputed by a neural model.
      data: input features used to produce the logits.

    Returns:
      A loss incurred by this dialog structure rule.
    """
    batch_size, dialog_size, _ = logits.shape

    # Creates predicates for differentable potentals.
    first_statement = self._first_statement(batch_size, dialog_size)
    greet_word = self._get_tensor_column(data, self.config['greet_index'],
                                         batch_size)
    greet_word_mapping = tf.equal(greet_word, self.config['includes_word'])
    has_greet_word = tf.cast(greet_word_mapping, tf.float32)
    state_init_request = self._get_tensor_column(logits,
                                                 self.class_map['init_request'],
                                                 batch_size)

    return self.template_rx_and_sx_implies_tx(first_statement,
                                              self.soft_not(has_greet_word),
                                              state_init_request)

  def rule_4(self, logits, data, **unused_kwargs):
    """Dialog structure rule.

    Rule:
      PreviousStatement(S1, S2) & State(S2, 'initial request')
                                  -> State(S1, 'second request')

    Meaning:
      IF: the previous utterance is an initial request.
      THEN: the current utterance is a second request.

    Note:
      This rule requires a space complexity of O(max_dialog_size^2). A more
      efficent implementation can be done with sparse operations, but currently
      this is sufficent as the max dialog length is reasonable.

    Args:
      logits: logits outputed by a neural model.
      data: input features used to produce the logits.

    Returns:
      A loss incurred by this dialog structure rule.
    """
    batch_size, dialog_size, _ = logits.shape

    # Creates predicates for differentable potentals.

    # Creates a (batch_size, dialog_size, dialog_size) cross product matrix
    # representing the previous statements. See docstring of _previous_statement
    # for details.
    previous_statement = self._previous_statement(data, batch_size, dialog_size)

    state_init_request = self._get_tensor_column(logits,
                                                 self.class_map['init_request'],
                                                 batch_size)
    state_second_request = self._get_tensor_column(
        logits, self.class_map['second_request'], batch_size)

    return self.template_rxy_and_sy_implies_tx(previous_statement,
                                               state_init_request,
                                               state_second_request)

  def rule_5(self, logits, data, **unused_kwargs):
    """Dialog structure rule.

    Rule:
      PreviousStatement(S1, S2) & !State(S2, 'greet')
                                                   -> !State(S1, 'init_request')

    Meaning:
      IF: the previous utterancce is not a greetings.
      THEN: the current utterance is not an initial request.

    Note:
      This rule requires a space complexity of O(max_dialog_size^2). A more
      efficent implementation can be done with sparse operations, but currently
      this is sufficent as the max dialog length is reasonable.

    Args:
      logits: logits outputed by a neural model.
      data: input features used to produce the logits.

    Returns:
      A loss incurred by this dialog structure rule.
    """
    batch_size, dialog_size, _ = logits.shape

    # Creates predicates for differentable potentals.

    # Creates a (batch_size, dialog_size, dialog_size) cross product matrix
    # representing the previous statements. See docstring of _previous_statement
    # for details.
    previous_statement = self._previous_statement(data, batch_size, dialog_size)
    state_greet = self._get_tensor_column(logits, self.class_map['greet'],
                                          batch_size)
    state_init_request = self._get_tensor_column(logits,
                                                 self.class_map['init_request'],
                                                 batch_size)

    return self.template_rxy_and_sy_implies_tx(
        previous_statement, self.soft_not(state_greet),
        self.soft_not(state_init_request))

  def rule_6(self, logits, data, **unused_kwargs):
    """Dialog structure rule.

    Rule:
      PreviousStatement(S1, S2) & State(S2, 'greet')
                                                   -> State(S1, 'init_request')

    Meaning:
      IF: the previous utterance is a greetings.
      THEN: the current utterance is an initial request.

    Note:
      This rule requires a space complexity of O(max_dialog_size^2). A more
      efficent implementation can be done with sparse operations, but currently
      this is sufficent as the max dialog length is reasonable.

    Args:
      logits: logits outputed by a neural model.
      data: input features used to produce the logits.

    Returns:
      A loss incurred by this dialog structure rule.
    """
    batch_size, dialog_size, _ = logits.shape

    # Creates predicates for differentable potentals.

    # Creates a (batch_size, dialog_size, dialog_size) cross product matrix
    # representing the previous statements. See docstring of _previous_statement
    # for details.
    previous_statement = self._previous_statement(data, batch_size, dialog_size)
    state_greet = self._get_tensor_column(logits, self.class_map['greet'],
                                          batch_size)
    state_init_request = self._get_tensor_column(logits,
                                                 self.class_map['init_request'],
                                                 batch_size)

    return self.template_rxy_and_sy_implies_tx(previous_statement, state_greet,
                                               state_init_request)

  def rule_7(self, logits, data, **unused_kwargs) -> float:
    """Dialog structure rule.

    Rule:
      LastStatement(S) & HasEndWord(S) -> State(S, 'end')

    Meaning:
      IF: the last utterance contains a commen end word.
      THEN: the utterance should be an ending.

    Args:
      logits: logits outputed by a neural model.
      data: input features used to produce the logits.

    Returns:
      A loss incurred by this dialog structure rule.
    """
    batch_size, _, _ = logits.shape

    # Creates predicates for differentable potentals.
    last_statement = self._end_statement(data, batch_size)
    end_word = self._get_tensor_column(data, self.config['end_index'],
                                       batch_size)
    end_word_mapping = tf.equal(end_word, self.config['includes_word'])
    has_end_word = tf.cast(end_word_mapping, tf.float32)
    state_end = self._get_tensor_column(logits, self.class_map['end'],
                                        batch_size)

    return self.template_rx_and_sx_implies_tx(last_statement, has_end_word,
                                              state_end)

  def compute_loss(self, data: tf.Tensor, logits: tf.Tensor) -> float:
    """Calculate the loss for all PSL rules."""
    total_loss = 0.0
    rule_kwargs = dict(logits=logits, data=data)

    for rule_weight, rule_function in zip(self.rule_weights,
                                          self.rule_functions):
      total_loss += rule_weight * rule_function(**rule_kwargs)

    return total_loss

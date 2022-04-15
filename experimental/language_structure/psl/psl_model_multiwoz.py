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

"""Differentable PSL constraints.

File consists of:
- Differentable PSL constraints for dialog structure rules.
"""

from typing import List

import tensorflow as tf
import psl_model  # local file import from experimental.language_structure.psl


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

  def _get_tensor_column(self, data, index, batch_size):
    """Gathers a column in a tensor and reshapes."""
    return tf.reshape(tf.gather(data, index, axis=-1), [batch_size, -1])

  def _has_word(self, data, batch_size, index):
    word = self._get_tensor_column(data, self.config[index], batch_size)
    word_mapping = tf.equal(word, self.config['includes_word'])
    return tf.cast(word_mapping, tf.float32)

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

  def _next_statement(self, data, batch_size, dialog_size):
    """Creates a cross product matrix mask representing the next statements.

    Creates a cross product matrix masked to contain only the next statement
    values. This matrix is max_dialog_size by max_dialog_size.

    For example, given a dialog with three utterances and a single padded
    utterance, the matrix would look like:

         Utt1 Utt2 Utt3 Pad1
    Utt1  0    0    0    0
    Utt2  1    0    0    0
    Utt3  0    1    0    0
    Pad1  0    0    0    0

    Here Utt2 is a next statement to Utt1 and Utt3 is the next
    statement to Utt2.

    To create this matrix, an off diagonal matrix is created:

         Utt1 Utt2 Utt3 Pad1
    Utt1  0    0    0    0
    Utt2  1    0    0    0
    Utt3  0    1    0    0
    Pad1  0    0    1    0

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
    off_diagonal_matrix = tf.linalg.diag([1.0] * (dialog_size - 1), k=-1)

    # Creates a (batch_size, dialog_size, dialog_size) tensor indicating which
    # utterances have padding (see docstirng for details).
    padding = self._get_tensor_column(data, self.config['mask_index'],
                                      batch_size)
    padding_mask = tf.math.logical_or(
        tf.equal(padding, self.config['utterance_mask']),
        tf.equal(padding, self.config['last_utterance_mask']))
    padding_mask = tf.cast(padding_mask, tf.float32)
    padding_mask = tf.repeat(padding_mask, dialog_size, axis=-1)
    padding_mask = tf.reshape(padding_mask, [-1, dialog_size, dialog_size])

    # Creates a (batch_size, dialog_size, dialog_size) tensor indicating what
    # the previous statements are in a dialog.
    return off_diagonal_matrix * padding_mask

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
    has_greet_word = self._has_word(data, batch_size, 'greet_index')
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
    has_greet_word = self._has_word(data, batch_size, 'greet_index')
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
    has_end_word = self._has_word(data, batch_size, 'end_index')
    state_end = self._get_tensor_column(logits, self.class_map['end'],
                                        batch_size)

    return self.template_rx_and_sx_implies_tx(last_statement, has_end_word,
                                              state_end)

  def rule_8(self, logits, data, **unused_kwargs) -> float:
    """Dialog structure rule.

    Rule:
      LastStatement(S) & HasAcceptWord(S) -> State(S, 'accept')

    Meaning:
      IF: the last utterance contains a commen accept word.
      THEN: the utterance should be an acceptance.

    Args:
      logits: logits outputed by a neural model.
      data: input features used to produce the logits.

    Returns:
      A loss incurred by this dialog structure rule.
    """
    batch_size, _, _ = logits.shape

    # Creates predicates for differentable potentals.
    last_statement = self._end_statement(data, batch_size)
    has_accept_word = self._has_word(data, batch_size, 'accept_index')
    state_accept = self._get_tensor_column(logits, self.class_map['accept'],
                                           batch_size)

    return self.template_rx_and_sx_implies_tx(last_statement, has_accept_word,
                                              state_accept)

  def rule_9(self, logits, data, **unused_kwargs):
    """Dialog structure rule.

    Rule:
      NextStatement(S1, S2) & State(S2, 'end') & HasCancelWord(S1)
                                                 -> State(S1, 'cancel')

    Meaning:
      IF: the next utterance is an ending and the current utterance has a cancel
        word.
      THEN: the current utterance is a cancel.

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
    # representing the next statements. See docstring of _next_statement
    # for details.
    next_statement = self._next_statement(data, batch_size, dialog_size)
    state_cancel = self._get_tensor_column(logits, self.class_map['cancel'],
                                           batch_size)
    has_cancel_word = self._has_word(data, batch_size, 'cancel_index')
    state_end = self._get_tensor_column(logits, self.class_map['end'],
                                        batch_size)

    return self.template_rxy_and_sy_and_tx_implies_ux(next_statement, state_end,
                                                      has_cancel_word,
                                                      state_cancel)

  def rule_10(self, logits, data, **unused_kwargs):
    """Dialog structure rule.

    Rule:
      PreviousStatement(S1, S2) & State(S2, 'second_request')
                       & HasInfoQuestionWord(S1) -> State(S1, 'info_question')

    Meaning:
      IF: the previous utterance is a second request and the current utterance
        has an info question word.
      THEN: the current utterance is an info question.

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
    # representing the next statements. See docstring of _next_statement
    # for details.
    previous_statement = self._previous_statement(data, batch_size, dialog_size)
    state_second_request = self._get_tensor_column(
        logits, self.class_map['second_request'], batch_size)
    has_info_question_word = self._has_word(data, batch_size,
                                            'info_question_index')
    state_info_question = self._get_tensor_column(
        logits, self.class_map['info_question'], batch_size)

    return self.template_rxy_and_sy_and_tx_implies_ux(previous_statement,
                                                      state_second_request,
                                                      has_info_question_word,
                                                      state_info_question)

  def rule_11(self, logits, data, **unused_kwargs) -> float:
    """Dialog structure rule.

    Rule:
      LastStatement(S) & HasInsistWord(S) -> State(S, 'insist')

    Meaning:
      IF: the current utterance is the last utterance in a dialog and it has an
        insist word.
      THEN: the utterance should be an insist.

    Args:
      logits: logits outputed by a neural model.
      data: input features used to produce the logits.

    Returns:
      A loss incurred by this dialog structure rule.
    """
    batch_size, _, _ = logits.shape

    # Creates predicates for differentable potentals.
    last_statement = self._end_statement(data, batch_size)
    has_insist_word = self._has_word(data, batch_size, 'insist_index')
    state_insist = self._get_tensor_column(logits, self.class_map['insist'],
                                           batch_size)

    return self.template_rx_and_sx_implies_tx(last_statement, has_insist_word,
                                              state_insist)

  def rule_12(self, logits, data, **unused_kwargs):
    """Dialog structure rule.

    Rule:
      PreviousStatement(S1, S2) & State(S2, 'second_request')
          & HasSlotQuestionWord(S1) & !HasInfoQuestionWord(S1)
                                      -> State(S1, 'slot_question')

    Meaning:
      IF: the previous utterance is a second request,the current utterance does
        not have a info question word, and the current utterance has a slot
        question word.
      THEN: the current utterance is a slot question.

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
    # representing the next statements. See docstring of _next_statement
    # for details.
    previous_statement = self._previous_statement(data, batch_size, dialog_size)
    state_second_request = self._get_tensor_column(
        logits, self.class_map['second_request'], batch_size)
    has_slot_question_word = self._has_word(data, batch_size,
                                            'slot_question_index')
    has_info_question_word = self._has_word(data, batch_size,
                                            'info_question_index')
    state_slot_question = self._get_tensor_column(
        logits, self.class_map['slot_question'], batch_size)

    return self.template_rxy_and_sy_and_tx_and_ux_implies_vx(
        previous_statement, state_second_request, has_slot_question_word,
        self.soft_not(has_info_question_word), state_slot_question)

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

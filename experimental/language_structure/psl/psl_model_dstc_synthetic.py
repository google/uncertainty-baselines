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

# Lint as: python3
"""Differentiable PSL constraints.

File consists of:
- Differentiable PSL constraints for dialog structure rules.
"""

from typing import List

import tensorflow as tf
import psl_model


class PSLModelDSTCSynthetic(psl_model.PSLModel):
  """Defining PSL rules for the DSTC Synthetic dataset."""

  def __init__(self, rule_weights: List[float], rule_names: List[str],
               **kwargs) -> None:
    super().__init__(rule_weights, rule_names, **kwargs)

    for option in ['config']:
      if option not in kwargs:
        raise KeyError('Missing argument: %s' % (option,))

    self.config = kwargs['config']
    self.batch_size = self.config['batch_size']
    self.dialog_size = self.config['max_dialog_size']
    self.state_transitions = self.config['state_transitions']

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
    return tf.constant(
      [[1.0] + [0.0] * (self.dialog_size - 1)] * self.batch_size)

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

  def rule_1(self, logits, **unused_kwargs) -> float:
    """Dialog structure rule.

      Rule:
        FirstStatement(U1) & Has_Usr_TFIDF(U1, State) -> State(U1, State)

      Meaning:
        IF: the utterance is the first utterance in a dialog and it has
            a known tf-idf word.
        THEN: the utterance is likely to belong to a start State.

      Args:
        logits: logits outputted by a neural model.

      Returns:
        A loss incurred by this dialog structure rule.
      """
    first_statement = self.predicates['first_statement']

    current_state = self._get_tensor_column(logits, 3)
    has_tfidf = self.predicates['3_usr_index']
    return_loss = self.template_rx_and_sx_implies_tx(first_statement,
                                                     has_tfidf,
                                                     current_state)

    current_state = self._get_tensor_column(logits, 7)
    has_tfidf = self.predicates['7_usr_index']
    return_loss += self.template_rx_and_sx_implies_tx(first_statement,
                                                      has_tfidf,
                                                      current_state)

    current_state = self._get_tensor_column(logits, 21)
    has_tfidf = self.predicates['21_usr_index']
    return_loss += self.template_rx_and_sx_implies_tx(first_statement,
                                                      has_tfidf,
                                                      current_state)

    return return_loss

  def rule_2(self, logits, **unused_kwargs) -> float:
    """Dialog structure rule.

      Rule:
        FirstStatement(U1) & Has_Sys_TFIDF(U1, State) -> State(U1, State)

      Meaning:
        IF: the utterance is the first utterance in a dialog and it has
            a known tf-idf word.
        THEN: the utterance is likely to belong to a start State.

      Args:
        logits: logits outputted by a neural model.

      Returns:
        A loss incurred by this dialog structure rule.
      """
    first_statement = self.predicates['first_statement']

    current_state = self._get_tensor_column(logits, 3)
    has_tfidf = self.predicates['3_sys_index']
    return_loss = self.template_rx_and_sx_implies_tx(first_statement,
                                                     has_tfidf,
                                                     current_state)

    current_state = self._get_tensor_column(logits, 7)
    has_tfidf = self.predicates['7_sys_index']
    return_loss += self.template_rx_and_sx_implies_tx(first_statement,
                                                      has_tfidf,
                                                      current_state)

    current_state = self._get_tensor_column(logits, 21)
    has_tfidf = self.predicates['21_sys_index']
    return_loss += self.template_rx_and_sx_implies_tx(first_statement,
                                                      has_tfidf,
                                                      current_state)

    return return_loss

  def rule_3(self, logits, **unused_kwargs) -> float:
    """Dialog structure rule.

      Rule:
        LastStatement(U1) & Has_Usr_TFIDF(U1, State) -> State(U1, State)

      Meaning:
        IF: the utterance is the last utterance in a dialog and it has
            a known tf-idf word.
        THEN: the utterance is likely to belong to a last state.

      Args:
        logits: logits outputted by a neural model.

      Returns:
        A loss incurred by this dialog structure rule.
      """
    last_statement = self.predicates['last_statement']

    current_state = self._get_tensor_column(logits, 5)
    has_tfidf = self.predicates['5_usr_index']
    return_loss = self.template_rx_and_sx_implies_tx(last_statement,
                                                     has_tfidf,
                                                     current_state)

    current_state = self._get_tensor_column(logits, 12)
    has_tfidf = self.predicates['12_usr_index']
    return_loss += self.template_rx_and_sx_implies_tx(last_statement,
                                                      has_tfidf,
                                                      current_state)

    current_state = self._get_tensor_column(logits, 15)
    has_tfidf = self.predicates['15_usr_index']
    return_loss += self.template_rx_and_sx_implies_tx(last_statement,
                                                      has_tfidf,
                                                      current_state)

    return return_loss

  def rule_4(self, logits, **unused_kwargs) -> float:
    """Dialog structure rule.

      Rule:
        LastStatement(U1) & Has_Sys_TFIDF(U1, State) -> State(U1, State)

      Meaning:
        IF: the utterance is the last utterance in a dialog and it has
            a known tf-idf word.
        THEN: the utterance is likely to belong to a last state.

      Args:
        logits: logits outputted by a neural model.

      Returns:
        A loss incurred by this dialog structure rule.
      """
    last_statement = self.predicates['last_statement']

    current_state = self._get_tensor_column(logits, 5)
    has_tfidf = self.predicates['5_sys_index']
    return_loss = self.template_rx_and_sx_implies_tx(last_statement,
                                                     has_tfidf,
                                                     current_state)

    current_state = self._get_tensor_column(logits, 12)
    has_tfidf = self.predicates['12_sys_index']
    return_loss += self.template_rx_and_sx_implies_tx(last_statement,
                                                      has_tfidf,
                                                      current_state)

    current_state = self._get_tensor_column(logits, 15)
    has_tfidf = self.predicates['15_sys_index']
    return_loss += self.template_rx_and_sx_implies_tx(last_statement,
                                                      has_tfidf,
                                                      current_state)

    return return_loss

  def rule_5(self, logits, **unused_kwargs):
    """Dialog structure rule.

      Rule:
        PreviousStatement(U1, U2) & State(U2, PreviousState)
          & Has_Usr_TFIDF(U1, CurrentState) -> State(U1, CurrentState)

      Meaning:
        IF: the previous utterance belongs to a known state transition,
            and the current utterance has a known usr tfidf token for the
            transitioned state.
        THEN: the utterance is likely to belong to that state.

      Args:
        logits: logits outputted by a neural model.

      Returns:
        A loss incurred by this dialog structure rule.
      """
    previous_statement = self.predicates['previous_statement']
    return_loss = None
    for [current_state_index, previous_state_index] in self.state_transitions:
      previous_state = self._get_tensor_column(logits, previous_state_index)
      current_state = self._get_tensor_column(logits, current_state_index)
      has_tfidf = self.predicates[str(current_state_index) + '_usr_index']

      if return_loss is None:
        return_loss = self.template_rxy_and_sy_and_tx_implies_ux(
          previous_statement,
          previous_state,
          has_tfidf,
          current_state)
      else:
        return_loss += self.template_rxy_and_sy_and_tx_implies_ux(
          previous_statement,
          previous_state,
          has_tfidf,
          current_state)

    return return_loss

  def rule_6(self, logits, **unused_kwargs):
    """Dialog structure rule.

      Rule:
        PreviousStatement(U1, U2) & State(U2, PreviousState)
          & Has_Sys_TFIDF(U1, CurrentState) -> State(U1, CurrentState)

      Meaning:
        IF: the previous utterance belongs to a known state transition,
            and the current utterance has a known sys tfidf token for the
            transitioned state.
        THEN: the utterance is likely to belong to that state.

      Args:
        logits: logits outputted by a neural model.

      Returns:
        A loss incurred by this dialog structure rule.
      """
    previous_statement = self.predicates['previous_statement']
    return_loss = None
    for [current_state_index, previous_state_index] in self.state_transitions:
      previous_state = self._get_tensor_column(logits, previous_state_index)
      current_state = self._get_tensor_column(logits, current_state_index)
      has_tfidf = self.predicates[str(current_state_index) + '_sys_index']

      if return_loss is None:
        return_loss = self.template_rxy_and_sy_and_tx_implies_ux(
          previous_statement,
          previous_state,
          has_tfidf,
          current_state)
      else:
        return_loss += self.template_rxy_and_sy_and_tx_implies_ux(
          previous_statement,
          previous_state,
          has_tfidf,
          current_state)

    return return_loss

  def generate_predicates(self, data: tf.Tensor):
    """Generates potentials used throughout the rules."""
    self.predicates['first_statement'] = self._first_statement()
    self.predicates['last_statement'] = self._last_statement(data)
    self.predicates['previous_statement'] = self._previous_statement(data)

    for class_name in self.config['words']:
      usr_index = self.config['words'][class_name]['usr']['index']
      sys_index = self.config['words'][class_name]['sys']['index']
      self.predicates[class_name + '_usr_index'] = self._has_word(data,
                                                                  usr_index)
      self.predicates[class_name + '_sys_index'] = self._has_word(data,
                                                                  sys_index)

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

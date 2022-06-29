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

"""Abstract class for PSL constraints."""

import abc
from typing import List, Tuple

import tensorflow as tf

_EPSILON = 1e-8
_LINEAR_LOSS = 'linear'
_LOG_LOSS = 'log'


def normalize(val, should_normalize: bool = False):
  if not should_normalize:
    return val
  val_sum = tf.maximum(tf.reduce_sum(val), _EPSILON)
  return val / val_sum


class PSLModel(abc.ABC):
  """Abstract class for PSL constraints."""

  def __init__(self,
               rule_weights: List[float],
               rule_names: List[str],
               logic: str = 'lukasiewicz',
               loss_function: str = _LINEAR_LOSS,
               **kwargs) -> None:
    assert len(rule_weights) == len(
        rule_names), 'Rule weights and rule names must be the same length.'

    self.rule_functions = [getattr(self, rule_name) for rule_name in rule_names]
    self.rule_weights = rule_weights
    self.kwargs = kwargs
    self.logic = logic
    self.loss_function = loss_function

  def generate_predicates(self, data: tf.Tensor):
    """Generates potentials used throughout the rules."""
    pass

  @abc.abstractmethod
  def compute_loss(self, data: tf.Tensor, logits: tf.Tensor) -> float:
    pass

  @abc.abstractmethod
  def compute_loss_per_rule(self, data: tf.Tensor,
                            logits: tf.Tensor) -> List[float]:
    pass

  @staticmethod
  def compute_rule_loss(body: List[tf.Tensor],
                        head: tf.Tensor,
                        logic: str = 'lukasiewicz',
                        loss_function: str = _LINEAR_LOSS) -> float:
    """Calculates loss for a soft rule."""

    if loss_function == _LINEAR_LOSS:
      body, head = PSLModel.soft_imply(body, head, logic=logic)
      # body -> head = max(1 - SUM(1 - body_i) - head, 0)
      return tf.reduce_sum(tf.nn.relu(1. - tf.add_n(body) - head))
    else:
      truth_value = PSLModel.soft_imply2(body, head, logic=logic)
      prob = tf.clip_by_value(truth_value, _EPSILON, 1. - _EPSILON)
      return tf.reduce_mean(-tf.reduce_sum(tf.math.log(prob), axis=-1))

  # Keep it as a part of the implementation of the original loss function.
  @staticmethod
  def soft_imply(
      body: List[tf.Tensor],
      head: tf.Tensor,
      logic: str = 'lukasiewicz') -> Tuple[List[tf.Tensor], tf.Tensor]:
    """Soft logical implication."""
    if logic == 'lukasiewicz':
      # body -> head = (1 - body_i), head
      return [1. - predicate for predicate in body], head
    else:
      raise ValueError('Unsuported logic: %s' % (logic,))

  @staticmethod
  def soft_imply2(body: List[tf.Tensor],
                  head: tf.Tensor,
                  logic: str = 'lukasiewicz') -> tf.Tensor:
    """Soft logical implication."""
    if logic == 'lukasiewicz':
      body = PSLModel.soft_and(body)
      # A -> B = min(1, 1 - A + B)
      return tf.minimum(1., 1 - body + head)
    else:
      raise ValueError('Unsuported logic: %s' % (logic,))

  @staticmethod
  def soft_and(body: List[tf.Tensor], logic: str = 'lukasiewicz') -> tf.Tensor:
    """Soft logical implication."""
    if logic == 'lukasiewicz':
      num_bodies = len(body)
      if num_bodies == 0:
        return tf.constant(0)
      # A & B = max(0, A + B - 1)
      val = body[0]
      for i in range(1, num_bodies):
        val = normalize(tf.nn.relu(val + body[i] - 1))
      return val
    else:
      raise ValueError('Unsuported logic: %s' % (logic,))

  @staticmethod
  def soft_not(predicate: tf.Tensor, logic: str = 'lukasiewicz') -> tf.Tensor:
    """Soft logical negation."""
    if logic == 'lukasiewicz':
      # !predicate = 1 - predicate
      return normalize(1. - predicate)
    else:
      raise ValueError('Unsuported logic: %s' % (logic,))

  def template_rx_implies_sx(self, r_x: tf.Tensor, s_x: tf.Tensor) -> float:
    """Template for R(x) -> S(x).

    Args:
      r_x: a (batch_size, example_size) tensor.
      s_x: a (batch_size, example_size) tensor.

    Returns:
      A computed loss for this type of rule.
    """
    body = [r_x]
    head = s_x

    return PSLModel.compute_rule_loss(body, head, logic=self.logic)

  def template_rx_and_sx_implies_tx(self, r_x: tf.Tensor, s_x: tf.Tensor,
                                    t_x: tf.Tensor) -> float:
    """Template for R(x) & S(x) -> T(x).

    Args:
      r_x: a (batch_size, example_size) tensor.
      s_x: a (batch_size, example_size) tensor.
      t_x: a (batch_size, example_size) tensor.

    Returns:
      A computed loss for this type of rule.
    """
    body = [r_x, s_x]
    head = t_x

    return PSLModel.compute_rule_loss(body, head, logic=self.logic)

  def template_rxy_and_sy_implies_tx(self, r_xy: tf.Tensor, s_y: tf.Tensor,
                                     t_x: tf.Tensor) -> float:
    """Template for R(x,y) & S(y) -> T(x).

    Converts s_y and t_x into (batch_size, example_size, example_size) tensors,
    inverts t_x, and computes the rule loss.

    Args:
      r_xy: a (batch_size, example_size, example_size) tensor.
      s_y: a (batch_size, example_size) tensor.
      t_x: a (batch_size, example_size) tensor.

    Returns:
      A computed loss for this type of rule.
    """
    s_y_matrix = PSLModel._unary_to_binary(s_y, transpose=False)
    t_x_matrix = PSLModel._unary_to_binary(t_x, transpose=True)

    body = [r_xy, s_y_matrix]
    head = t_x_matrix

    return PSLModel.compute_rule_loss(body, head, logic=self.logic)

  def template_rxy_and_sy_and_tx_implies_ux(self, r_xy: tf.Tensor,
                                            s_y: tf.Tensor, t_x: tf.Tensor,
                                            u_x: tf.Tensor) -> float:
    """Template for R(x,y) & S(y) & T(x) -> U(x).

    Converts s_y, t_x, and u_x into (batch_size, example_size, example_size)
    tensors, inverts t_x and u_x, and computes the rule loss.

    Args:
      r_xy: a (batch_size, example_size, example_size) tensor.
      s_y: a (batch_size, example_size) tensor.
      t_x: a (batch_size, example_size) tensor.
      u_x: a (batch_size, example_size) tensor.

    Returns:
      A computed loss for this type of rule.
    """
    s_y_matrix = PSLModel._unary_to_binary(s_y, transpose=False)
    t_x_matrix = PSLModel._unary_to_binary(t_x, transpose=True)
    u_x_matrix = PSLModel._unary_to_binary(u_x, transpose=True)

    body = [r_xy, s_y_matrix, t_x_matrix]
    head = u_x_matrix

    return PSLModel.compute_rule_loss(body, head, logic=self.logic)

  def template_rxy_and_sy_and_tx_and_ux_implies_vx(self, r_xy: tf.Tensor,
                                                   s_y: tf.Tensor,
                                                   t_x: tf.Tensor,
                                                   u_x: tf.Tensor,
                                                   v_x: tf.Tensor) -> float:
    """Template for R(x,y) & S(y) & T(x) & U(x) -> V(x).

    Converts s_y, t_x, u_x, and v_x into (batch_size, example_size,
    example_size) tensors, inverts t_x, u_x, and v_x, and computes the rule
    loss.

    Args:
      r_xy: a (batch_size, example_size, example_size) tensor.
      s_y: a (batch_size, example_size) tensor.
      t_x: a (batch_size, example_size) tensor.
      u_x: a (batch_size, example_size) tensor.
      v_x: a (batch_size, example_size) tensor.

    Returns:
      A computed loss for this type of rule.
    """
    s_y_matrix = PSLModel._unary_to_binary(s_y, transpose=False)
    t_x_matrix = PSLModel._unary_to_binary(t_x, transpose=True)
    u_x_matrix = PSLModel._unary_to_binary(u_x, transpose=True)
    v_x_matrix = PSLModel._unary_to_binary(v_x, transpose=True)

    body = [r_xy, s_y_matrix, t_x_matrix, u_x_matrix]
    head = v_x_matrix

    return PSLModel.compute_rule_loss(body, head, logic=self.logic)

  @staticmethod
  def _unary_to_binary(predicate: tf.Tensor, transpose: bool) -> tf.Tensor:
    predicate_matrix = tf.repeat(predicate, predicate.shape[-1], axis=-1)
    predicate_matrix = tf.reshape(
        predicate_matrix, [-1, predicate.shape[-1], predicate.shape[-1]])
    if transpose:
      predicate_matrix = tf.transpose(predicate_matrix, perm=[0, 2, 1])
    return predicate_matrix

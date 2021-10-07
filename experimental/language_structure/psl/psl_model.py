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

"""Abstract class for PSL constraints."""

import abc
from typing import List, Tuple

import tensorflow as tf


class PSLModel(abc.ABC):
  """Abstract class for PSL constraints."""

  def __init__(self, rule_weights: List[float], rule_names: List[str],
               **kwargs) -> None:
    assert len(rule_weights) == len(
        rule_names), 'Rule weights and rule names must be the same length.'

    self.rule_functions = [getattr(self, rule_name) for rule_name in rule_names]
    self.rule_weights = rule_weights
    self.kwargs = kwargs

  @abc.abstractmethod
  def compute_loss(self, data: tf.Tensor, logits: tf.Tensor) -> float:
    pass

  @staticmethod
  def compute_rule_loss(body: List[tf.Tensor],
                        head: tf.Tensor,
                        logic: str = 'lukasiewicz') -> float:
    """Calculates loss for a soft rule."""
    body, head = PSLModel.soft_imply(body, head, logic=logic)

    if logic == 'lukasiewicz':
      # body -> head = max(1 - SUM(1 - body_i) - head, 0)
      return tf.reduce_sum(tf.nn.relu(1. - tf.add_n(body) - head))
    else:
      raise ValueError('Unsuported logic: %s' % (logic,))

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
  def soft_not(predicate: tf.Tensor, logic: str = 'lukasiewicz') -> tf.Tensor:
    """Soft logical negation."""
    if logic == 'lukasiewicz':
      # !predicate = 1 - predicate
      return 1. - predicate
    else:
      raise ValueError('Unsuported logic: %s' % (logic,))

  @staticmethod
  def template_rx_implies_sx(r_x: tf.Tensor,
                             s_x: tf.Tensor,
                             logic: str = 'lukasiewicz') -> float:
    """Template for R(x) -> S(x).

    Args:
      r_x: a (batch_size, example_size) tensor.
      s_x: a (batch_size, example_size) tensor.
      logic: the type of logic being used.

    Returns:
      A computed loss for this type of rule.
    """
    body = [r_x]
    head = s_x

    return PSLModel.compute_rule_loss(body, head, logic=logic)

  @staticmethod
  def template_rx_and_sx_implies_tx(r_x: tf.Tensor,
                                    s_x: tf.Tensor,
                                    t_x: tf.Tensor,
                                    logic: str = 'lukasiewicz') -> float:
    """Template for R(x) & S(x) -> T(x).

    Args:
      r_x: a (batch_size, example_size) tensor.
      s_x: a (batch_size, example_size) tensor.
      t_x: a (batch_size, example_size) tensor.
      logic: the type of logic being used.

    Returns:
      A computed loss for this type of rule.
    """
    body = [r_x, s_x]
    head = t_x

    return PSLModel.compute_rule_loss(body, head, logic=logic)

  @staticmethod
  def template_rxy_and_sy_implies_tx(r_xy: tf.Tensor,
                                     s_y: tf.Tensor,
                                     t_x: tf.Tensor,
                                     logic: str = 'lukasiewicz') -> float:
    """Template for R(x,y) & S(y) -> T(x).

    Converts s_y and t_x into (batch_size, example_size, example_size) tensors,
    inverts t_x, and computes the rule loss.

    Args:
      r_xy: a (batch_size, example_size, example_size) tensor.
      s_y: a (batch_size, example_size) tensor.
      t_x: a (batch_size, example_size) tensor.
      logic: the type of logic being used.

    Returns:
      A computed loss for this type of rule.
    """
    s_y_matrix = PSLModel._unary_to_binary(s_y, transpose=False)
    t_x_matrix = PSLModel._unary_to_binary(t_x, transpose=True)

    body = [r_xy, s_y_matrix]
    head = t_x_matrix

    return PSLModel.compute_rule_loss(body, head, logic=logic)

  @staticmethod
  def _unary_to_binary(predicate: tf.Tensor, transpose: bool) -> tf.Tensor:
    predicate_matrix = tf.repeat(predicate, predicate.shape[-1], axis=-1)
    predicate_matrix = tf.reshape(
        predicate_matrix, [-1, predicate.shape[-1], predicate.shape[-1]])
    if transpose:
      predicate_matrix = tf.transpose(predicate_matrix, perm=[0, 2, 1])
    return predicate_matrix

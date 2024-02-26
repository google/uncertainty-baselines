# coding=utf-8
# Copyright 2024 The Uncertainty Baselines Authors.
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

"""Tree utils.

On the surface, the structures encoded in PENMAN notation are a tree,
and only by resolving repeated node index (variables) as reentrancies
does the actual graph become accessible.
"""
from typing import Tuple, List, Any
# Node indexes.
Variable = str
# Relation/attribute names.
Role = str
# Node context branch.
Branch = Tuple[Role, Any]
# Node contexts for constructing a tree structure.
Node = Tuple[Variable, List[Branch]]


class Tree(object):
  """Represents a tree example."""

  def __init__(self, node: Node):
    self.node = node

  def nodes(self) -> List[Node]:
    """Returns the nodes in the tree as a flat list."""
    return _nodes(self.node)

  def format(self, indent: int = 2, new_line: bool = True):
    """Formats the tree structure into a PENMAN string.

    Example:
      >>> tree = Tree(
              'x0', [('/', 'unknown'), (':ARG', ('x1', [('/', '_look_v_up')]))])
      >>> print(tree.format())
      (x0 / unknown
        :ARG (x1 / _look_v_up))
    Args:
      indent: How to indent formatted strings.
      new_line: Whether to have new line for each node context.
    Returns:
      The PENMAN-serialized string of the tree.
    """
    if indent < 0 or not new_line: indent = 0
    node_indexes = set([idx for idx, _ in self.nodes()])
    parts = [_format_node(self.node, indent, new_line, 0, node_indexes)]
    if new_line:
      return '\n'.join(parts)
    else:
      return ' '.join(parts)


def is_atomic(x) -> bool:
  """Returns ``True`` if *x* is a valid atomic value.

  Examples:
    >>> is_atomic('a')
    True
    >>> is_atomic(None)
    True
    >>> is_atomic(3.14)
    True
    >>> is_atomic(('a', [('/', 'alpha')]))
    False

  Args:
    x: input string.

  Returns:
    True if input string is a valid atomic value.
  """
  return x is None or isinstance(x, (str, int, float))


def _nodes(node: Node) -> List[Node]:
  idx, contexts = node
  ns = [] if idx is None else [node]
  for _, target in contexts:
    # If target is not atomic, assume it's a valid tree node
    if not is_atomic(target):
      ns.extend(_nodes(target))
  return ns


def _format_node(node: Node,
                 indent: int,
                 new_line: bool,
                 column: int,
                 node_indexes) -> str:
  """Formats node into a PENMAN string."""
  idx, contexts = node
  if not idx:
    return '()'  # Empty node.
  if not contexts:
    return f'({idx!s})'  # Index-only node.

  # Determines appropriate joiner based on value of indent.
  column += indent
  joiner = '\n' + ' ' * column if new_line else ' '

  # Formats the contexts and join them.
  # If index is non-empty, all initial attributes are compactly
  # joined on the same line, otherwise they use joiner.
  parts: List[str] = []
  compact = bool(node_indexes)
  for context in contexts:
    target = context[1]
    if compact and (not is_atomic(target) or target in node_indexes):
      compact = False
      if parts:
        parts = [' '.join(parts)]
    parts.append(
        _format_context(context, indent, new_line, column, node_indexes))
  # Checks if all contexts can be compactly written.
  if compact:
    parts = [' '.join(parts)]

  return f'({idx!s} {joiner.join(parts)})'


def _format_context(context, indent, new_line, column, node_indexes):
  """Formats node context into a PENMAN string."""
  role, target = context

  if role != '/' and not role.startswith(':'):
    role = ':' + role

  sep = ' '
  if not target:
    target = sep = ''
  elif not is_atomic(target):
    target = _format_node(target, indent, new_line, column, node_indexes)

  return f'{role}{sep}{target!s}'

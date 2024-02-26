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

"""Graph linearization utils.

Encodes the graph triples into a tree structure, and formats
the tree structure into a PENMAN anotation.
Specifically, going from a graph triples to a penman string is called
*encoding*, going from a graph triples to a tree is called *configuration*,
and going from a tree to a penman string is called *formatting*.
"""
from typing import Union, List, Tuple, Dict, Any, cast
from data.deepbank import tree_utils  # local file import from baselines.t5

# Node indexes.
Variable = str
# Relation/attribute names.
Role = str
# Node context branch.
Branch = Tuple[Role, Any]
# Node contexts for constructing a tree structure.
Node = Tuple[Variable, List[Branch]]
# Node index to node contexts map.
NodeMap = Dict[Variable, Union[Node, None]]
# A graph triple.
Triple = Tuple[str, str, str]
# A list of triples from the graph.
Triples = List[Tuple[str, str, str]]


def encode(triples: Triples,
           root: Variable,
           indent: int = 2,
           new_line: bool = True) -> str:
  """Serializes the triples from the graph into PENMAN string.

  Example:
    >>> encode([('x0', 'instance', 'unknown')])
    '(x0 / unknown)'

  Args:
    triples: A list of triples from the graph, e.g.,
      [('x0', 'instance', 'unknown'), ('x1', 'instance', '_look_v_up'),
       ('x0', 'ARG', 'x1')].
    root: The node index of root in the serialization, which should
      be defined in the triples.
    indent: How to indent formatted strings.
    new_line: Whether to have new line for each node context.
  Returns:
    The PENMAN-serialized string of the graph.
  """
  tree = get_tree_from_graph(triples, root)
  return tree.format(indent, new_line)


def get_tree_from_graph(triples: Triples, root: Variable) -> tree_utils.Tree:
  """Creates a tree object from a graph.

  Example:
    >>> t = get_tree_from_graph([('x0', ':instance', 'unknown'),
    ...                          ('x1', ':instance', '_look_v_up'),
    ...                          ('x0', ':ARG', 'x1')], root='x0')
    >>> print(t.node)
    ('x0', [('/', 'unknown'), (':ARG', ('x1', [('/', '_look_v_up')]))])

  Args:
    triples: A list of triples from the graph, e.g.,
      [('x0', 'instance', 'unknown'), ('x1', 'instance', '_look_v_up'),
       ('x0', 'ARG', 'x1')].
    root: The node index of root in the serialization, which should
      be defined in the triples.

  Returns:
    The configured tree object.
  """
  tree = _configure(triples, root)
  return tree


def _configure(triples: Triples, root: str) -> tree_utils.Tree:
  """Configures the tree structure from the graph triples."""
  if not triples:
    raise ValueError('Gets empty graph triples.')
  node_indexes = set(src for src, _, _ in triples)
  if root not in node_indexes:
    raise ValueError(f'Root is not a node index: {root!r}')
  # The map from node index to node contexts, where the contexts include
  # node definition and related edge infos. E.g,
  # for triples [('x0', ':instance', 'unknown'),
  #              ('x1', ':instance', '_look_v_up'),
  #              ('x0', ':ARG', 'x1')]
  # the nodemap is:
  # {'x0': ('x0', [('/', 'unknown'),
  #                (':ARG', ('x1', [('/', '_look_v_up')])),
  #                (':BV-of', ('x2', [('/', '_udef_q')]))]),
  #  'x1': ('x1', [('/', '_look_v_up')]),
  #  'x2': ('x2', [('/', '_udef_q')])}
  # Initialization of nodemap.
  nodemap = {idx: None for idx in node_indexes}
  nodemap[root] = (root, [])
  data = list(reversed(triples))
  # Initialization of node.
  node = _configure_node(root, data, nodemap)
  while data:
    skipped, idx, data = _find_next(data, nodemap)
    data_count = len(data)
    if idx is None or data_count == 0:
      raise ValueError('Possibly disconnected graph')
    _configure_node(idx, data, nodemap)
    if len(data) >= data_count:
      raise ValueError('Possible cycle in configuration')
    data = skipped + data
  tree = tree_utils.Tree(node)
  return tree


def _find_next(data: Triples, nodemap: NodeMap):
  """Finds the next node contexts; establishes contexts if necessary."""
  idx = None
  for i in range(len(data) - 1, -1, -1):
    source, _, target = data[i]
    if source in nodemap and _get_or_establish_node_contexts(source, nodemap):
      idx = source
      break
    elif target in nodemap and _get_or_establish_node_contexts(target, nodemap):
      idx = target
      break
  pivot = i + 1
  return data[pivot:], idx, data[:pivot]


def invert_role(role: Role) -> Role:
  """Inverts role."""
  if role.endswith('-of'):
    inverse = role[:-3]
  else:
    inverse = role + '-of'
  return inverse


def invert(triple: Triple) -> Tuple[Variable, Role, Variable]:
  """Inverts graph triple.

  This will invert or deinvert a triple regardless of its
  current state, e.g., ('x0', 'ARG', 'x1') -> ('x0', 'ARG-of', 'x1'),
  or ('x0', 'ARG-of', 'x1') -> ('x0', 'ARG', 'x1').

  Args:
    triple: A graph triple, can be a node, attribute, or edge in the graph.

  Returns:
    target: The end index of the triple.
    inversed_role: The inversed role.
    source: The start index of the triple.
  """
  source, role, target = triple
  inversed_role = invert_role(role)
  # Casting is just for the benefit of the type checker; it does
  # not actually check that target is a valid variable type.
  target = cast(Variable, target)
  return target, inversed_role, source


def _configure_node(idx: Variable, data: Triples, nodemap: NodeMap) -> Node:
  """Configures a node and any descendants of this node.

  Note that `data` and `nodemap` will be modified here.

  Args:
    idx: The current node index.
    data: Triple data.
    nodemap: The map from node index to node contexts.

  Returns:
    node: Node.
  """
  node = nodemap[idx]
  contexts = node[1]

  while data:
    triple = data.pop()
    if triple[0] == idx:
      _, role, target = triple
    elif triple[2] == idx:
      _, role, target = invert(triple)
    else:
      # Misplaced triple.
      data.append(triple)
      break

    if role == ':instance':
      if not target:
        # Prefers '(a)' over '(a /)' when concept is missing.
        continue
      role = '/'
      index = 0
    else:
      index = len(contexts)

    if role != '/' and target in nodemap and nodemap[target] is None:
      # Site of potential node context.
      nodemap[target] = node

    contexts.insert(index, (role, target))

  return node


def _get_or_establish_node_contexts(idx, nodemap):
  """Turns a node index target into node contexts."""
  # First checks if the index is available at all.
  if nodemap[idx] is not None:
    idx_, contexts = nodemap[idx]
    # If the mapped node index doesn't match it can be established.
    if idx != idx_:
      node = (idx, [])
      nodemap[idx] = node
      for i in range(len(contexts)):
        # Replaces the node index in the tree with the new node.
        if contexts[i][1] == idx and contexts[i][0] != '/':
          context = list(contexts[i])
          context[1] = node
          contexts[i] = tuple(context)
          break
    else:
      pass  # Otherwise the node already exists so we're good.
    return True
  # Index is not yet available.
  return False

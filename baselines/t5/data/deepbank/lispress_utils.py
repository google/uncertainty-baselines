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

"""Utility functions for parsing lispress expressions in SMCalflow."""
from typing import List, Union
from data import metrics_utils  # local file import from baselines.t5


LEFT_PAREN = '('
RIGHT_PAREN = ')'
ESCAPE = '\\'
DOUBLE_QUOTE = '"'
META = '^'
READER = '#'
Sexp = Union[str, List['Sexp']]
Lispress = Sexp
# For rewriting the special tokens to avoid failure in parsing to DAGs.
SPECIAL_TOKEN_MAP = {
    '#': '_reader', '+': '_add', '-': '_minus', '==': '_equal',
    '>': '_greater', '<': '_less', '>=': '_g_eq', '<=': '_l_equal',
    '?=': '_q_equal', '?~=': '_q_t_qual', '?>': '_q_greater',
    '?<': '_q_less', '?>=': '_q_g_equal', '?<=': '_q_l_equal',
    '[]': '_empty_list', 'x0': '_refer_x0', 'x1': '_refer_x1',
    'x2': '_refer_x2', 'x3': '_refer_x3', 'x4': '_refer_x4',
    'x5': '_refer_x5', 'x6': '_refer_x6'}


def _is_beginning_control_char(next_c: str):
  return (next_c.isspace() or next_c == LEFT_PAREN or next_c == RIGHT_PAREN or
          next_c == DOUBLE_QUOTE or next_c == READER or next_c == META)


def parse_lispress(s: str) -> Lispress:
  r"""Parses a Lispress string into a Lispress object.

  E.g.,
  >>> s = \
  "(describe" \
  "  (:start" \
  "    (findNextEvent" \
  "      (Constraint[Event]" \
  "        :attendees (attendeeListHasRecipientConstraint" \
  "          (recipientWithNameLike" \
  "            (Constraint[Recipient])" \
  "            #(PersonName "Elaine")))))))"
  >>> parse_lispress(s)
  ['describe', [':start', ['findNextEvent', ['Constraint[Event]', ':attendees',
    ['attendeeListHasRecipientConstraint', ['recipientWithNameLike',
      ['Constraint[Recipient]'], '#', ['PersonName', '"Elaine"']]]]]]]

  Args:
    s: Input lispress string.

  Returns:
    A lispress object.
  """
  offset = 0

  # eoi = end of input
  def is_eoi():
    nonlocal offset
    return offset == len(s)

  def peek():
    nonlocal offset
    return s[offset]

  def next_char():
    nonlocal offset
    cn = s[offset]
    offset += 1
    return cn

  def skip_whitespace():
    while (not is_eoi()) and peek().isspace():
      next_char()

  def skip_then_peek():
    skip_whitespace()
    return peek()

  def read() -> Sexp:
    skip_whitespace()
    c = next_char()
    if c == LEFT_PAREN:
      return read_list()
    elif c == DOUBLE_QUOTE:
      return read_string()
    elif c == META:
      meta = read()
      expr = read()
      return [META, meta, expr]
    elif c == READER:
      return [READER, read()]
    else:
      out_inner = ''
      if c != '\\':
        out_inner += c
      if not is_eoi():
        next_c = peek()
        escaped = c == '\\'
        while (not is_eoi()) and (escaped or
                                  not _is_beginning_control_char(next_c)):
          if (not escaped) and next_c == '\\':
            next_char()
            escaped = True
          else:
            out_inner += next_char()
            escaped = False
          if not is_eoi():
            next_c = peek()
      return out_inner

  def read_list():
    out_list = []
    while skip_then_peek() != RIGHT_PAREN:
      out_list.append(read())
    next_char()
    return out_list

  def read_string():
    out_str = ''
    while peek() != '"':
      c_string = next_char()
      out_str += c_string
      if c_string == '\\':
        out_str += next_char()
    next_char()
    return f'"{out_str}"'

  out = read()
  skip_whitespace()
  assert offset == len(
      s
  ), f'Failed to exhaustively parse {s}, maybe you are missing a close paren?'
  return out


def pre_processing_graph_str(graph_str: str) -> str:
  """Pre-processing for graph string to avoid failure in parsing to DAGs."""
  graph_str_list = []
  for token in graph_str.split():
    if token in SPECIAL_TOKEN_MAP:
      token = SPECIAL_TOKEN_MAP[token]
    graph_str_list.append(token)
  return ' '.join(graph_str_list)


def lispress_to_graph(sexp: Sexp) -> str:
  """Transfers the Lispress object into the variable-free penman graph string."""
  if len(sexp) == 1:
    # Input example: ['Constraint[Recipient]'].
    # Output example: ' ( Constraint[Recipient] ) '.
    return ' ( ' + sexp[0] + ' ) '
  elif len(sexp) == 2:
    if isinstance(sexp[1], str):
      if sexp[0] == 'Number':
        # Input example: ['Number', '0'].
        # Output example: ' ( Number :carg " 0 " ) '.
        return ' ( Number :carg " %s " ) ' % sexp[1]
      elif sexp[0] in ['Boolean', 'RespondShouldSend']:
        # Input example: ['Boolean', 'true'].
        # Output example: ' ( Boolean :carg " true " ) '.
        return ' ( %s :carg " %s " ) ' % (sexp[0], sexp[1])
      elif sexp[0] in ['List[Path]', 'List[Any]']:
        # Input example: ['List[Path]', '[]'].
        # Output example: ' ( List[Path] :carg " [] " ) '.
        return ' ( %s :carg " %s " ) ' % (sexp[0], sexp[1])
      elif sexp[1].startswith('"'):
        # Input example: ['String', '" math lecture "'].
        # Output example: '( String :carg " math@-@lecture " )'.
        attribute_value = sexp[1].replace(
            '\\"', '<quote>').replace('" ', '').replace(' "', '')
        return ' ( %s :carg " %s " ) ' % (sexp[0], '@-@'.join(
            attribute_value.split()))
      elif sexp[0] in metrics_utils.SMCALFLOW_ARG_EDGES:
        # Input example: [':item', 'x0'].
        # Output example: ' ( NoneNode :item ( x0 ) ) '.
        return ' ( NoneNode %s ( %s ) ) ' % (sexp[0], sexp[1])
      else:
        # Input example: ['extensionConstraint', 'x0'].
        # Output example: ' ( extensionConstraint :NoneARG ( x0 ) ) '.
        return ' ( %s :NoneARG ( %s ) ) ' % (sexp[0], sexp[1])
    else:
      if sexp[0] in metrics_utils.SMCALFLOW_ARG_EDGES:
        # Input example: [':results', List].
        # Output example: ' ( NoneNode :results%s) ' % lispress_to_graph(List).
        return ' ( NoneNode %s%s) ' % (sexp[0], lispress_to_graph(sexp[1]))
      else:
        # Input example: ['size', List].
        # Output example: ' ( size :NoneARG%s) ' % lispress_to_graph(List).
        # Case 2: ['size', List]
        return ' ( %s :NoneARG%s) ' % (sexp[0], lispress_to_graph(sexp[1]))
  elif len(sexp) == 3:
    if isinstance(sexp[1], str):
      if sexp[1].startswith(':'):
        if isinstance(sexp[2], list):
          # Input example: ['Yield', ':output', List].
          # Output example: ' ( Yield :output%s) ' % lispress_to_graph(List).
          return ' ( %s %s%s) ' % (sexp[0], sexp[1], lispress_to_graph(sexp[2]))
        else:
          # Input example: ['Yield', ':output', 'x0'].
          # Output example: ' ( Yield :output ( x0 ) ) '.
          return ' ( %s %s ( %s ) ) ' % (sexp[0], sexp[1], sexp[2])
      else:
        if sexp[2] in metrics_utils.SMCALFLOW_REFERENCE_NODES:
          # Input example: ['nextDayOfWeek', 'x1', 'x0'].
          # Output example: ' ( nextDayOfWeek :NoneARG ( x1 ) '
          #                 ':NoneARG ( x0 ) ) '.
          return ' ( %s :NoneARG ( %s ) :NoneARG ( %s ) ) ' % (
              sexp[0], sexp[1], sexp[2])
        else:
          # Input example: ['andConstraint', 'x0', List].
          # Output example: ' ( andConstraint :NoneARG ( x0 ) :NoneARG%s) ' % (
          #                     lispress_to_graph(List)).
          return ' ( %s :NoneARG ( %s ) :NoneARG%s) ' % (
              sexp[0], sexp[1], lispress_to_graph(sexp[2]))
    else:
      if sexp[2] in metrics_utils.SMCALFLOW_REFERENCE_NODES:
        # Input example: ['previousDayOfWeek', List, 'x0'].
        # Output example: ' ( previousDayOfWeek :NoneARG%s:NoneARG ( x0 ) ) ' %
        #                 (lispress_to_graph(List)).
        return ' ( %s :NoneARG%s:NoneARG ( %s ) ) ' % (
            sexp[0], lispress_to_graph(sexp[1]), sexp[2])
      else:
        # Input example: ['let', List1, List2].
        # Output example: ' ( let :NoneARG%s:NoneARG%s) ' % (
        #                   lispress_to_graph(List1), lispress_to_graph(List2)).
        return ' ( %s :NoneARG%s:NoneARG%s) ' % (sexp[0], lispress_to_graph(
            sexp[1]), lispress_to_graph(sexp[2]))
  else:
    if sexp[0] in metrics_utils.SMCALFLOW_REFERENCE_NODES:
      # Input example: ['x0', List1, 'x1', List2, ...].
      # Output example: ' ( NoneNode :NoneARG ( x0 :NoneARG%s) '
      #                 ':NoneARG ( x1 :NoneARG%s) ) ' % (
      #                     lispress_to_graph(List1), lispress_to_graph(List2)).
      graph_str = ' ( NoneNode '
      for i, j in zip(range(0, len(sexp), 2), range(1, len(sexp), 2)):
        reference_node = sexp[i]
        reference_list = sexp[j]
        graph_str += ':NoneARG ( %s :NoneARG%s) ' % (
            reference_node, lispress_to_graph(reference_list))
      graph_str += ') '
      return graph_str
    elif isinstance(sexp[1],
                    list) or sexp[1] in metrics_utils.SMCALFLOW_REFERENCE_NODES:
      # Input example: [NODE, List1, List2, x3, ...].
      # Output example: ' ( Node :NoneARG lispress_to_graph(List1) '
      #                 ':NoneARG lispress_to_graph(List2) '
      #                 ':NoneARG ( x3 ) ... ) '.
      graph_str = ' ( %s ' %  sexp[0]
      for i in range(1, len(sexp)):
        if isinstance(sexp[i], list):
          graph_str += ':NoneARG%s' % lispress_to_graph(sexp[i])
        else:
          graph_str += ':NoneARG ( %s ) ' % (sexp[i])
      graph_str += ') '
      return graph_str
    else:
      # Input example: [NODE, :ARG1, List, :ARG2, REFERENCE_NODE, ...].
      # Output example: ' ( NODE :ARG1 lispress_to_graph(List) '
      #                 ':ARG2 ( REFERENCE_NODE )) '.
      graph_str = ' ( %s ' % sexp[0]
      for i, j in zip(range(1, len(sexp), 2), range(2, len(sexp), 2)):
        arg_edge = sexp[i]
        arg_node = sexp[j]
        if isinstance(arg_node, str):
          # [:ARG, REFERENCE_NODE] -> :ARG ( REFERENCE_NODE )
          graph_str += '%s ( %s ) ' % (arg_edge, arg_node)
        else:
          # [:ARG, List] -> :ARG lispress_to_graph(List)
          graph_str += '%s%s' % (arg_edge, lispress_to_graph(arg_node))
      graph_str += ') '
      return graph_str


def retok_graph_str(graph_str: str, data_name: str = 'smcalflow') -> str:
  """Retokenizes graph string to special tokens."""
  new_graph_str_list = []
  name_to_token_map = metrics_utils.NAME_TO_TOKEN_MAPS[data_name]
  for name in graph_str.split():
    if name in name_to_token_map:
      name = name_to_token_map[name]
    new_graph_str_list.append(name)
  return ' '.join(new_graph_str_list)

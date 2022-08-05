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

"""Utility functions for processing PENMAN string (linearization of graph).

For graph semantic parsing, the input to seq2seq model is a serialization
format for DAGs. Specifically, we use PENMAN notation here. It looks similar to
Lisp’s S-Expressions in using parentheses to indicate nested structures.
"""

import collections
import re
from typing import Any, Dict, List, Text

from absl import logging
import numpy as np
from data import metrics_utils  # local file import from baselines.t5
from data.deepbank import graph_utils  # local file import from baselines.t5
from data.deepbank import lispress_utils  # local file import from baselines.t5


class PENMANStr(object):
  """Represents a penman string example."""

  def __init__(self,
               graph_str: Text,
               variable_free: bool = False,
               retokenized: bool = False,
               data_version: Text = 'v0'):
    """Initialization of the penman string.

    There are three versions of penman string.
    - The original penman string with variables, e.g., '( x0 / unknown )'.
    - The variable-free panman string, which removes the variable
      identifiers (e.g., 'x0'), e.g., '( unknown )'.
    - The retokenized variable-free penman string, which retokenizes the
      tokens in the variable-free penman string to make some of the token
      non-tokenizable and thus can save the sequence length.

    Args:
      graph_str: the input graph string.
      variable_free: whether the input graph string is a variable free
        penman string.
      retokenized: whether the input graph string has been retokenized.
        Note that for the orignal penman this argument does not matter since
        we will never retokenize the orignal penman string.
      data_version: the DeepBank data version.
    """
    if variable_free:
      if retokenized:
        self.retokened_variable_free_penman = graph_str
        self.variable_free_penman = reverse_tokened_graph_str(
            self.retokened_variable_free_penman, data_version)
      else:
        self.variable_free_penman = graph_str
        self.retokened_variable_free_penman = retoken_graph_str(
            self.variable_free_penman, data_version)
      self.penman = transfer_to_penman(self.variable_free_penman)
    else:
      self.penman = graph_str
      self.variable_free_penman = transfer_to_variable_free_penman(self.penman)
      self.retokened_variable_free_penman = retoken_graph_str(
          self.variable_free_penman, data_version)


def transfer_to_variable_free_penman(penman_str: Text) -> Text:
  """Tranfers penman graph strings to variable-free penman strings.

  This function mainly addresses the rename of reentrancies (node reference),
  removes node variables, and adds necessary whitespaces near brackets,
  stars, and double quotes. E.g,

  '(x0/unknown) ... :ARG x0' -> '( unknown * ) ... :ARG ( unknown * )'

  Args:
    penman_str: the input penman string.

  Returns:
    variable-free penman string.
  """
  dag = graph_utils.parse_string_to_dag(penman_str)
  instances, _, _ = dag.get_triples()
  node_dict = graph_utils.transfer_triple_to_dict(instances, 'node')
  # Pattern for detecting reentrancies (node reference).
  pattern = re.compile(r' [e|x|i][0-9]+')
  reference_idxs = [x.lstrip() for x in re.findall(pattern, penman_str)]
  reference_values = [node_dict[x] for x in reference_idxs]
  # Assign new value to reentrancies by adding stars.
  reference_new_value_dict = {}
  # `value_counts` is for counting number of reference per node value.
  # There can be more than one reference that has the same node value.
  # We use different number of stars to specify this.
  # Input example: (x0/unknown) ... :ARG x0 ... (x1/unknown) ... :ARG x1
  # Output example: ( unknown * ) ... :ARG ( unknown * ) ...
  #                   ( unknown ** ) ... :ARG (unknown ** )
  value_counts = collections.defaultdict(int)
  for idx, value in zip(reference_idxs, reference_values):
    if idx not in reference_new_value_dict:
      value_counts[idx] += 1
      reference_new_value_dict[idx] = '%s %s' % (value, '*' * value_counts[idx])
  # Adds whitespaces near brackets, stars, and double quotes.
  variable_free_penman_str = penman_str
  variable_free_penman_str = variable_free_penman_str.replace(')', ' )')
  for idx, new_value in reference_new_value_dict.items():
    variable_free_penman_str = variable_free_penman_str.replace(
        '%s / %s' % (idx, node_dict[idx]), '%s / %s' % (idx, new_value))
    variable_free_penman_str = variable_free_penman_str.replace(
        ' %s ' % idx, ' (%s) ' % new_value)
  variable_free_penman_str = variable_free_penman_str.replace('(', '( ')
  variable_free_penman_str = variable_free_penman_str.replace('*)', '* )')
  variable_free_penman_str = re.sub(r'"([\S]*)"', r'" \1 "',
                                    variable_free_penman_str)
  pattern = re.compile(r'[e|x|i|_][0-9]+ / ')
  variable_free_penman_str = re.sub(pattern, '', variable_free_penman_str)
  return variable_free_penman_str


def _post_processing(graph_str: Text) -> Text:
  """Post-processing for variable-free penman strings."""
  # Merges the quote to the value of attributes.
  # Example: " John " to "John".
  graph_str = re.sub(r'" ([\S]*) "', r'"\1"', graph_str)
  # Handles peculiar tokens generated by the model, e.g., " ⁇ ".
  graph_str = graph_str.replace(' ⁇ ', '⁇')
  graph_str = graph_str.replace(' *', '*')

  if graph_str.split()[-1][0] in ['(', ':']:
    # The graph is incomplete, i.e., end with a left bracket or edge.
    # Example: '( unknown :ARG' or '( unknown :ARG ('.
    last_right_bracket_index = graph_str.rfind(')')
    graph_str = graph_str[:last_right_bracket_index + 1]

  # The number of left/right bracket is for matching the brackets.
  num_left_bracket, num_right_bracket = 0, 0
  # The `quote_count` is for check if current token is in quote
  # (attribute value). The bracket in quote does not count towards
  # total number of brackets.
  num_quote = 0
  new_graph_str = ''
  for x in graph_str:
    new_graph_str += x
    if x == '"':
      num_quote += 1
    if x == '(' and num_quote % 2 == 0:
      num_left_bracket += 1
    if x == ')' and num_quote % 2 == 0:
      num_right_bracket += 1
    if num_right_bracket == num_left_bracket:
      # If the number of right bracket has reached the number of
      # left brackets, the rest of the graph become illegal and
      # we just drop it.
      break
  if num_left_bracket > num_right_bracket:
    # After going through the whole graph string, if the number of left
    # brackets is greater than the number of right brackets,
    # we need to match the number of left brackets.
    new_graph_str += ' )' * (num_left_bracket - num_right_bracket)
  graph_str = new_graph_str
  return graph_str


def transfer_to_penman(graph_str: Text) -> Text:
  """Tranfers variable-free penman strings to the original penman strings.

  Args:
    graph_str: variable-free linearized graph, e.g.,
      "( unknown :ARG ( _book_n_1 ) )".

  Returns:
    penman_graph_str: e.g., "( x0 / unknown :ARG ( x1 / _book_n_1 ) )".
  """
  graph_str = _post_processing(graph_str)
  graph_str_list = []
  node_dict = {}
  count = 0
  for i, x in enumerate(graph_str.split()):
    if x[0] not in ['(', ')', ':', '"']:
      # x here is a node.
      if '*' in x:
        # Address coreference.
        # Example: replace 'unknown*' to 'x0' if previously
        # we defined '( x0 / unknown* )'.
        # There are two different versions of inputs,
        # [1] Without probabilities, e.g., unknown**.
        # [2] With probabilities, e.g., unknown**_1.0.
        # Here we need retrieve the node name 'unknown*'.
        last_star_index = x.rfind('*')
        node_name = x[:last_star_index + 1]
        if node_name not in node_dict:
          # The node name has not been defined previously.
          node_id = 'x' + str(count)
          node_dict[node_name] = node_id
          graph_str_list.append(node_id + ' / ' + x.replace('*', ''))
          count += 1
        else:
          # The node name has been defined previously, replace the
          # node name to its index.
          # Example '( unknown* )' -> 'x0'.
          graph_str_list.append(node_dict[node_name])
      else:
        graph_str_list.append('x' + str(count) + ' / ' + x)
        count += 1
    else:
      graph_str_list.append(x)
  graph_str = ' '.join(graph_str_list)

  # Addresses the duplicate coreference bracket issues.
  # Example: :ARG1 ( x0 ) -> :ARG1 x0.
  for _, v in node_dict.items():
    graph_str = graph_str.replace('( %s )' % v, v)

  # Addresses the duplicate coreference bracket issues.
  # Example: :ARG1 ( x0 :BV-of ( ... ) ) -> :ARG1 x0 :BV-of ( ... ).
  for _, v in node_dict.items():
    while '( %s :' % v in graph_str:
      index_left_bracket = graph_str.index('( %s :' % v)
      num_left_bracket, num_right_bracket = 0, 0
      for i in range(index_left_bracket, len(graph_str)):
        if graph_str[i] == '(':
          num_left_bracket += 1
        if graph_str[i] == ')':
          num_right_bracket += 1
        if num_left_bracket == num_right_bracket:
          # Removes the duplicate left bracket.
          graph_str = graph_str[:index_left_bracket] + graph_str[
              index_left_bracket + 2:]
          # Removes the duplicate right bracket.
          graph_str = graph_str[:i-2] + graph_str[i:]
          break
  return graph_str


def retoken_graph_str(graph_str: Text, data_version: str = 'v0') -> Text:
  """Retokenizes the graph string using custom tokenization."""
  new_graph_str_list = []
  name_to_token_maps_ = metrics_utils.NAME_TO_TOKEN_MAPS[data_version]
  for token in graph_str.split():
    if token.startswith(':'):
      # The token here is an edge.
      edge_name = token.split('-of')[0]
      retoken = token.replace(edge_name, name_to_token_maps_[edge_name])
      if '-of' in retoken:
        retoken = retoken.replace('-of', ' ' + name_to_token_maps_['-of'])
      new_graph_str_list.append(retoken)
    elif token.startswith('_'):
      # The token here is a content node, we move the postfix before
      # the lemma to make the tokenization recognize the postfix as
      # non-tokenizable token, e.g., _look_v_up -> v_up_look_.
      lemma = token.split('_')[1]
      postfix = '_'.join(token.split('_')[2:])
      if postfix in name_to_token_maps_:
        retoken = name_to_token_maps_[postfix] + '_' + lemma + '_'
      else:
        retoken = postfix + '_' + lemma + '_'
      new_graph_str_list.append(retoken)
    else:
      if token in name_to_token_maps_:
        new_graph_str_list.append(name_to_token_maps_[token])
      else:
        new_graph_str_list.append(token)
  return ' '.join(new_graph_str_list)


def reverse_tokened_graph_str(graph_str: Text,
                              data_version: str = 'v0') -> Text:
  """Reverses the retokened tokens in graph string to their original tokens."""
  token_map = metrics_utils.TOKEN_TO_NAME_MAPS[data_version]
  new_graph_str_list = []
  for token in graph_str.split():
    search_result = re.search(re.compile(r'<extra_id_[0-9]+>'), token)
    if search_result:
      # The token is in MISC + ARG_EDGES + FUNC_NODES.
      match_str = search_result.group(0)
      if match_str:
        retoken = token_map[match_str]
        if retoken == '-of' and new_graph_str_list:
          retoken = new_graph_str_list.pop() + '-of'
      else:
        retoken = token
    elif token.endswith('_'):
      # The token here is a content node.
      postfix = '_'.join(token.split('_')[:-2])
      lemma = token.split('_')[-2]
      if postfix in token_map:
        retoken = '_' + lemma + '_' + token_map[postfix]
      elif lemma in token_map:
        retoken = '_' + '<nolemma>' + '_' + token_map[lemma]
      else:
        retoken = '_' + lemma + '_' + postfix
    else:
      retoken = token
    new_graph_str_list.append(retoken)
  graph_str = ' '.join(new_graph_str_list)
  return graph_str


def _merge_token_prob(token_list: List[Text],
                      beam_scores: List[float],
                      data_version: str = 'v0') -> List[Dict[Text, Any]]:
  """Merges tokens to graph subgraphs (nodes/edges), and sums up beam scores.

  For example, for tokens ['p', '_', 'down', '_'],
  and beam scores [0.0, -1.1920928955078125e-07, 0.0, 0.0],
  we will merge tokens into 'p_down_', and compute the corresponding
  probability exp(sum([0.0, -1.1920928955078125e-07, 0.0, 0.0])).

  Args:
    token_list: a list of tokens to be merged.
    beam_scores: a list of beam scores for each token position, the length is
      equal to the length of `token_list`.
    data_version: the DeepBank version.

  Returns:
    subgraph_infos: a list of dictionaries, which contiains the values and
      probaibilties.
  """
  subgraph_infos = []
  # If the node/edge name is not finished, store the previous tokens
  # in `node_stack`.
  node_stack = []
  # Records each token's start index and end index.
  start, end = 0, 0
  # Checks if the current token is in quotes.
  start_quote = False
  for i, token in enumerate(token_list):
    token = token.replace(' ', '')
    end_symbol_case = token and token[0] not in ['(', ')', '"', '*']
    edge_case = token not in metrics_utils.ARG_EDGES
    piece_case1 = token not in metrics_utils.FUNC_NODES[data_version] + [
        'polite', 'addressee']
    piece_case2 = i + 1 < len(token_list) and token_list[i + 1] == '_'
    piece_case3 = i > 0 and token_list[i - 1] == '_'
    func_node_case = piece_case1 or piece_case2 or piece_case3
    if not token:
      end += 1
      continue
    elif not start_quote and token == '-of':
      end += 1
      previous_info = subgraph_infos.pop()
      subgraph_infos.append({
          'value': previous_info['value'] + token,
          'prob': np.exp(sum(beam_scores[start - 1:end])),
          'align': '%s-%s' % (start - 1, end)
      })
      start = end
    elif token == '"':
      # The start or end of a double quote.
      if token_list[:i].count('"') % 2 == 0:
        start_quote = True
      else:
        start_quote = False
      if not start_quote and node_stack:
        subgraph_infos.append({
            'value': ''.join(node_stack),
            'prob': np.exp(sum(beam_scores[start:end])),
            'align': '%s-%s' % (start, end)
        })
        node_stack = []
        start = end
      end += 1
      # Adds subgraph info for quote, which is non-mergable symbol.
      subgraph_infos.append({
          'value': token,
          'prob': np.exp(sum(beam_scores[start:end])),
          'align': '%s-%s' % (start, end)
      })
      start = end
    elif start_quote or (end_symbol_case and edge_case and func_node_case):
      # Merges the pieces of node/attribute name in to a full name,
      # e.g., ['p', '_', 'down', '_'] into 'p_down_'.
      # `end_symbol_case`: the token is an end symbol (brackets,
      #   double quote or star)
      # `edge_case`: the token is an argument edge.
      # `func_node_case`: the token is a functional node. Ensures that
      #   pieces of node/attribute name are not included in the
      #   function nodes (func_node_case), e.g., 'comp' in 'compact'.
      node_stack.append(token)
      end += 1
    else:
      # Gets non-mergable symbol, first write the merged node from node_stack,
      # and then write the non-mergable symbol.
      # Example: for tokens, 'comp', 'act', ')', node_stack = ['comp', 'act'].
      # We first write node name 'compact', and then write non-mergable
      # symbol ')'.
      if node_stack:
        subgraph_infos.append({
            'value': ''.join(node_stack),
            'prob': np.exp(sum(beam_scores[start:end])),
            'align': '%s-%s' % (start, end)
        })
        node_stack = []
        start = end
      end += 1
      subgraph_infos.append({
          'value': token,
          'prob': np.exp(sum(beam_scores[start:end])),
          'align': '%s-%s' % (start, end)
      })
      start = end
  if node_stack:
    # The graph is incomplete and `node_stack` has something left.
    subgraph_infos.append({
        'value': ''.join(node_stack),
        'prob': np.exp(sum(beam_scores[start:end])),
        'align': '%s-%s' % (start, end)
    })
  return subgraph_infos


def _assign_prob_to_variable_free_penman(token_list: List[Text],
                                         beam_scores: List[float],
                                         data_version: str = 'v0') -> Text:
  """Assigns the probabilities from the model output to each node/edge in the varialbe PENMAN string.

  Example output: ( unknown_1.0 :ARG1_0.9999 ( _look_v_1_0.9987 ))

  Args:
    token_list: a list of tokens from the model output.
    beam_scores: a list of beam scores for each token position, the length is
      equal to the length of `token_list`.
    data_version: the DeepBank version.

  Returns:
    A variable-free penman string with probabilities attached to
      nodes/attributes/edges.
  """
  subgraph_infos = _merge_token_prob(token_list, beam_scores, data_version)
  graph_str_list = []
  quote_count = 0
  for subgraph_info in subgraph_infos:
    token = subgraph_info['value']
    prob = subgraph_info['prob']
    if token in metrics_utils.ARG_EDGES and token != ':carg':
      # The token here is an edge.
      token_prob = token + '_' + str(prob)
    elif '-of' in token and token.split(
        '-of')[0] in metrics_utils.ARG_EDGES and quote_count % 2 == 0:
      # The token here is a reversed version of edge, e.g., 'ARG1-of'.
      token_prob = token[:-3] + '_' + str(prob) + '-of'
    elif token in metrics_utils.FUNC_NODES[data_version] + [
        'polite', 'addressee'] and quote_count % 2 == 0:
      # The token here is a functional node, e.g., 'pron'.
      token_prob = token + '_' + str(prob)
    elif token[-1] == '_' and quote_count % 2 == 0:
      # The token here is a surface node, e.g., 'v_1_look_'.
      # Here we need to reorder the node to '_look_v_1'.
      lemma = token.split('_')[-2]
      postfix = '_'.join(token.split('_')[:-2])
      token_prob = '_' + lemma + '_' + postfix + '_' + str(prob)
    elif '*' in token and quote_count % 2 == 0:
      previous_component = graph_str_list.pop()
      previous_token = '_'.join(previous_component.split('_')[:-1])
      try:
        previous_prob = float(previous_component.split('_')[-1])
        token_prob = previous_token + '*' * token.count('*') + '_' + str(
            previous_prob * prob)
      except ValueError:
        logging.warning('Unable to retrieve prob in previous '
                        'component %s.', previous_component)
        token_prob = previous_token + '*' * token.count('*') + '_' + str(prob)
    elif token in ['(', ')', ':carg']:
      # For those symbol, there is no need to assign probabilities.
      token_prob = token
    elif token == '"':
      quote_count += 1
      token_prob = token
    else:
      token_prob = token + '_' + str(prob)
    graph_str_list.append(token_prob)
  return ' '.join(graph_str_list)


def assign_prob_to_penman(token_list: List[Text],
                          beam_scores: List[float],
                          data_version: str = 'v0') -> Text:
  """Assigns the probabilities from the model output to each node/edge in the PENMAN string.

  Example output: ( x0 / unknown_1.0 :ARG1_0.9999 ( x1 / _look_v_1_0.9987 ))

  Args:
    token_list: a list of tokens from the model output.
    beam_scores: a list of beam scores for each token position, the length is
      equal to the length of `token_list`.
    data_version: the DeepBank version.

  Returns:
    A penman string with probabilities attached to nodes/attributes/edges.
  """
  variable_free_penman_with_prob = _assign_prob_to_variable_free_penman(
      token_list, beam_scores, data_version)
  return transfer_to_penman(variable_free_penman_with_prob)


def _transfer_dataflow_src_history(input_str: str,
                                   use_custom_token: bool = True) -> str:
  """Transfers the Dataflow s-expression in source history into the penman graph string."""
  history_list = re.split('__User|__StartOfProgram', input_str)[1:-1]
  user_program_triple_list = []
  last_user_sent = history_list[-1]
  for i in range(0, len(history_list)-1, 2):
    user_program_triple_list.append((history_list[i], history_list[i+1]))
  output_str = ''
  for user_sent, program_str in user_program_triple_list:
    entity_name = ''
    lispress_str = program_str
    entity_count = program_str.count('entity@')
    if entity_count:
      entity_name = ' '.join(program_str.split()[-entity_count:]) + ' '
      lispress_str = lispress_str.replace(entity_name, '')
    lispress = lispress_utils.parse_lispress(lispress_str)
    penman_str = lispress_utils.lispress_to_graph(lispress)
    penman_str = lispress_utils.pre_processing_graph_str(penman_str)
    if use_custom_token:
      penman_str = lispress_utils.retok_graph_str(penman_str)
    output_str += '__User%s__StartOfProgram %s %s' % (user_sent, penman_str,
                                                      entity_name)
  output_str += '__User%s__StartOfProgram' % last_user_sent
  return output_str


def convert_dataflow_to_penman(orig_str: str,
                               orig_type: str = 'src',
                               use_custom_token: bool = True) -> str:
  """Converts the original input/output in Dataflow into the penman graph string."""
  if orig_type == 'tgt':
    tgt_lispress = lispress_utils.parse_lispress(orig_str)
    tgt_penman = lispress_utils.lispress_to_graph(tgt_lispress)
    tgt_penman = lispress_utils.pre_processing_graph_str(tgt_penman)
    if use_custom_token:
      tgt_penman = lispress_utils.retok_graph_str(tgt_penman)
    return tgt_penman
  else:
    return _transfer_dataflow_src_history(orig_str, use_custom_token)


def convert_snips_mtop_to_penman(output_str: str,
                                 dataset_name: str,
                                 use_custom_token: bool = True) -> str:
  """Converts the output string into the penman graph string."""
  output_str = output_str.replace('[', '[ ')
  graph_str_list = []
  retok_graph_str_list = []
  level_arg_dict = collections.defaultdict(int)
  current_level = -1
  carg_stack = []
  name_to_token_maps = metrics_utils.NAME_TO_TOKEN_MAPS[dataset_name]
  for token in output_str.split():
    if token == '[':
      current_level += 1
      if current_level > 0:
        arg_name = ':ARG' + str(level_arg_dict[current_level] + 1)
        graph_str_list.append(arg_name)
        retok_graph_str_list.append(name_to_token_maps[arg_name])
        level_arg_dict[current_level] += 1
      graph_str_list.append('(')
      retok_graph_str_list.append('(')
    elif token == ']':
      if carg_stack:
        graph_str_list.append(':carg " ' + '_'.join(carg_stack) + ' "')
        retok_graph_str_list.append(name_to_token_maps[':carg'] + ' " ' +
                                    '_'.join(carg_stack) + ' "')
        carg_stack = []
      graph_str_list.append(')')
      retok_graph_str_list.append(')')
      current_level -= 1
    elif token in name_to_token_maps:
      graph_str_list.append(token)
      retok_graph_str_list.append(name_to_token_maps[token])
    else:
      carg_stack.append(token)
  if use_custom_token:
    return ' '.join(retok_graph_str_list)
  return ' '.join(graph_str_list)

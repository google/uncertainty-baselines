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

"""Utility functions for processing PENMAN string (linearization of graph)."""

import re
import lispress_utils  # local file import from baselines.t5.data.deepbank


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

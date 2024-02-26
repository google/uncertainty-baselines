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

"""Retrieval utils."""
import re
from typing import Text, Dict, Any, List, Optional

import t5.data
from data.deepbank import graph_utils  # local file import from baselines.t5
from data.deepbank import meta_graph_utils  # local file import from baselines.t5
from data.deepbank import penman_utils  # local file import from baselines.t5

DEFAULT_VOCAB = t5.data.get_default_vocabulary()


def _check_pred_input(pred: Dict[Text, Any], beam_id: int = 0):
  """Checks if prediction input is well-formed."""
  if not (f'prediction_{beam_id}_ids' in pred and
          f'prediction_{beam_id}' in pred):
    raise ValueError(f'Some of the following items are missing in {pred}:'
                     f'`prediction_{beam_id}_ids`, `prediction_{beam_id}`.')


def _get_pred_graph_info(sentence: Text,
                         pred: Dict[Text, Any],
                         tgt_penman: penman_utils.PENMANStr,
                         beam_id: int = 0,
                         data_version: str = 'v0',
                         prefix: str = 'x'):
  """Gets prediction graph infos from prediction Dict."""
  return meta_graph_utils.GraphInfo(
      token_ids=pred[f'prediction_{beam_id}_ids'],
      beam_scores=pred['beam_scores'][beam_id],
      sentence=sentence,
      prediction=pred[f'prediction_{beam_id}'],
      target=tgt_penman.retokened_variable_free_penman,
      data_version=data_version,
      prefix=prefix
  )


def _random_update_src(src: Text, tgt: Text,
                       total_num_examplar: int = 0, max_num_examplar: int = 1,
                       depth: int = 1, use_alignment: bool = True,
                       use_custom_token: bool = True,
                       max_seq_length: int = 512, data_version: str = 'v0'):
  """Updates input sentence with new randomly retrieved examplars."""
  src_length = len(DEFAULT_VOCAB.encode(src))
  while True:
    if use_alignment:
      subgraph_penman_str, align_sent = graph_utils.get_random_linear_subgraph(
          tgt, src, level=depth)
      examplar_str = ' @@ %s' % align_sent
    else:
      subgraph_penman_str = graph_utils.get_random_linear_subgraph(
          tgt, src, level=depth, return_align_sent=False)
      examplar_str = ''
    subgraph_penman = penman_utils.PENMANStr(
        subgraph_penman_str,
        variable_free=False,
        data_version=data_version)
    subgraph_output = (
        subgraph_penman.retokened_variable_free_penman if use_custom_token
        else subgraph_penman.variable_free_penman)
    examplar_str += ' ## %s' % subgraph_output
    examplar_length = len(DEFAULT_VOCAB.encode(examplar_str))
    if src_length + examplar_length < max_seq_length and (total_num_examplar <
                                                          max_num_examplar):
      total_num_examplar += 1
      src += examplar_str
      src_length += examplar_length
    else:
      break
  return src


def _oracle_update_src(src: Text, tgt: Text,
                       pred_graph_info: meta_graph_utils.GraphInfo,
                       tgt_dag: graph_utils.DAG,
                       tgt_prefix: Text = 'x', pred_prefix: Text = 'y',
                       total_num_examplar: int = 0,
                       max_num_examplar: int = 1,
                       depth: int = 1, use_alignment: bool = True,
                       use_custom_token: bool = True,
                       max_seq_length: int = 512, data_version: str = 'v0',
                       with_uncertain: bool = False):
  """Updates input sentence with new retrieved examplars based on oracle."""
  src_length = len(DEFAULT_VOCAB.encode(src))
  tgt_instances, tgt_attributes, tgt_relations = tgt_dag.get_triples()
  pred_graph = meta_graph_utils.MetaGraph(pred_graph_info, pred_prefix)
  # Sets `compate_attribute` to False to avoid compate alignment information,
  # which the prediction does not have.
  mapping, _ = graph_utils.get_best_match(tgt_instances, tgt_attributes,
                                          tgt_relations,
                                          pred_graph_info.instances,
                                          pred_graph_info.attributes,
                                          pred_graph_info.relations, tgt_prefix,
                                          pred_prefix,
                                          compare_attribute=False)
  oracle_node_idxs = list(
      graph_utils.find_mismatched_node_idxs(mapping, tgt_prefix, pred_prefix,
                                            tgt_instances, tgt_attributes,
                                            tgt_relations,
                                            pred_graph_info.instances,
                                            pred_graph_info.attributes,
                                            pred_graph_info.relations))
  if with_uncertain:
    # If `with_uncertain`, re-order `oracle_node_idxs` based on probabilties
    # (ascending order).
    mapping_dict = graph_utils.get_mapping_dict(
        tgt_prefix, pred_prefix, mapping, reverse=True)
    uncertain_node_idxs = graph_utils.find_uncertain_node_idxs(
        pred_graph.instance_prob_dict,
        pred_graph.attribute_prob_dict,
        pred_graph.relation_prob_dict,
        mapping_dict)
    oracle_node_idxs = [
        idx for idx in uncertain_node_idxs if idx in oracle_node_idxs]
  excluded_node_idxs = []
  while oracle_node_idxs:
    oracle_node_idx = oracle_node_idxs.pop()
    excluded_node_idxs.append(oracle_node_idx)
    if use_alignment:
      (subgraph_penman_str, align_sent,
       oracle_node_idxs) = graph_utils.get_oracle_linear_subgraph(
           tgt_instances, tgt_attributes, tgt_relations,
           src, oracle_node_idx, oracle_node_idxs, level=depth)
      examplar_str = ' @@ %s' % align_sent
    else:
      (subgraph_penman_str,
       oracle_node_idxs) = graph_utils.get_oracle_linear_subgraph(
           tgt_instances, tgt_attributes, tgt_relations,
           src, oracle_node_idx, oracle_node_idxs, level=depth,
           return_align_sent=False)
      examplar_str = ''
    subgraph_penman = penman_utils.PENMANStr(
        subgraph_penman_str,
        variable_free=False,
        data_version=data_version)
    subgraph_output = (
        subgraph_penman.retokened_variable_free_penman if use_custom_token
        else subgraph_penman.variable_free_penman)
    examplar_str += ' ## %s' % subgraph_output
    examplar_length = len(DEFAULT_VOCAB.encode(examplar_str))
    if src_length + examplar_length < max_seq_length and (total_num_examplar <
                                                          max_num_examplar):
      total_num_examplar += 1
      src += examplar_str
      src_length += examplar_length
    else:
      break
  if src_length < max_seq_length and total_num_examplar < max_num_examplar:
    # If the number of oracle retrival examplars has not reached the budget,
    # and the input sequence length has not reached the max sequence length,
    # we left the rest for uncertain retrieval or random retrieval based
    # on gold.
    if with_uncertain:
      src = _uncertain_update_src(src, tgt, pred_graph_info,
                                  tgt_dag, tgt_prefix,
                                  pred_prefix, total_num_examplar,
                                  max_num_examplar, depth, use_alignment,
                                  use_custom_token, max_seq_length,
                                  data_version, excluded_node_idxs)
    else:
      src = _random_update_src(src, tgt, total_num_examplar, max_num_examplar,
                               depth, use_alignment, use_custom_token,
                               max_seq_length, data_version)
  return src


def _uncertain_update_src(src: Text, tgt: Text,
                          pred_graph_info: meta_graph_utils.GraphInfo,
                          tgt_dag: graph_utils.DAG,
                          tgt_prefix: Text = 'x', pred_prefix: Text = 'y',
                          total_num_examplar: int = 0,
                          max_num_examplar: int = 1,
                          depth: int = 1, use_alignment: bool = True,
                          use_custom_token: bool = True,
                          max_seq_length: int = 512, data_version: str = 'v0',
                          excluded_node_idxs: Optional[List[Text]] = None):
  """Updates input sentence with new retrieved examplars based on uncertainty."""
  src_length = len(DEFAULT_VOCAB.encode(src))
  tgt_instances, tgt_attributes, tgt_relations = tgt_dag.get_triples()
  pred_graph = meta_graph_utils.MetaGraph(pred_graph_info, pred_prefix)
  # Sets `compate_attribute` to False to avoid compate alignment information,
  # which the prediction does not have.
  mapping, _ = graph_utils.get_best_match(pred_graph.instances,
                                          pred_graph.attributes,
                                          pred_graph.relations,
                                          tgt_instances, tgt_attributes,
                                          tgt_relations,
                                          pred_prefix,
                                          tgt_prefix,
                                          compare_attribute=False)
  mapping_dict = graph_utils.get_mapping_dict(pred_prefix, tgt_prefix, mapping)
  uncertain_node_idxs = graph_utils.find_uncertain_node_idxs(
      pred_graph.instance_prob_dict, pred_graph.attribute_prob_dict,
      pred_graph.relation_prob_dict, mapping_dict)
  # Excludes node indexes in predefined node index list.
  if excluded_node_idxs:
    uncertain_node_idxs = [
        idx for idx in uncertain_node_idxs if idx not in excluded_node_idxs]
  while uncertain_node_idxs:
    uncertain_node_idx = uncertain_node_idxs.pop()
    if use_alignment:
      (subgraph_penman_str, align_sent,
       uncertain_node_idxs) = graph_utils.get_uncertain_linear_subgraph(
           tgt_instances, tgt_attributes, tgt_relations,
           src, uncertain_node_idx, uncertain_node_idxs, level=depth)
      examplar_str = ' @@ %s' % align_sent
    else:
      (subgraph_penman_str,
       uncertain_node_idxs) = graph_utils.get_uncertain_linear_subgraph(
           tgt_instances, tgt_attributes, tgt_relations,
           src, uncertain_node_idx, uncertain_node_idxs, level=depth,
           return_align_sent=False)
      examplar_str = ''
    subgraph_penman = penman_utils.PENMANStr(
        subgraph_penman_str,
        variable_free=False,
        data_version=data_version)
    subgraph_output = (
        subgraph_penman.retokened_variable_free_penman if use_custom_token
        else subgraph_penman.variable_free_penman)
    examplar_str += ' ## %s' % subgraph_output
    examplar_length = len(DEFAULT_VOCAB.encode(examplar_str))
    if src_length + examplar_length < max_seq_length and (total_num_examplar <
                                                          max_num_examplar):
      total_num_examplar += 1
      src += examplar_str
      src_length += examplar_length
    else:
      break
  if src_length < max_seq_length and total_num_examplar < max_num_examplar:
    # If the number of uncertain retrival examplars has not reached the budget,
    # and the input sequence length has not reached the max sequence length,
    # we left the rest for random retrieval based on gold.
    src = _random_update_src(src, tgt, total_num_examplar, max_num_examplar,
                             depth, use_alignment, use_custom_token,
                             max_seq_length, data_version)
  return src


def random_retrieval_on_gold(src: Text,
                             tgt: Text,
                             max_num_examplar: int = 1,
                             depth: int = 1,
                             use_alignment: bool = True,
                             use_custom_token: bool = True,
                             max_seq_length: int = 512,
                             data_version: str = 'v0'):
  """Retrieves random subgraphs based on gold graphs."""
  if not max_num_examplar:
    max_num_examplar = max_seq_length
  tgt_no_alignment = re.sub(r' :lnk "<[0-9]+:[0-9]+>"', '', tgt)
  tgt_penman = penman_utils.PENMANStr(
      tgt_no_alignment, variable_free=False, data_version=data_version)
  src = _random_update_src(src, tgt, 0, max_num_examplar,
                           depth, use_alignment, use_custom_token,
                           max_seq_length, data_version)
  if use_custom_token:
    return src, tgt_penman.retokened_variable_free_penman
  else:
    return src, tgt_penman.variable_free_penman


def oracle_retrieval_on_gold(src: Text,
                             tgt: Text,
                             pred: Dict[Text, Any],
                             max_num_examplar: int = 1,
                             depth: int = 1,
                             use_alignment: bool = True,
                             use_custom_token: bool = True,
                             max_seq_length: int = 512,
                             data_version: str = 'v0',
                             beam_id: int = 0,
                             with_uncertain: bool = False):
  """Retrieves oracle subgraphs based on gold graphs."""
  if not max_num_examplar:
    max_num_examplar = max_seq_length
  tgt_prefix = 'x'
  pred_prefix = 'y'
  tgt_no_alignment = re.sub(r' :lnk "<[0-9]+:[0-9]+>"', '', tgt)
  tgt_penman = penman_utils.PENMANStr(
      tgt_no_alignment, variable_free=False, data_version=data_version)
  tgt_dag = graph_utils.parse_string_to_dag(tgt)
  tgt_dag.change_node_prefix(tgt_prefix)
  _check_pred_input(pred, beam_id)
  pred_graph_info = _get_pred_graph_info(src, pred, tgt_penman, beam_id,
                                         data_version, pred_prefix)
  if not pred_graph_info.pred_parsed:
    # The prediction is an ill-formed graph, if `with_uncertain` is True,
    # use uncertain retrieval on gold instead, otherwise use random retrieval
    # on gold instead.
    if with_uncertain:
      return uncertain_retrieval_on_gold(src, tgt, pred, max_num_examplar,
                                         depth, use_alignment,
                                         use_custom_token, max_seq_length,
                                         data_version)
    else:
      return random_retrieval_on_gold(src, tgt, max_num_examplar, depth,
                                      use_alignment, use_custom_token,
                                      max_seq_length, data_version)
  src = _oracle_update_src(src, tgt, pred_graph_info, tgt_dag, tgt_prefix,
                           pred_prefix, 0, max_num_examplar,
                           depth, use_alignment, use_custom_token,
                           max_seq_length, data_version, with_uncertain)
  if use_custom_token:
    return src, tgt_penman.retokened_variable_free_penman
  else:
    return src, tgt_penman.variable_free_penman


def uncertain_retrieval_on_gold(src: Text,
                                tgt: Text,
                                pred: Dict[Text, Any],
                                max_num_examplar: int = 1,
                                depth: int = 1,
                                use_alignment: bool = True,
                                use_custom_token: bool = True,
                                max_seq_length: int = 512,
                                data_version: str = 'v0',
                                beam_id: int = 0):
  """Retrieves uncertain subgraphs based on gold graphs."""
  if not max_num_examplar:
    max_num_examplar = max_seq_length
  tgt_prefix = 'x'
  pred_prefix = 'y'
  tgt_no_alignment = re.sub(r' :lnk "<[0-9]+:[0-9]+>"', '', tgt)
  tgt_penman = penman_utils.PENMANStr(
      tgt_no_alignment, variable_free=False, data_version=data_version)
  tgt_dag = graph_utils.parse_string_to_dag(tgt)
  tgt_dag.change_node_prefix(tgt_prefix)
  _check_pred_input(pred, beam_id)
  pred_graph_info = _get_pred_graph_info(src, pred, tgt_penman, beam_id,
                                         data_version, pred_prefix)
  if not pred_graph_info.pred_parsed:
    # The prediction is an ill-formed graph, use random retrieval
    # on gold instead.
    return random_retrieval_on_gold(src, tgt, max_num_examplar, depth,
                                    use_alignment, use_custom_token,
                                    max_seq_length, data_version)
  src = _uncertain_update_src(src, tgt, pred_graph_info, tgt_dag, tgt_prefix,
                              pred_prefix, 0, max_num_examplar,
                              depth, use_alignment, use_custom_token,
                              max_seq_length, data_version)
  if use_custom_token:
    return src, tgt_penman.retokened_variable_free_penman
  else:
    return src, tgt_penman.variable_free_penman


def oracle_uncertain_retrieval_on_gold(src: Text,
                                       tgt: Text,
                                       pred: Dict[Text, Any],
                                       max_num_examplar: int = 1,
                                       depth: int = 1,
                                       use_alignment: bool = True,
                                       use_custom_token: bool = True,
                                       max_seq_length: int = 512,
                                       data_version: str = 'v0',
                                       beam_id: int = 0):
  """Retrieves oracle subgraphs based on uncertainty-ordered gold graphs."""
  return oracle_retrieval_on_gold(src, tgt, pred, max_num_examplar, depth,
                                  use_alignment, use_custom_token,
                                  max_seq_length, data_version, beam_id,
                                  with_uncertain=True)

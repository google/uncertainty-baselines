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

"""Meta graph utils."""
from typing import List, Text, Dict, Tuple, Any

import numpy as np
import seqio
import t5.data
from data import metrics_utils  # local file import from baselines.t5
from data.deepbank import graph_utils  # local file import from baselines.t5
from data.deepbank import penman_utils  # local file import from baselines.t5

DEFAULT_VOCAB = t5.data.get_default_vocabulary()
ObjectList = List[Tuple[Text, Text, Text]]
ObjectProbDict = Dict[Text, Any]
EdgeProbDict = Dict[Tuple[Text, Text, Text], Any]
REDUCE_FUNC_MAP = {'min': np.min, 'max': np.max, 'mean': np.mean}


class GraphInfo(object):
  """Represents significant information of a single graph prediction."""

  def __init__(self,
               token_ids: List[int],
               beam_scores: List[float],
               sentence: Text,
               prediction: Text,
               target: Text,
               vocab: seqio.SentencePieceVocabulary = DEFAULT_VOCAB,
               data_version: Text = 'v0',
               prefix: Text = 'a'):
    """Constructs a GraphInfo Example.

    Args:
      token_ids: List of token ids from the model vocabulary.
      beam_scores: List of token-level log probabilities with shape
        (max_seq_length, ).
      sentence: The example sentence.
      prediction: Pretokenized predicted output.
      target: Pretokenized target output.
      vocab: The SentencePieceVocabulary used for model training.
      data_version: Deepbank data version.
      prefix: The prefix of graph nodes.
    """
    self.data_version = data_version
    # token id '0' represents padding and '1' represents EOS symbol.
    self.token_ids = [i for i in token_ids if i > 1]
    self.tokens = [
        metrics_utils.single_token_transfer(vocab.decode([id]), data_version)
        for id in self.token_ids
    ]
    self.log_probs = beam_scores[:len(self.token_ids)]
    # Sequence log_prob should include EOS symbol.
    self.seq_log_prob = np.sum(beam_scores[:len(self.token_ids)+1])
    self.sentence = sentence
    self.pred_penman = penman_utils.PENMANStr(
        prediction, variable_free=True, retokenized=True,
        data_version=data_version).penman
    self.target_penman = penman_utils.PENMANStr(
        target, variable_free=True, retokenized=True,
        data_version=data_version).penman
    self.instances, self.attributes, self.relations = [], [], []
    self.pred_parsed = True
    try:
      dag = graph_utils.parse_string_to_dag(self.pred_penman)
      dag.change_node_prefix(prefix)
    except LookupError:
      self.pred_parsed = False
    if self.pred_parsed:
      self.instances, self.attributes, self.relations = dag.get_triples()
      if not self.instances: self.pred_parsed = False
    else:
      self.instances, self.attributes, self.relations = [], [], []

  def get_performance_score(self):
    """Get precision, recall and Smatch scores."""
    if not self.pred_parsed:
      return 0.00, 0.00, 0.00
    return graph_utils.get_smatch(self.pred_penman,
                                  self.target_penman)


class MetaGraph(object):
  """Represents a meta graph object.

  A meta graph is a merged graph from different beam candidates.
  For example, we have 3 beam candidate graphs:

    (unknown_1.0
      :ARG2_0.64 ( _look_v_up_1.0 ) )

    (unknown_0.98
      :ARG1_0.36 ( _look_v_up_1.0 ) )

    (unknown_1.0
      :ARG2_0.89 ( _look_v_up_1.0 ) )

  The linearized meta graph before probability normalization is like:

    (unknown_1.0#0.98#1.0
      :ARG1|ARG2_0.36|0.64#0.89 ( _look_v_up_1.0#1.0#1.0 ) )

  The graphical probabilities are attached to nodes/attributes/edges
  separated by symbol '_', with symbol '|' as a separator for
  different node/edge values, and with symbol '#' as a separator for
  different probabilty values for the same node/edge value.
  """

  def __init__(self,
               oracle_graph_info: GraphInfo,
               oracle_prefix: Text = 'a',
               normalization_reduce_type: Text = 'max'):
    """Constructs a MetaGraph Example.

    Args:
      oracle_graph_info: GraphInfo object from the first graph.
      oracle_prefix: The oracle prefix is to ensure that the prefix
        is unified during each process of adding new graph.
      normalization_reduce_type: The reduce type for normalizating the
        graphical probabilities. Should be one of ['min', 'max', 'mean'].
    """
    self.sentence = oracle_graph_info.sentence
    self.target_penman = oracle_graph_info.target_penman
    self.oracle_prefix = oracle_prefix
    if normalization_reduce_type not in REDUCE_FUNC_MAP.keys():
      raise ValueError(
          f'Reduce type {normalization_reduce_type} if not supported.')
    self.normalization_reduce_type = normalization_reduce_type
    if not oracle_graph_info.pred_parsed:
      # The oracle graph is ill-formed.
      self.instances, self.attributes, self.relations = [], [], []
      self.instance_prob_dict = {}
      self.attribute_prob_dict = {}
      self.relation_prob_dict = {}
    else:
      (self.instances, self.attributes, self.relations,
       self.instance_prob_dict, self.attribute_prob_dict,
       self.relation_prob_dict) = self.get_object_lists_and_prob_dicts(
           oracle_graph_info, self.oracle_prefix,
           oracle_graph_info.data_version)
    (self.normalized_instance_prob_dict, self.normalized_attribute_prob_dict,
     self.normalized_relation_prob_dict) = self.get_normalized_graph_probs(
         self.normalization_reduce_type)

  def get_object_lists_and_prob_dicts(self,
                                      graph_info: GraphInfo,
                                      prefix: Text = 'a',
                                      data_version: Text = 'v0'):
    """Gets node/attribute/edge lists with probability dicts.

    This function is basically transferring sequential level probabilities
    into graphical probabilities.

    Args:
      graph_info: Information required from the graph prediction.
      prefix: The target prefix.
      data_version: DeepBank data version.

    Returns:
      Object (node/attribute/edge) lists with probability dicts.
    """
    # TODO(lzi): check if the length of lists will change due to error in
    #   `get_object_lists_and_prob_dicts`.
    penman_with_prob = penman_utils.get_assign_prob_func(data_version)(
        graph_info.tokens, graph_info.log_probs, data_version)
    dag_with_prob = graph_utils.parse_string_to_dag(penman_with_prob)
    dag_with_prob.change_node_prefix(prefix)
    (instances_with_prob,
     attributes_with_prob,
     relations_with_prob) = dag_with_prob.get_triples()
    return graph_utils.get_object_lists_and_prob_dicts(instances_with_prob,
                                                       attributes_with_prob,
                                                       relations_with_prob)

  def merge_value_and_prob(self, v1: Text, v2: Text, p1: Text, p2: Text):
    """Merges values and probabilities between oracle and added graphs.

    Args:
      v1: The oracle node/attribute/edge values.
      v2: The added node/attribute/edge value.
      p1: The oracle node/attribute/edge probabilities.
      p2: The added node/attribute/edge probability.

    Returns:
      new_value: The merged value.
      new_prob: The merged probability.
    """
    if v1 == v2:
      # Example input: v1 = 'unknown', v2 = 'unknown'; p1 = '1.0', p2 = '0.98'.
      # Example output: new_value = 'unknown', new_prob = '1.0#0.98'
      new_value = v1
      new_prob = p1 + '#' + p2
    elif '|' not in v1:
      # Example input: v1 = 'unknown', v2 = '_see_v_1'; p1 = '1.0', p2 = '0.98'.
      # Example output: new_value = 'unknown|_see_v_1', new_prob = '1.0|0.98'.
      new_value = v1 + '|' + v2
      new_prob = p1 + '|' + p2
    else:
      all_values = v1.split('|')
      all_probs = p1.split('|')
      if v2 in all_values:
        # Example input: v1 = 'unknown|_see_v_1', v2 = '_see_v_1';
        #                p1 = '1.0|0.98', p2 = '0.95'.
        # Example output: new_value = 'unknown|_see_v_1',
        #                 new_prob = '1.0|0.98#0.95'.
        pos = all_values.index(v2)
        new_value = v1
        all_probs[pos] = all_probs[pos] + '#' + p2
        new_prob = '|'.join(all_probs)
      else:
        # Example input: v1 = 'unknown|_see_v_1', v2 = '_see_v_from';
        #                p1 = '1.0|0.98', p2 = '0.95'.
        # Example output: new_value = 'unknown|_see_v_1|_see_v_from',
        #                 new_prob = '1.0|0.98|0.95'.
        new_value = v1 + '|' + v2
        new_prob = p1 + '|' + p2
    return new_value, new_prob

  def merge_nodes(self,
                  added_instances: ObjectList,
                  added_instance_prob_dict: ObjectProbDict,
                  mapping_dict: Dict[Text, Text]):
    """Merges graph nodes."""
    node_dict = graph_utils.transfer_triple_to_dict(self.instances, 'node')
    instance_prob_dict = self.instance_prob_dict
    added_node_dict = graph_utils.transfer_triple_to_dict(
        added_instances, 'node')
    new_instances = []
    new_instance_prob_dict = {}
    node_id_mapping = {}
    num_new_node = 0
    for node_id, node_value in node_dict.items():
      if node_id in mapping_dict:
        # The node has been matched and should be merged.
        node_prob = str(instance_prob_dict[node_id])
        added_node_id = mapping_dict[node_id]
        added_node_value = added_node_dict[added_node_id]
        added_node_prob = str(added_instance_prob_dict[added_node_id])
        new_node_value, new_node_prob = self.merge_value_and_prob(
            node_value, added_node_value, node_prob, added_node_prob)
        new_node_id = self.oracle_prefix + str(num_new_node)
        num_new_node += 1
        node_id_mapping[node_id] = new_node_id
        node_id_mapping[added_node_id] = new_node_id
        new_instances.append(('instance', new_node_id, new_node_value))
        new_instance_prob_dict[new_node_id] = new_node_prob
      else:
        # The node has not been matched. No need to merge.
        new_node_id = self.oracle_prefix + str(num_new_node)
        num_new_node += 1
        node_id_mapping[node_id] = new_node_id
        new_instances.append(('instance', new_node_id, node_value))
        new_instance_prob_dict[new_node_id] = str(instance_prob_dict[node_id])
    for added_node_id, added_node_value in added_node_dict.items():
      if added_node_id not in mapping_dict.values():
        # The added node has not been matched. No need to merge.
        new_node_id = self.oracle_prefix + str(num_new_node)
        num_new_node += 1
        node_id_mapping[added_node_id] = new_node_id
        new_instances.append(('instance', new_node_id, added_node_value))
        new_instance_prob_dict[new_node_id] = str(
            added_instance_prob_dict[added_node_id])
    return new_instances, new_instance_prob_dict, node_id_mapping

  def merge_attributes(self, added_attributes: ObjectList,
                       added_attribute_prob_dict: ObjectProbDict,
                       mapping_dict: Dict[Text, Text],
                       node_id_mapping: Dict[Text, Text]):
    """Merges graph attributes."""
    # TODO(lzi): Change the config of `attribute_dict`, for DeepBank the
    #   attribute name is always ':carg' but it is not the case for other
    #   datasets.
    attribute_dict = graph_utils.transfer_triple_to_dict(
        self.attributes, 'attribute')
    attribute_prob_dict = self.attribute_prob_dict
    added_attribute_dict = graph_utils.transfer_triple_to_dict(
        added_attributes, 'attribute')
    new_attributes = []
    new_attribute_prob_dict = {}
    for attribute_id, attribute_value in attribute_dict.items():
      if attribute_id in mapping_dict:
        # The attribute has been matched.
        attribute_prob = str(attribute_prob_dict[attribute_id])
        added_attribute_id = mapping_dict[attribute_id]
        if added_attribute_id not in added_attribute_dict:
          # The corresponding node has no attribute. No need to merge.
          new_attribute_id = node_id_mapping[attribute_id]
          new_attributes.append(('carg', new_attribute_id, attribute_value))
          new_attribute_prob_dict[new_attribute_id] = str(
              attribute_prob_dict[attribute_id])
        else:
          # The corresponding node has attribute and should be merged.
          added_attribute_value = added_attribute_dict[added_attribute_id]
          added_attribute_prob = str(
              added_attribute_prob_dict[added_attribute_id])
          new_attribute_value, new_attribute_prob = self.merge_value_and_prob(
              attribute_value, added_attribute_value, attribute_prob,
              added_attribute_prob)
          new_attribute_id = node_id_mapping[attribute_id]
          new_attributes.append(('carg', new_attribute_id, new_attribute_value))
          new_attribute_prob_dict[new_attribute_id] = new_attribute_prob
      else:
        # The attribute has not been matched. No need to merge.
        new_attribute_id = node_id_mapping[attribute_id]
        new_attributes.append(('carg', new_attribute_id, attribute_value))
        new_attribute_prob_dict[new_attribute_id] = str(
            attribute_prob_dict[attribute_id])
    for added_attribute_id, added_attribute_value in added_attribute_dict.items(
    ):
      if added_attribute_id not in mapping_dict.values():
        new_attribute_id = node_id_mapping[added_attribute_id]
        new_attributes.append(('carg', new_attribute_id, added_attribute_value))
        new_attribute_prob_dict[new_attribute_id] = str(
            added_attribute_prob_dict[added_attribute_id])
    return new_attributes, new_attribute_prob_dict

  def merge_edges(self, added_relations: ObjectList,
                  added_relation_prob_dict: EdgeProbDict,
                  mapping_dict: Dict[Text, Text],
                  node_id_mapping: Dict[Text, Text]):
    """Merges graph edges."""
    relation_prob_dict = self.relation_prob_dict
    added_edge_dict = graph_utils.transfer_triple_to_dict(
        added_relations, 'edge')
    new_relations = []
    new_relation_prob_dict = {}
    added_edge_cache = []  # For recording the added edge in added graph.
    for (edge_value, edge_start,
         edge_end), edge_prob in relation_prob_dict.items():
      if edge_start in mapping_dict and edge_end in mapping_dict:
        # Both start and end node are matched.
        added_edge_start = mapping_dict[edge_start]
        added_edge_end = mapping_dict[edge_end]
        if (added_edge_start, added_edge_end) not in added_edge_dict:
          # Though the start and end node are matched, the correponding
          # edge in added graph cannot be found. No need to merge.
          new_relations.append((edge_value, node_id_mapping[edge_start],
                                node_id_mapping[edge_end]))
          new_relation_prob_dict[(edge_value, node_id_mapping[edge_start],
                                  node_id_mapping[edge_end])] = str(edge_prob)
        else:
          added_edge_value_list = added_edge_dict[(added_edge_start,
                                                   added_edge_end)]
          if edge_value in added_edge_value_list:
            # Edge values are matched and shoud be merged.
            added_edge_prob = str(added_relation_prob_dict[(edge_value,
                                                            added_edge_start,
                                                            added_edge_end)])
            new_edge_value, new_edge_prob = self.merge_value_and_prob(
                edge_value, edge_value, str(edge_prob), added_edge_prob)
            new_relations.append((new_edge_value, node_id_mapping[edge_start],
                                  node_id_mapping[edge_end]))
            new_relation_prob_dict[(new_edge_value, node_id_mapping[edge_start],
                                    node_id_mapping[edge_end])] = new_edge_prob
            added_edge_cache.append(
                (new_edge_value, node_id_mapping[edge_start],
                 node_id_mapping[edge_end]))
          else:
            # Edge values are not matched and should be merged.
            # Here we defaultly merge `edge_value` and
            # `added_edge_value_list[0]`.
            added_edge_value = added_edge_value_list[0]
            added_edge_prob = str(added_relation_prob_dict[(added_edge_value,
                                                            added_edge_start,
                                                            added_edge_end)])
            new_edge_value, new_edge_prob = self.merge_value_and_prob(
                edge_value, added_edge_value, str(edge_prob), added_edge_prob)
            new_relations.append((new_edge_value, node_id_mapping[edge_start],
                                  node_id_mapping[edge_end]))
            new_relation_prob_dict[(new_edge_value, node_id_mapping[edge_start],
                                    node_id_mapping[edge_end])] = new_edge_prob
            added_edge_cache.append(
                (added_edge_value, node_id_mapping[edge_start],
                 node_id_mapping[edge_end]))
      else:
        # Either start or end node is not matched. No need to merge.
        new_relations.append((edge_value, node_id_mapping[edge_start],
                              node_id_mapping[edge_end]))
        new_relation_prob_dict[(edge_value, node_id_mapping[edge_start],
                                node_id_mapping[edge_end])] = str(edge_prob)
    for (added_edge_value, added_edge_start,
         added_edge_end), added_edge_prob in added_relation_prob_dict.items():
      new_edge_start = node_id_mapping[added_edge_start]
      new_edge_end = node_id_mapping[added_edge_end]
      if (added_edge_value, new_edge_start,
          new_edge_end) not in added_edge_cache:
        # This edge value has not been added yet.
        # Note that this edge can be either matched or mismatched.
        new_relations.append((added_edge_value, new_edge_start, new_edge_end))
        new_relation_prob_dict[(added_edge_value, new_edge_start,
                                new_edge_end)] = str(added_edge_prob)
    return new_relations, new_relation_prob_dict

  def get_merged_graph_from_mapping(
      self, added_instances: ObjectList, added_attributes: ObjectList,
      added_relations: ObjectList, added_instance_prob_dict: ObjectProbDict,
      added_attribute_prob_dict: ObjectProbDict,
      added_relation_prob_dict: EdgeProbDict, mapping_dict: Dict[Text, Text]):
    """Merges graphs based on `mapping_dict`."""
    # Merges nodes.
    new_instances, new_instance_prob_dict, node_id_mapping = self.merge_nodes(
        added_instances, added_instance_prob_dict, mapping_dict)
    # Merges attributes.
    new_attributes, new_attribute_prob_dict = self.merge_attributes(
        added_attributes, added_attribute_prob_dict, mapping_dict,
        node_id_mapping)
    # Merges edges.
    new_relations, new_relation_prob_dict = self.merge_edges(
        added_relations, added_relation_prob_dict, mapping_dict,
        node_id_mapping)
    return (new_instances, new_attributes, new_relations,
            new_instance_prob_dict, new_attribute_prob_dict,
            new_relation_prob_dict)

  def get_merged_graph_from_added_graph(self,
                                        added_graph_info: GraphInfo,
                                        added_prefix: Text = 'b'):
    """Gets merged graph from added graph.

    Args:
      added_graph_info: GraphInfo object from the added graph.
      added_prefix: The added prefix is to ensure that the prefix
        is different from the oracle prefix.
    Returns:
      Updated object lists and prob dicts.
    """
    if added_prefix == self.oracle_prefix:
      raise ValueError(
          'The added prefix should not be the same as the oracle prefix.')
    (added_instances, added_attributes, added_relations,
     added_instance_prob_dict, added_attribute_prob_dict,
     added_relation_prob_dict) = self.get_object_lists_and_prob_dicts(
         added_graph_info, added_prefix, added_graph_info.data_version)

    mapping, _ = graph_utils.get_best_match(
        self.instances, self.attributes, self.relations,
        added_instances, added_attributes, added_relations,
        self.oracle_prefix, added_prefix,
        on_meta=True)
    mapping_dict = graph_utils.get_mapping_dict(self.oracle_prefix,
                                                added_prefix, mapping)
    return self.get_merged_graph_from_mapping(added_instances, added_attributes,
                                              added_relations,
                                              added_instance_prob_dict,
                                              added_attribute_prob_dict,
                                              added_relation_prob_dict,
                                              mapping_dict)

  def add_graph(self, added_graph_info: GraphInfo, added_prefix: Text = 'b'):
    """Adds new graph into the meta graph."""
    if not self.instances and added_graph_info.pred_parsed:
      # The oracle graph is ill-formed, but the added graph is well-formed.
      # Replaces the oracle graph with the added graph.
      (self.instances, self.attributes, self.relations, self.instance_prob_dict,
       self.attribute_prob_dict,
       self.relation_prob_dict) = self.get_object_lists_and_prob_dicts(
           added_graph_info, self.oracle_prefix, added_graph_info.data_version)
    elif self.instances and added_graph_info.pred_parsed:
      # Both the oracle graph and added graph are well-formed.
      # Need to be merged.
      (self.instances, self.attributes, self.relations, self.instance_prob_dict,
       self.attribute_prob_dict,
       self.relation_prob_dict) = self.get_merged_graph_from_added_graph(
           added_graph_info, added_prefix)
    (self.normalized_instance_prob_dict, self.normalized_attribute_prob_dict,
     self.normalized_relation_prob_dict) = self.get_normalized_graph_probs(
         self.normalization_reduce_type)

  def get_normalized_object_probs(self, prob_dict, reduce_type: Text = 'max'):
    """Normalizes object probabilities based on certain objective."""
    new_prob_dict = {}
    reduce_function = REDUCE_FUNC_MAP[reduce_type]
    for k, prob_str in prob_dict.items():
      # Ensures the values in `prob_dict` is strings, which is not the
      # case at initialization.
      prob_str = str(prob_str)
      probs_per_value = prob_str.split('|')
      new_probs_per_value = []
      for prob_per_value in probs_per_value:
        probs_per_pred = [float(p) for p in prob_per_value.split('#')]
        new_probs_per_value.append(str(reduce_function(probs_per_pred)))
      new_prob_dict[k] = '|'.join(new_probs_per_value)
    return new_prob_dict

  def get_normalized_graph_probs(self, reduce_type: Text = 'max'):
    """""Normalizes graph probabilities based on certain objective."""
    normalized_instance_prob_dict = self.get_normalized_object_probs(
        self.instance_prob_dict, reduce_type)
    normalized_attribute_prob_dict = self.get_normalized_object_probs(
        self.attribute_prob_dict, reduce_type)
    normalized_relation_prob_dict = self.get_normalized_object_probs(
        self.relation_prob_dict, reduce_type)
    return (normalized_instance_prob_dict, normalized_attribute_prob_dict,
            normalized_relation_prob_dict)

  def get_dag_match_for_graphical_calibration(self):
    """Gets node/attribute/edge probability match lists for graphical calibration."""
    normalized_object_lists_and_prob_dicts = (
        self.instances, self.attributes, self.relations,
        self.normalized_instance_prob_dict,
        self.normalized_attribute_prob_dict,
        self.normalized_relation_prob_dict)
    return graph_utils.get_dag_match_for_calibration(
        normalized_object_lists_and_prob_dicts, self.target_penman,
        self.oracle_prefix)

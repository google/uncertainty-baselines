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

"""Graph utils.

Parse linearized DAG graph into sets of nodes and edges.
Graph alignment via hill-climbing algorithm.
Smatch score based on graph alignment (match).
"""
import collections
import random
import re
from absl import logging
from data.deepbank import graph_linear_utils  # local file import from baselines.t5


class DAG(object):
  """Represents a directed acyclic graph (DAG) example."""

  def __init__(self,
               node_list=None,
               node_value_list=None,
               relation_list=None,
               attribute_list=None):
    """Constructs a DAGExample.

    Args:
      node_list: names of nodes in DAG, e.g., "a0", "a1".
      node_value_list: values of nodes in DAG, e.g., "_look_v_up" for a node
        named "a0".
      relation_list: list of relations between two nodes.
      attribute_list: list of attributes (links between one node and one
        constant value).
    """
    # Initialize graph nodes using list of nodes name.
    # Root, by default, is the first in var_list.
    self.nodes = [] if node_list is None else node_list[:]
    self.root = None if not self.nodes else self.nodes[0]
    self.node_values = [] if node_value_list is None else node_value_list[:]
    self.relations = [] if relation_list is None else relation_list[:]
    self.attributes = [] if attribute_list is None else attribute_list[:]

  def change_node_prefix(self, prefix):
    """Renames graph prefix to avoid same node index in two different graphs."""
    node_map_dict = {}
    # Map each node to its new name (e.g., "a1").
    for i in range(len(self.nodes)):
      node_map_dict[self.nodes[i]] = prefix + str(i)
    # Update node named.
    for i, v in enumerate(self.nodes):
      self.nodes[i] = node_map_dict[v]
    # Update node name in relations.
    for node_relations in self.relations:
      for i, l in enumerate(node_relations):
        node_relations[i][1] = node_map_dict[l[1]]

  def get_triples(self, return_attribute=True):
    """Get the triples in three lists.

    Args:
      return_attribute: whether to return attribute triples. If not, only return
        instance triples and relation triples, where attribute triples will
        be taken as subset of relation triples.

    Returns:
      instance_triple: list of triples representing instances,
        e.g., instance(a0, _look_v_up).
      attribute_triple: list of triples representing attributes,
        e.g., carg(a1, "John").
      relation_triple: list of triples representing relations,
        e.g., ARG1(a1, a2).
    """
    instance_triple = []
    relation_triple = []
    attribute_triple = []
    for i in range(len(self.nodes)):
      instance_triple.append(("instance", self.nodes[i], self.node_values[i]))
      # l[0] is the relation name.
      # l[1] is the other node this node has relation with.
      for l in self.relations[i]:
        relation_triple.append((l[0], self.nodes[i], l[1]))
      # l[0] is the attribute name.
      # l[1] is the attribute value.
      for l in self.attributes[i]:
        attribute_triple.append((l[0], self.nodes[i], l[1]))
    if return_attribute:
      return instance_triple, attribute_triple, relation_triple
    relation_triple = relation_triple + attribute_triple
    return instance_triple, relation_triple


def parse_string_to_dag(line) -> DAG:
  """Parse a linearized DAG to a graph object.

  This parsing algorithm scans the line once and process each character,
  in a shift reduce style.

  Reference:
    [1] Michael Wayne Goodman. Penman: An open-Source Library and Tool
    for AMR graphs. In _Annual Meeting of the Association for Computational
    Linguistics: System Demonstrations_, 2020.
    https://aclanthology.org/2020.acl-demos.35.pdf

  Args:
    line: linearized DAG in one-line string.

  Returns:
    parsed_dag: a DAG example derived from line.
  """
  def update_triple(node_relation_dict, u, r, v):
    """Detect relation and reverse it if `r` contains "-of"."""
    # We detect a relation (r) between u and v, with direction u to v.
    # In most cases, if relation name ends with "-of", e.g."ARG1-of",
    # it is reverse of some relation. For example, if a is "ARG1-of" b,
    # so we can also say b is "ARG1" a.
    # If the relation name ends with "-of", we store the reverse relation.
    if r.endswith("-of"):
      node_relation_dict[v].append((r[:-3], u))
    else:
      node_relation_dict[u].append((r, v))

  # Parses the linearized string into graph nodes and edges
  # in shift-reduce style.
  stack = []  # Node stack for parsing.
  cur_charseq = []  # Current not-yet-reduced character sequence.
  node_name_list = []  # Node name list (order: occurrence of the node).

  # Dict of nodes. key: node name; value: node value.
  node_dict = {}
  # Dict of node and its related edges (with seen node name).
  # key: node name; value: list of (relation name, the other node name).
  node_relation_dict_with_seen_node = collections.defaultdict(list)
  # Dict of node and its related attributes and edges (with unseen node name).
  # key: node name; value: list of (attribute name, const value) or
  # (relation name, unseen node name).
  node_relation_dict_with_attr_and_unseen_node = collections.defaultdict(list)
  # Current relation name.
  cur_relation_name = ""
  # Having unmatched quote string.
  in_quote = False

  # Current state. It denotes the last significant symbol encountered, e.g.,
  #  1 for (, 2 for :, 3 for /, and 0 for start state or ')'.
  # Last significant symbol is ( --- start processing node name.
  # Last significant symbol is : --- start processing relation name.
  # Last significant symbol is / --- start processing node value
  #   (concept name).
  # Last significant symbol is ) --- current node processing is complete
  # Note that if these symbols are inside parenthesis, they are not
  #   significant symbols.
  state = 0
  for i, c in enumerate(line.strip()):
    if c == " ":
      # Allows space in relation name.
      if state == 2:
        cur_charseq.append(c)
      continue
    if c == "\"":
      # Flips in_quote value when a quote symbol is encountered.
      # Insert placeholder if in_quote from last symbol.
      if in_quote:
        cur_charseq.append("_")
      in_quote = not in_quote
    elif c == "(":
      # Not significant symbol if inside quote.
      if in_quote:
        cur_charseq.append(c)
        continue
      # Case: Encounters the attribute name.
      # Example: For string ":ARG1 (x0 ...". Here we want to retrieve
      # the symbol "ARG1".
      if state == 2:
        # In this state, current relation name should be empty.
        if cur_relation_name:
          logging.warning("Format error when processing %s", line[0:i + 1])
          return DAG()
        # Update current relation name for future use.
        cur_relation_name = "".join(cur_charseq).strip()
        cur_charseq[:] = []
      state = 1
    elif c == ":":
      if in_quote:
        # Not significant symbol if inside quote.
        cur_charseq.append(c)
        continue
      if state == 3:
        # Case: Last significant symbol is "/", and now we encounter ":".
        # Example: For string ":ARG1 (x1 / named :carg "James")". We get
        # node value "named" at this point.
        node_value = "".join(cur_charseq)
        # Clear current char sequence.
        cur_charseq[:] = []
        # Pop node name ("x1" in the above example).
        cur_node_name = stack[-1]
        # Update node name/value map.
        node_dict[cur_node_name] = node_value
      elif state == 2:
        # Case: Last significant symbol is ":", and now we encounter ":".
        # Example: For string ":ARG1 x1 :card "23"", the problem is that
        # we cannot decide if node value is attribute value
        # (constant) or node value (variable) at this moment.
        # The solution is to check if the node value has been defined
        # before.
        temp_attr_value = "".join(cur_charseq)
        cur_charseq[:] = []
        parts = temp_attr_value.split()
        if len(parts) < 2:
          logging.warning("Format in processing; part len < 2: %s",
                          line[0:i + 1])
          return DAG()
        # For the above example, relation name is "ARG1", and node value is
        # "x1". Note that this node name might not be encountered before.
        relation_name = parts[0].strip()
        relation_value = parts[1].strip()
        # We need to link upper level node to the current top of stack
        # is upper level node.
        if not stack:
          logging.warning("Error in processing %s; %s; %s.", line[:i],
                          relation_name, relation_value)
          return DAG()
        if relation_value not in node_dict:
          # If we have not seen this node name before.
          update_triple(node_relation_dict_with_attr_and_unseen_node, stack[-1],
                        relation_name, relation_value)
        else:
          update_triple(node_relation_dict_with_seen_node, stack[-1],
                        relation_name, relation_value)
      state = 2
    elif c == "/":
      if in_quote:
        cur_charseq.append(c)
        continue
      if state == 1:
        # Case: Last significant symbol is "(", and now we encounter "/".
        # Example: For string "(x1 / named", we want to retrieve "x1" here.
        node_name = "".join(cur_charseq)
        cur_charseq[:] = []
        # If this node name is already in node_dict, it is duplicate
        if node_name in node_dict:
          logging.warning("Duplicate node name: %s in parsing DAG", node_name)
          return DAG()
        # Push the node name to stack.
        stack.append(node_name)
        # Add it to node name list.
        node_name_list.append(node_name)
        # If this node is part of the relation, e.g.,
        # :ARG1 (x1 / named)
        # cur_relation_name is ARG1, and node name is x1,
        # we have a relation ARG1(upper level node, x1).
        if cur_relation_name:
          update_triple(node_relation_dict_with_seen_node, stack[-2],
                        cur_relation_name, node_name)
          cur_relation_name = ""
      else:
        # Error if in other state
        logging.warning("Error in parsing DAG %s", line[0:i + 1])
        return DAG()
      state = 3
    elif c == ")":
      if in_quote:
        cur_charseq.append(c)
        continue
      # Stack should be non-empty to find upper level node.
      if not stack:
        logging.warning("Unmatched parenthesis at position %d in processing %s",
                        i, line[0:i + 1])
        return DAG()
      if state == 2:
        # Case: Last significant symbol is ":", and now we encounter ")".
        # Example: For string ":carg "Brown")" and ":ARG1 x1)", and we want
        # to retrieve "Brown" and "x1" here.
        temp_attr_value = "".join(cur_charseq)
        cur_charseq[:] = []
        parts = temp_attr_value.split()
        if len(parts) < 2:
          logging.warning("Error proceesing %s; %s", line[:i + 1],
                          temp_attr_value)
          return DAG()
        relation_name = parts[0].strip()
        relation_value = parts[1].strip()
        # Attribute value not seen before.
        # Note that it might be a constant attribute value, or an unseen node
        # process this after we have seen all the node names.
        if relation_value not in node_dict:
          update_triple(node_relation_dict_with_attr_and_unseen_node, stack[-1],
                        relation_name, relation_value)
        else:
          update_triple(node_relation_dict_with_seen_node, stack[-1],
                        relation_name, relation_value)
      elif state == 3:
        # Case: Last significant symbol is "/", and now we encounter ")".
        # Example: ":ARG1 (x1 / named)", and we want to retrieve "named"
        # here.
        node_value = "".join(cur_charseq)
        cur_charseq[:] = []
        cur_node_name = stack[-1]
        # Map node name to its value.
        node_dict[cur_node_name] = node_value
      # Pop from stack, as the current node has been processed.
      stack.pop()
      cur_relation_name = ""
      state = 0
    else:
      # Not significant symbols, so we just shift.
      cur_charseq.append(c)
  # Creates data structures to initialize a DAG.
  node_value_list = []
  relation_list = []
  attribute_list = []
  for v in node_name_list:
    if v not in node_dict:
      logging.warning("Error: node name not found %s.", v)
      return DAG()
    else:
      node_value_list.append(node_dict[v])
    # Builds relation list and attribute list for this node.
    node_rel_list = []
    node_attr_list = []
    if v in node_relation_dict_with_seen_node:
      for v1 in node_relation_dict_with_seen_node[v]:
        node_rel_list.append([v1[0], v1[1]])
    if v in node_relation_dict_with_attr_and_unseen_node:
      for v2 in node_relation_dict_with_attr_and_unseen_node[v]:
        # If value is in quote, it is a constant value.
        # Strips the quote and put it in attribute map.
        if v2[1][0] == "\"" and v2[1][-1] == "\"":
          node_attr_list.append([[v2[0]], v2[1][1:-1]])
        # If value is a node name.
        elif v2[1] in node_dict:
          node_rel_list.append([v2[0], v2[1]])
        else:
          node_attr_list.append([v2[0], v2[1]])
    # Each node has a relation list and attribute list.
    relation_list.append(node_rel_list)
    attribute_list.append(node_attr_list)
  # Adds TOP as an attribute. The attribute value just needs to be constant
  # attribute_list[0].append(["TOP", "top"])
  return DAG(node_name_list, node_value_list, relation_list, attribute_list)


def normalize(item):
  """Lowercase."""
  return item.lower()


def compare_values(value1, value2, on_meta=False):
  """Compare different node/attribute/edge names.

  Args:
    value1: first node/attribute/edge value.
    value2: second node/attribute/edge value.
    on_meta: whether the match is based on meta graph, which is a merged graph
      from different beam candidates. If set to True, this means that the
      component name can be concatenated from multiple names generated by
      different beam candidates using symbol "|", e.g., node name is
      "nn_u_unknown|_halftime_n_1", and both node name "nn_u_unknown" and
      "_halftime_n_1" are valid for node name match.

  Returns:
    True if names are matched.
  """
  if not on_meta:
    return value1 == value2
  else:
    return value2 in value1.split("|")


def get_candidate_mappings_with_weights(instance1, attribute1, relation1,
                                        instance2, attribute2, relation2,
                                        prefix1, prefix2,
                                        compare_instance=True,
                                        compare_attribute=True,
                                        compare_relation=True,
                                        on_meta=False):
  """Computes all possible node mapping candidates and their weights.

  (the triple matching number gain resulting from mapping one node in DAG1
  to another node in DAG2)

  Reference:
    [1]: Shu Cai and Kevin Knight. Smatch: an Evaluation Metric for
    Semantic Feature Structures. In _Annual Meeting
    of the Association for Computational Linguistics_, 2013.
    https://aclanthology.org/P13-2131.pdf

  Args:
    instance1: instance triples of DAG1.
    attribute1: attribute triples of DAG1.
    relation1: relation triples of DAG1.
    instance2: instance triples of DAG2.
    attribute2: attribute triples of DAG2.
    relation2: relation triples of DAG2.
    prefix1: prefix label for DAG1.
    prefix2: prefix label for DAG2.
    compare_instance: if compute pool based on instances.
    compare_attribute: if compute pool based on attributes.
    compare_relation: if compute pool based on relations.
    on_meta: whether the match is based on meta graph, which is a merged graph
      from different beam candidates. If set to True, this means that the
      component name can be concatenated from multiple names generated by
      different beam candidates using symbol "|", e.g., node name is
      "nn_u_unknown|_halftime_n_1", and both node name "nn_u_unknown" and
      "_halftime_n_1" are valid for node name match.

  Returns:
    candidate_mapping: a list of candidates nodes. The ith element contains
      the node indices (in DAG2) the ith node (in DAG1) can map to. (resulting
      in non-zero triple match).
    weight_dict: a dictionary which contains the matching triple number for
      every pair of node mapping. The key is a node pair. The value is another
      dictionary. key {-1} is triple match resulting from this node pair
      alone (instance triples and attribute triples), and other keys are
      node pairs that can result in relation triple match together with
      the first node pair.
  """
  candidate_mapping = []
  weight_dict = {}
  for instance1_item in instance1:
    # Each candidate mapping is a set of node indices.
    candidate_mapping.append(set())
    if compare_instance:
      for instance2_item in instance2:
        # If both triples are instance triples and have the same value.
        if normalize(instance1_item[0]) == normalize(
            instance2_item[0]) and compare_values(
                normalize(instance1_item[2]),
                normalize(instance2_item[2]),
                on_meta=on_meta):
          # Gets node index by stripping the prefix.
          node1_index = int(instance1_item[1][len(prefix1):])
          node2_index = int(instance2_item[1][len(prefix2):])
          candidate_mapping[node1_index].add(node2_index)
          node_pair = (node1_index, node2_index)
          # Uses (-1,) as key in weight_dict for instance triples and attribute
          # triples.
          if node_pair in weight_dict:
            weight_dict[node_pair][(-1,)] += 1
          else:
            weight_dict[node_pair] = {}
            weight_dict[node_pair][(-1,)] = 1

  if compare_attribute:
    for attribute1_item in attribute1:
      for attribute2_item in attribute2:
        # If both attribute relation triple have the same relation name and
        # value.
        if normalize(attribute1_item[0]) == normalize(
            attribute2_item[0]) and compare_values(
                normalize(attribute1_item[2]),
                normalize(attribute2_item[2]),
                on_meta=on_meta):
          node1_index = int(attribute1_item[1][len(prefix1):])
          node2_index = int(attribute2_item[1][len(prefix2):])
          candidate_mapping[node1_index].add(node2_index)
          node_pair = (node1_index, node2_index)
          # Uses (-1,) as key in weight_dict for instance triples and attribute
          # triples.
          if node_pair in weight_dict:
            weight_dict[node_pair][(-1,)] += 1
          else:
            weight_dict[node_pair] = {}
            weight_dict[node_pair][(-1,)] = 1

  if compare_relation:
    for relation1_item in relation1:
      for relation2_item in relation2:
        # If both relation share the same name.
        if compare_values(
            normalize(relation1_item[0]),
            normalize(relation2_item[0]),
            on_meta=on_meta):
          node1_index_dag1 = int(relation1_item[1][len(prefix1):])
          node1_index_dag2 = int(relation2_item[1][len(prefix2):])
          node2_index_dag1 = int(relation1_item[2][len(prefix1):])
          node2_index_dag2 = int(relation2_item[2][len(prefix2):])
          # Adds mapping between two nodes.
          candidate_mapping[node1_index_dag1].add(node1_index_dag2)
          candidate_mapping[node2_index_dag1].add(node2_index_dag2)
          node_pair1 = (node1_index_dag1, node1_index_dag2)
          node_pair2 = (node2_index_dag1, node2_index_dag2)
          if node_pair2 != node_pair1:
            # Updates weight_dict weight. Note that we need to update both
            # entries for future search:
            # i.e weight_dict[node_pair1][node_pair2]
            #     weight_dict[node_pair2][node_pair1]
            if node1_index_dag1 > node2_index_dag1:
              # Swaps node_pair1 and node_pair2.
              node_pair1 = (node2_index_dag1, node2_index_dag2)
              node_pair2 = (node1_index_dag1, node1_index_dag2)
            if node_pair1 in weight_dict:
              if node_pair2 in weight_dict[node_pair1]:
                weight_dict[node_pair1][node_pair2] += 1
              else:
                weight_dict[node_pair1][node_pair2] = 1
            else:
              weight_dict[node_pair1] = {(-1,): 0, node_pair2: 1}
            if node_pair2 in weight_dict:
              if node_pair1 in weight_dict[node_pair2]:
                weight_dict[node_pair2][node_pair1] += 1
              else:
                weight_dict[node_pair2][node_pair1] = 1
            else:
              weight_dict[node_pair2] = {(-1,): 0, node_pair1: 1}
          else:
            # Two node pairs are the same. So we only update the weight_dict
            # once. This generally should not happen.
            if node_pair1 in weight_dict:
              weight_dict[node_pair1][(-1,)] += 1
            else:
              weight_dict[node_pair1] = {(-1,): 1}
  return candidate_mapping, weight_dict


def smart_init_mapping(candidate_mappings, instance1, instance2, on_meta=False):
  """Initialize mapping based on the concept mapping (smart initialization).

  Args:
    candidate_mappings: candidate node match list.
    instance1: instance triples of DAG1.
    instance2: instance triples of DAG2.
    on_meta: whether the match is based on meta graph, which is a merged graph
      from different beam candidates. If set to True, this means that the
      component name can be concatenated from multiple names generated by
      different beam candidates using symbol "|", e.g., node name is
      "nn_u_unknown|_halftime_n_1", and both node name "nn_u_unknown" and
      "_halftime_n_1" are valid for node name match.

  Returns:
    initialized node mapping between two DAGs.
  """
  random.seed()
  matched_dict = {}
  result = []
  # List to store node indices that have no concept match.
  no_word_match = []
  for i, candidates in enumerate(candidate_mappings):
    if not candidates:
      # No possible mapping.
      result.append(-1)
      continue
    # Node value in instance triples of DAG1.
    value1 = instance1[i][2]
    for node_index in candidates:
      value2 = instance2[node_index][2]
      # Find the first instance triple match in the candidates.
      # Instance triple match is having the same concept value.
      if compare_values(value1, value2, on_meta):
        if node_index not in matched_dict:
          result.append(node_index)
          matched_dict[node_index] = 1
          break
    if len(result) == i:
      no_word_match.append(i)
      result.append(-1)
  # If no concept match, generate a random mapping.
  for i in no_word_match:
    candidates = list(candidate_mappings[i])
    while candidates:
      # Get a random node index from candidates.
      rid = random.randint(0, len(candidates) - 1)
      candidate = candidates[rid]
      if candidate in matched_dict:
        candidates.pop(rid)
      else:
        matched_dict[candidate] = 1
        result[i] = candidate
        break
  return result


def random_init_mapping(candidate_mappings):
  """Generate a random node mapping.

  Args:
    candidate_mappings: candidate ndoe match list.

  Returns:
    randomly-generated node mapping between two DAGs.
  """
  # If needed, a fixed seed could be passed here to generate some random
  # (to help debugging).
  random.seed()
  matched_dict = {}
  result = []
  for c in candidate_mappings:
    candidates = list(c)
    if not candidates:
      # -1 indicates no possible mapping.
      result.append(-1)
      continue
    found = False
    while candidates:
      # Randomly generate an index in [0, length of candidates].
      rid = random.randint(0, len(candidates) - 1)
      candidate = candidates[rid]
      # Check if it has alreaddy been matched.
      if candidate in matched_dict:
        candidates.pop(rid)
      else:
        matched_dict[candidate] = 1
        result.append(candidate)
        found = True
        break
    if not found:
      result.append(-1)
  return result


def compute_match(mapping, weight_dict):
  """Given a node mapping, compute match number based on weight_dict.

  Args:
    mapping: a list of node index in DAG2.
      The ith element (value j) means node i in DAG1 maps to node j in DAG2.
    weight_dict: weight dictionary.

  Returns:
    match_num: matching triple number.

  Complexity: O(m*n), where m is the node number of DAG1, and n is the node
    number of DAG2.
  """
  match_num = 0
  # i is node index in DAG1, m is node index in DAG2.
  for i, m in enumerate(mapping):
    if m == -1:
      # No node maps to this node.
      continue
    # Node i in DAG1 maps to node m in DAG2.
    current_node_pair = (i, m)
    if current_node_pair not in weight_dict:
      continue
    for key in weight_dict[current_node_pair]:
      if len(key) == 1:
        # Matching triple resulting from instance/attribute triples.
        match_num += weight_dict[current_node_pair][key]
        # Only consider node index larger than i to avoid duplicates.
        # As we strore both weight_dict[node_pair1][node_pair2] and
        # weight_dict[node_pair2][node_pair1] for a relation.
      elif key[0] < i:
        continue
      elif mapping[key[0]] == key[1]:
        match_num += weight_dict[current_node_pair][key]
  return match_num


def move_gain(mapping, node_id, old_id, new_id, weight_dict):
  """Compute the triple match number gain from the move operation.

  Args:
    mapping: current node mapping
    node_id: remapped node in DAG1
    old_id: original node id in DAG2 to which node_id is mapped
    new_id: new node in to which node_id is mapped
    weight_dict: weight dictionary

  Returns:
    the triple match gain number (might be negative)
  """
  # New node mapping after moving.
  new_mapping = (node_id, new_id)
  # Node mapping before moving.
  old_mapping = (node_id, old_id)
  # New nodes mapping list (all node pairs).
  new_mapping_list = mapping[:]
  new_mapping_list[node_id] = new_id
  # If this mapping is already been investigated,
  # use saved one to avoid duplicate computing.
  gain = 0
  # Add the triple match incurred by new_mapping to gain.
  if new_mapping in weight_dict:
    for key in weight_dict[new_mapping]:
      if len(key) == 1:
        # Instance/attribute triple match.
        gain += weight_dict[new_mapping][key]
      elif new_mapping_list[key[0]] == key[1]:
        # Relation gain incurred by new_mapping and another node pair
        # in new_mapping_list.
        gain += weight_dict[new_mapping][key]
  # Deduct the triple match incurred by old_mapping from gain.
  if old_mapping in weight_dict:
    for key in weight_dict[old_mapping]:
      if len(key) == 1:
        gain -= weight_dict[old_mapping][key]
      elif mapping[key[0]] == key[1]:
        gain -= weight_dict[old_mapping][key]
  return gain


def swap_gain(mapping, node_id1, mapping_id1, node_id2, mapping_id2,
              weight_dict):
  """Compute the triple match number gain from the swapping.

  Args:
    mapping: current node mapping list
    node_id1: node 1 index in DAG1
    mapping_id1: the node index in DAG2 node 1 maps to (in the current mapping)
    node_id2: node 2 index in DAG1
    mapping_id2: the node index in DAG2 node 2 maps to (in the current mapping)
    weight_dict: weight dictionary

  Returns:
    gain: the gain number (might be negative)
  """
  new_mapping_list = mapping[:]
  # Before swapping, node_id1 maps to mapping_id1,
  #   and node_id2 maps to mapping_id2.
  # After swapping, node_id1 maps to
  #   mapping_id2 and node_id2 maps to mapping_id1.
  new_mapping_list[node_id1] = mapping_id2
  new_mapping_list[node_id2] = mapping_id1
  gain = 0
  new_mapping1 = (node_id1, mapping_id2)
  new_mapping2 = (node_id2, mapping_id1)
  old_mapping1 = (node_id1, mapping_id1)
  old_mapping2 = (node_id2, mapping_id2)
  if node_id1 > node_id2:
    new_mapping2 = (node_id1, mapping_id2)
    new_mapping1 = (node_id2, mapping_id1)
    old_mapping1 = (node_id2, mapping_id2)
    old_mapping2 = (node_id1, mapping_id1)
  if new_mapping1 in weight_dict:
    for key in weight_dict[new_mapping1]:
      if len(key) == 1:
        gain += weight_dict[new_mapping1][key]
      elif new_mapping_list[key[0]] == key[1]:
        gain += weight_dict[new_mapping1][key]
  if new_mapping2 in weight_dict:
    for key in weight_dict[new_mapping2]:
      if len(key) == 1:
        gain += weight_dict[new_mapping2][key]
      # To avoid duplicate.
      elif key[0] == node_id1:
        continue
      elif new_mapping_list[key[0]] == key[1]:
        gain += weight_dict[new_mapping2][key]
  if old_mapping1 in weight_dict:
    for key in weight_dict[old_mapping1]:
      if len(key) == 1:
        gain -= weight_dict[old_mapping1][key]
      elif mapping[key[0]] == key[1]:
        gain -= weight_dict[old_mapping1][key]
  if old_mapping2 in weight_dict:
    for key in weight_dict[old_mapping2]:
      if len(key) == 1:
        gain -= weight_dict[old_mapping2][key]
      # To avoid duplicate.
      elif key[0] == node_id1:
        continue
      elif mapping[key[0]] == key[1]:
        gain -= weight_dict[old_mapping2][key]
  return gain


def get_best_gain(mapping, candidate_mappings, weight_dict, instance_len):
  """Hill-climbing method to return the best gain swap/move can get.

  Args:
    mapping: current node mapping.
    candidate_mappings: the candidates mapping list.
    weight_dict: the weight dictionary.
    instance_len: the number of the nodes in DAG2.

  Returns:
    large_gain: the best gain we can get via swap/move operation.
    cur_mapping: current mappings.
  """
  largest_gain = 0
  # True: using swap; False: using move.
  use_swap = True
  # The node to be moved/swapped.
  node1 = None
  # Store the other node affected.
  # In swap, this other node is the node swapping with node1.
  # In move, this other node is the node node1 will move to.
  node2 = None
  # Unmatched nodes in DAG2.
  unmatched = set(range(instance_len))
  # Exclude nodes in current mapping and get unmatched nodes.
  for nid in mapping:
    if nid in unmatched:
      unmatched.remove(nid)
  for i, nid in enumerate(mapping):
    # Current node i in DAG1 maps to node nid in DAG2.
    for nm in unmatched:
      if nm in candidate_mappings[i]:
        # Remap i to another unmatched node (move).
        # (i, m) -> (i, nm)
        mv_gain = move_gain(mapping, i, nid, nm, weight_dict)
        if mv_gain > largest_gain:
          largest_gain = mv_gain
          node1 = i
          node2 = nm
          use_swap = False
  # Compute swap gain.
  for i, m in enumerate(mapping):
    for j in range(i + 1, len(mapping)):
      m2 = mapping[j]
      # No need to compute swap gain if both (i, m2) (j, m) are
      # not in candidate mappings.
      # Such a swap cannot incur any gains.
      if (m2 not in candidate_mappings[i]) and (m not in candidate_mappings[j]):
        continue
      # Swap operation (i, m) (j, m2) -> (i, m2) (j, m),
      # j starts from i+1, to avoid duplicate swap.
      sw_gain = swap_gain(mapping, i, m, j, m2, weight_dict)
      if sw_gain > largest_gain:
        largest_gain = sw_gain
        node1 = i
        node2 = j
        use_swap = True
  # Generate a new mapping based on swap/move.
  cur_mapping = mapping[:]
  if node1 is not None:
    if use_swap:
      temp = cur_mapping[node1]
      cur_mapping[node1] = cur_mapping[node2]
      cur_mapping[node2] = temp
    else:
      cur_mapping[node1] = node2
  return largest_gain, cur_mapping


def get_best_match(instance1, attribute1, relation1,
                   instance2, attribute2, relation2,
                   prefix1, prefix2,
                   compare_instance=True,
                   compare_attribute=True,
                   compare_relation=True,
                   iteration_num=5,
                   on_meta=False):
  """Get the highest triple match number between two graphs via hill-climbing.

  Reference:
    [1]: Shu Cai and Kevin Knight. Smatch: an Evaluation Metric for
    Semantic Feature Structures. In _Annual Meeting
    of the Association for Computational Linguistics_, 2013.
    https://aclanthology.org/P13-2131.pdf

  Args:
    instance1: instance triple of DAG1 ("instance", node name, node value).
    attribute1: attribute triple of DAG1 (attribute name, node name,
      attribute value).
    relation1: relation triples of DAG1 (relation name, node1 name, node2 name).
    instance2: instance triple of DAG2 ("instance", node name, node value).
    attribute2: attribute triple of DAG2 (attribute name, node name,
      attribute value).
    relation2: relation triples of DAG2 (relation name, node1 name, node2 name).
    prefix1: prefix label of DAG1.
    prefix2: prefix label of DAG2.
    compare_instance: whether match instances (nodes).
    compare_attribute: whether match attributes.
    compare_relation: whether match relations.
    iteration_num: number of iterations.
    on_meta: whether the match is based on meta graph, which is a merged graph
      from different beam candidates. If set to True, this means that the
      component name can be concatenated from multiple names generated by
      different beam candidates using symbol "|", e.g., node name is
      "nn_u_unknown|_halftime_n_1", and both node name "nn_u_unknown" and
      "_halftime_n_1" are valid for node name match.

  Returns:
    best_match: the node mapping that results in the highest triple matching
      number.
    best_match_num: the highest triple matching number.
  """
  # Computes candidate pool - all possible node match candidates.
  # In the hill-climbing, we only consider candidate in this pool to save
  # computing time.
  # `weight_dict` is a dictionary that maps a pair of node.
  (candidate_mappings, weight_dict) = get_candidate_mappings_with_weights(
      instance1, attribute1, relation1,
      instance2, attribute2, relation2,
      prefix1, prefix2,
      compare_instance=compare_instance,
      compare_attribute=compare_attribute,
      compare_relation=compare_relation,
      on_meta=on_meta)

  best_match_num = 0
  # Initializes best match mapping.
  # The ith entry is the node index in DAG2 which maps to the ith node in DAG1.
  best_mapping = [-1] * len(instance1)
  for i in range(iteration_num):
    if i == 0:
      # Smart initialization used for the first round.
      cur_mapping = smart_init_mapping(
          candidate_mappings, instance1, instance2, on_meta=on_meta)
    else:
      # Random initialiation for the other round.
      cur_mapping = random_init_mapping(candidate_mappings)
    match_num = compute_match(cur_mapping, weight_dict)

    # Compares the current mapping and candidate mappings to see if we can
    # get more gain via swap/move operation. If so, update the match number
    # and current mapping.
    while True:
      # Get the best gain.
      (gain, new_mapping) = get_best_gain(cur_mapping, candidate_mappings,
                                          weight_dict, len(instance2))
      if gain <= 0:
        break
      # Otherwise update match_num and mapping.
      match_num += gain
      cur_mapping = new_mapping[:]
    if match_num > best_match_num:
      best_mapping = cur_mapping[:]
      best_match_num = match_num
  return best_mapping, best_match_num


def get_dag_match(cur_dag1,
                  cur_dag2,
                  just_match_instance=False,
                  just_match_attribute=False,
                  just_match_relation=False):
  """Get DAG matches."""
  # Check if the linearized DAG can be successfully parsed.
  dag1_parsed = True
  dag2_parsed = True
  try:
    dag1 = parse_string_to_dag(cur_dag1)
  except LookupError:
    logging.warning("Fail to parse DAG: %s", cur_dag1)
    dag1_parsed = False
  try:
    dag2 = parse_string_to_dag(cur_dag2)
  except LookupError:
    logging.warning("Fail to parse DAG: %s", cur_dag2)
    dag2_parsed = False

  if dag1_parsed and not dag1.root:
    logging.warning("Fail to parse DAG: %s", cur_dag1)
    dag1_parsed = False
  if dag2_parsed and not dag2.root:
    logging.warning("Fail to parse DAG: %s", cur_dag2)
    dag2_parsed = False

  prefix1 = "a"
  prefix2 = "b"
  if dag1_parsed:
    dag1.change_node_prefix(prefix1)
    (instance1, attribute1, relation1) = dag1.get_triples()
  if dag2_parsed:
    dag2.change_node_prefix(prefix2)
    (instance2, attribute2, relation2) = dag2.get_triples()

  # Optionally turn off some of the node comparision.
  compare_instance = compare_attribute = compare_relation = True
  if just_match_instance:
    compare_attribute = compare_relation = False
  if just_match_attribute:
    compare_instance = compare_relation = False
  if just_match_relation:
    compare_instance = compare_attribute = False
  if dag1_parsed and dag2_parsed:
    (best_mapping, best_match_num) = get_best_match(
        instance1, attribute1, relation1,
        instance2, attribute2, relation2,
        prefix1, prefix2,
        compare_instance=compare_instance,
        compare_attribute=compare_attribute,
        compare_relation=compare_relation)
  else:
    best_mapping = []
    best_match_num = 0
  # Get the number of prediction and gold, for calculating the F score.
  test_triple_num, gold_triple_num = 0, 0
  if just_match_instance:
    if dag1_parsed:
      test_triple_num = len(instance1)
    if dag2_parsed:
      gold_triple_num = len(instance2)
  elif just_match_attribute:
    if dag1_parsed:
      test_triple_num = len(attribute1)
    if dag2_parsed:
      gold_triple_num = len(attribute2)
  elif just_match_relation:
    if dag1_parsed:
      test_triple_num = len(relation1)
    if dag2_parsed:
      gold_triple_num = len(relation2)
  else:
    if dag1_parsed:
      test_triple_num = len(instance1) + len(attribute1) + len(relation1)
    if dag2_parsed:
      gold_triple_num = len(instance2) + len(attribute2) + len(relation2)
  return best_match_num, test_triple_num, gold_triple_num, best_mapping


def get_smatch(cur_dag1,
               cur_dag2,
               just_match_instance=False,
               just_match_attribute=False,
               just_match_relation=False):
  """Get Smatch score between two DAG stings."""
  match_num, test_num, gold_num, _ = get_dag_match(cur_dag1, cur_dag2,
                                                   just_match_instance,
                                                   just_match_attribute,
                                                   just_match_relation)
  # logging.info(
  #     f"match_num: {match_num}; test_num: {test_num}; gold_num: {gold_num};")
  if test_num == 0 or gold_num == 0:
    return 0.00, 0.00, 0.00
  precision = match_num / test_num
  recall = match_num / gold_num
  if (precision + recall) != 0:
    f_score = 2 * precision * recall / (precision + recall)
    return precision, recall, f_score
  else:
    return precision, recall, 0.00


def get_mapping_dict(prefix1, prefix2, mapping):
  """Transfers the mapping list to mapping dict."""
  mapping_dict = {}
  for i, j in enumerate(mapping):
    if j != -1:
      mapping_dict[prefix1 + str(i)] = prefix2 + str(j)
  return mapping_dict


def transfer_triple_to_dict(triple_list, object_type="node"):
  """Transfers the list of triple set to dict for searching graph components."""
  # TODO(lzi): Change `attribute_dict` since there can be more than one
  #   attribute name, though for DeepBank there is only 'carg'.
  object_dict = {}
  if object_type == "edge":
    # Note that for edge, there might be multiple edges between the same set
    # set of start and end nodes, so the value of dict should be a list.
    for edge_name, start, end in triple_list:
      if (start, end) in object_dict:
        object_dict[(start, end)].append(edge_name)
      else:
        object_dict[(start, end)] = [edge_name]
  else:
    # The object here is a node or an attribute.
    for _, object_id, object_name in triple_list:
      object_dict[object_id] = object_name
  return object_dict


def _find_float_in_str(x):
  """Finds the float probability value in string."""
  float_pattern = re.compile(r"(\d(\.\d+)?)([eE][-+]?\d+)?")
  search_result = float_pattern.search(x)
  if not search_result:
    return None
  return float(search_result.group(0))


def get_object_lists_and_prob_dicts(instances, attributes, relations):
  """Gets probability dictionary for instances, attributes, relations.

  Args:
    instances: nodes with probabilities attached to node names,
      e.g., ("instance", "a0", "pron_1.0").
    attributes: attributes with probabilities attached to attribute values,
      e.g., ("carg", "a1", "John_0.9996_").
    relations: relations with probabilties attached to edge names,
      e.g., ("ARG1_1.0", "b1", "b2").

  Returns:
    new_instances: instances without probabilities.
    new_attributes: attributes without probabilities.
    new_edges: edges without probabilities。
    instance_prob_dict: probability dictionary for instances.
    attribute_prob_dict: probability dictionary for attributes.
    relation_prob_dict: probability dictionary for edges.
  """
  new_instances, new_attributes, new_relations = [], [], []
  instance_prob_dict, attribute_prob_dict, relation_prob_dict = {}, {}, {}
  # If we fail to retrieve the prob value, this node/attibute/edge
  # is invalid, and the corresponding prob value is set to -1.
  for i in instances:
    node_name = "_".join(i[2].split("_")[:-1])
    prob = _find_float_in_str(i[2].split("_")[-1])
    if not prob:
      logging.warning("Unable to retrieve node name or prob in %s.", i[2])
      new_instances.append((i[0], i[1], "<invalid>" + i[2]))
      instance_prob_dict[i[1]] = -1
    else:
      new_instances.append((i[0], i[1], node_name))
      instance_prob_dict[i[1]] = prob
  for a in attributes:
    attr_name = "_".join(a[2].rstrip("_").split("_")[:-1])
    prob = _find_float_in_str(a[2].rstrip("_").split("_")[-1])
    if not prob:
      logging.warning("Unable to retrieve attribute name or prob in %s.", a[2])
      new_attributes.append((a[0], a[1], "<invalid>" + a[2]))
      attribute_prob_dict[a[1]] = -1
    else:
      new_attributes.append((a[0], a[1], attr_name))
      attribute_prob_dict[a[1]] = prob
  for r in relations:
    edge_name = r[0].split("_")[0]
    prob = _find_float_in_str(r[0].split("_")[-1])
    if not prob:
      logging.warning("Unable to retrieve edge name or prob in %s.", r[0])
      new_relations.append(("<invalid>" + r[0], r[1], r[2]))
      relation_prob_dict[("<invalid>" + r[0], r[1], r[2])] = -1
    else:
      new_relations.append((edge_name, r[1], r[2]))
      relation_prob_dict[(edge_name, r[1], r[2])] = prob
  return (new_instances, new_attributes, new_relations, instance_prob_dict,
          attribute_prob_dict, relation_prob_dict)


def get_dag_match_for_calibration(pred_dag_object,
                                  gold_dag_str,
                                  pred_prefix="a",
                                  gold_prefix="b"):
  """Performs graph match between predicted and gold graph for calibration.

  Args:
    pred_dag_object: predicted penman graph with probabilities, which can be
      a linearized string or a tuple of graph object lists with prob dicts,
      i.e, `(instances, attributes, relations, instance_prob_dict,
      attribute_prob_dict, relation_prob_dict)`.
    gold_dag_str: gold penman graph without probaibilties.
    pred_prefix: prefix for predicted graph.
    gold_prefix: prefix for gold graph.

  Returns:
    node_prob_match_list: a list of triples (node_prob, node_matched),
      where `node_prob` indicates the probability for the node, and
      `node_matched` indicates whether this node can be found in
      the gold graph.
    attribute_prob_match_list: a list of triples (attribute_prob,
      attribute_matched), where `attribute_prob` indicates the probability
      for the attribute, and `attribute_matched` indicates whether this
      attribute can be found in the gold graph.
    edge_prob_match_list: a list of triples (edge_prob, edge_matched)
      where `edge_prob` indicates the probability for the edge, and
      `edge_matched` indicates whether this edge can be found in
      the gold graph.
  """
  assert isinstance(pred_dag_object, str) or isinstance(pred_dag_object, tuple)

  # Get gold graph object lists.
  gold_dag = parse_string_to_dag(gold_dag_str)
  gold_dag.change_node_prefix(gold_prefix)
  gold_instances, gold_attributes, gold_relations = gold_dag.get_triples()

  if isinstance(pred_dag_object, str):
    # `pred_dag` here is a linearized graph string.
    pred_dag = parse_string_to_dag(pred_dag_object)
    pred_dag.change_node_prefix(pred_prefix)
    pred_instances, pred_attributes, pred_relations = pred_dag.get_triples()
    # Retrieves the probability dictionary from the name strings.
    # For example, for the original `pred_instances`
    # [("instance", "a0", "pron_1.0")], the new `pred_instances` should be
    # [("instance", "a0", "pron")], and the `instance_prob_dict` should be
    # {'a0': 1.0}.
    (pred_instances, pred_attributes, pred_relations, instance_prob_dict,
     attribute_prob_dict, relation_prob_dict) = get_object_lists_and_prob_dicts(
         pred_instances, pred_attributes, pred_relations)
  else:
    # `pred_dag` here is a tuple of graph object lists with prob dicts.
    (pred_instances, pred_attributes, pred_relations, instance_prob_dict,
     attribute_prob_dict, relation_prob_dict) = pred_dag_object

  node_prob_match_list = []
  attribute_prob_match_list = []
  edge_prob_match_list = []
  if not pred_instances or not gold_instances:
    # Fails to parse predicted or gold penman string. Returns empty lists.
    return node_prob_match_list, attribute_prob_match_list, edge_prob_match_list

  # Aligns the prediciton and gold graphs and gets the mapping dictionary.
  mapping, _ = get_best_match(pred_instances, pred_attributes,
                              pred_relations, gold_instances,
                              gold_attributes, gold_relations,
                              pred_prefix, gold_prefix)
  mapping_dict = get_mapping_dict(pred_prefix, gold_prefix, mapping)
  gold_node_dict = transfer_triple_to_dict(gold_instances, "node")
  gold_attribute_dict = transfer_triple_to_dict(gold_attributes, "attribute")
  gold_edge_dict = transfer_triple_to_dict(gold_relations, "edge")

  # Checks each instance to see whether it is matched in the gold graph
  # and its corresponding probability.
  for pred_i in pred_instances:
    pred_index = pred_i[1]
    pred_names = pred_i[2]
    probs = str(instance_prob_dict[pred_index])
    for pred_name, prob in zip(pred_names.split("|"), probs.split("|")):
      prob = float(prob)
      if prob < 0:
        # This node is invalid. Skip.
        continue
      if pred_index not in mapping_dict:
        node_prob_match_list.append((prob, 0))
        continue
      gold_index = mapping_dict[pred_index]
      gold_name = gold_node_dict[gold_index]
      if pred_name == gold_name:
        node_prob_match_list.append((prob, 1))
      else:
        node_prob_match_list.append((prob, 0))

  # Checks each attribute to see whether it is matched in the gold graph
  # and its corresponding probability.
  for pred_a in pred_attributes:
    pred_index = pred_a[1]
    pred_names = pred_a[2]
    probs = str(attribute_prob_dict[pred_index])
    for pred_name, prob in zip(pred_names.split("|"), probs.split("|")):
      prob = float(prob)
      if prob < 0:
        # This attribute is invalid. Skip.
        continue
      if pred_index not in mapping_dict:
        attribute_prob_match_list.append((prob, 0))
        continue
      gold_index = mapping_dict[pred_index]
      if gold_index not in gold_attribute_dict:
        attribute_prob_match_list.append((prob, 0))
        continue
      # Note that there is a "_" at the end of the gold attribute name,
      # e.g., "John_".
      gold_name = gold_attribute_dict[gold_index][:-1]
      if pred_name.rstrip("_") == gold_name.rstrip("_"):
        attribute_prob_match_list.append((prob, 1))
      else:
        attribute_prob_match_list.append((prob, 0))

  # Checks each edge to see whether it is matched in the gold graph
  # and its corresponding probability.
  for pred_r in pred_relations:
    pred_names = pred_r[0]
    pred_start = pred_r[1]
    pred_end = pred_r[2]
    probs = str(relation_prob_dict[(pred_names, pred_start, pred_end)])
    for pred_name, prob in zip(pred_names.split("|"), probs.split("|")):
      prob = float(prob)
      if prob < 0:
        # This edge is invalid. Skip.
        continue
      if pred_start not in mapping_dict or pred_end not in mapping_dict:
        edge_prob_match_list.append((prob, 0))
        continue
      gold_start = mapping_dict[pred_start]
      gold_end = mapping_dict[pred_end]
      if (gold_start, gold_end) not in gold_edge_dict:
        edge_prob_match_list.append((prob, 0))
      elif pred_name in gold_edge_dict[(gold_start, gold_end)]:
        edge_prob_match_list.append((prob, 1))
      else:
        edge_prob_match_list.append((prob, 0))

  return node_prob_match_list, attribute_prob_match_list, edge_prob_match_list


def get_dot_dag_str(instances, attributes, relations, instance_prob_dict,
                    attribute_prob_dict, relation_prob_dict):
  r"""Gets dot string for visualization using graphviz in Colab.

  Usage:
  >>> import graphviz
  >>> dot_dag_str = get_dot_dag_str(instances,
                                    attributes,
                                    relations,
                                    instance_prob_dict,
                                    attribute_prob_dict,
                                    relation_prob_dict)
  >>> %graphviz dot_dag_str

  Args:
    instances: list of node triples.
    attributes: list of attribute triples.
    relations: list of edge triples.
    instance_prob_dict: dict of node probabilities.
    attribute_prob_dict: dict of attribute probabilities.
    relation_prob_dict: dict of edge probabilities.

  Returns:
    dot_str: a DOT string representing the DAG.
  """
  dot_str = ""
  for i in instances:
    node_name = i[2]
    prob_str = instance_prob_dict[i[1]]
    color_str = ""
    # Sets the color to red if uncertain at this node.
    if "|" in node_name:
      color_str = ", color=\"red\", fontcolor=\"red\""
    dot_str += "%s [label=\"%s\np=%s\"%s];\n" % (i[1], node_name, prob_str,
                                                 color_str)
  for a_id, a in enumerate(attributes):
    attr_id = "attr" + str(a_id)
    attr_name = a[2]
    prob_str = attribute_prob_dict[a[1]]
    color_str = ""
    # Sets the color to red if uncertain at this attribute.
    if "|" in attr_name:
      color_str = ", color=\"red\", fontcolor=\"red\""
    dot_str += "%s [label=\"\\\"%s\\\"\np=%s\"%s];\n" % (attr_id, attr_name,
                                                         prob_str, color_str)
    dot_str += "%s -> %s [label=\"carg\"%s];\n" % (a[1], attr_id, color_str)
  for l in relations:
    edge_name = l[0]
    prob_str = relation_prob_dict[(l[0], l[1], l[2])]
    color_str = ""
    # Sets the color to red if uncertain at this edge.
    if "|" in edge_name:
      color_str = ", color=\"red\", fontcolor=\"red\""
    dot_str += "%s -> %s [label=\"%s\np=%s\"%s];\n" % (l[1], l[2], edge_name,
                                                       prob_str, color_str)
  dot_str = "digraph {\n" + dot_str + "}"
  return dot_str


def _transfer_to_linear_triples(instances, attributes, relations):
  """Transfers instance, attribute and relation triples into universal triples for linearization.

  The universal triples are used for encoding the graph triples into
  PENMAN annotation in `graph_linear_utils.encode`, which treat all nodes
  and attributes as edges. For example,
  - Instance: ('instance', 'x0', 'unknown') -> ('x0', ':instance', 'unknown').
  - Attribute: ('carg', 'x1', 'John_') -> ('x1', ':carg', '"John"').
  - Relation: ('ARG', 'x0', 'x1') -> ('x0', 'ARG', 'x1').

  Args:
    instances: list of node triples.
    attributes: list of attribute triples.
    relations: list of edge triples.

  Returns:
    triples: the transferred universal triples.
  """
  triples = []
  for _, node_id, node_value in instances:
    triples.append((node_id, ":instance", node_value))
  for attr_name, attr_id, attr_value in attributes:
    if attr_name == "lnk":
      # The alignment information is stored in attributes with attribute
      # name `lnk`. For the final penman output, we do not want this
      # information to be included.
      continue
    triples.append(
        (attr_id, ":" + attr_name, "\"%s\"" % (attr_value.rstrip("_"))))
  for edge_name, edge_start, edge_end in relations:
    triples.append((edge_start, ":" + edge_name, edge_end))
  return triples


def _get_parent_and_child_dict(relations):
  """Gets info of parents and children of each node from relations."""
  parent_dict = collections.defaultdict(list)
  child_dict = collections.defaultdict(list)
  for _, edge_start, edge_end in relations:
    parent_dict[edge_end].append(edge_start)
    child_dict[edge_start].append(edge_end)
  return parent_dict, child_dict


def _get_valid_parent_idx(idx, parent_dict, child_dict):
  """Gets valid parent indexes that can be included in the subgraph.

  Specifically, a valid parent node is a node that has no parents and has only
  one child.

  Args:
    idx: the node index for finding the valid parents.
    parent_dict: dict of parents of each node in the graph.
    child_dict: dict of children of each node in the graph.

  Returns:
    parent_idxs: indexes of valid parent nodes.
  """
  parent_idxs = set()
  for parent_idx in parent_dict[idx]:
    if parent_idx not in parent_dict and len(child_dict[parent_idx]) == 1:
      parent_idxs.add(parent_idx)
  return parent_idxs


def _get_subgraph_idxs(subgraph_idxs, idx, parent_dict, child_dict, level=3):
  """Gets subgraph indexes given root index and number of levels.

  Specifically, starting from the root index, and given number of levels
  to traverse, we will get the subgraph indexes by recursively getting the
  subgraph indexes of the children of the node. Note that for parent of the
  node in the subgraph at each level, if the parent does not have parent
  and only has one child, it will be also included in the subgraph.

  Args:
    subgraph_idxs: set of indexes of the subgraph, which will be updated here.
    idx: the root index.
    parent_dict: dict of parents of each node in the graph.
    child_dict: dict of children of each node in the graph.
    level: number of levels to traverse.

  Returns:
    subgraph_idxs: indexes of the subgraph.
  """
  if level == 1:
    subgraph_idxs.add(idx)
    subgraph_idxs = subgraph_idxs | _get_valid_parent_idx(
        idx, parent_dict, child_dict)
    return subgraph_idxs
  subgraph_idxs.add(idx)
  subgraph_idxs = subgraph_idxs | _get_valid_parent_idx(idx, parent_dict,
                                                        child_dict)
  for child_idx in child_dict[idx]:
    subgraph_idxs = _get_subgraph_idxs(
        subgraph_idxs, child_idx, parent_dict, child_dict, level - 1)
  return subgraph_idxs


def _get_subgraph_triples(subgraph_idxs, instances, attributes, relations):
  """Gets instance, attribute and relation triples from selected subgraph indexes."""
  subgraph_instances, subgraph_attributes, subgraph_relations = [], [], []
  edge_dict = transfer_triple_to_dict(relations, "edge")
  for i in instances:
    if i[1] in subgraph_idxs:
      subgraph_instances.append(i)
  for a in attributes:
    if a[1] in subgraph_idxs:
      subgraph_attributes.append(a)
  for (edge_start, edge_end), edge_names in edge_dict.items():
    if edge_start in subgraph_idxs and edge_end in subgraph_idxs:
      for edge_name in edge_names:
        subgraph_relations.append((edge_name, edge_start, edge_end))
  return subgraph_instances, subgraph_attributes, subgraph_relations


def _get_align_dict_from_attributes(attributes):
  """Gets alignment information from attributes by retrieving `lnk`."""
  align_dict = {}
  for a in attributes:
    if a[0] == "lnk":
      # The alignment information is stored in attributes with attribute
      # name `lnk`.
      align_dict[a[1]] = tuple(
          [int(x) for x in a[2].strip("_")[1:-1].split(":")])
  if not align_dict:
    logging.warning(
        "Empty `align_dict`. Perhaps no alignment info in attributes.")
  return align_dict


def _get_align_sent(subgraph_idxs, align_dict, sentence):
  """Gets aligned sentence given subgraph indexes and alignment information."""
  align_list = []
  for idx in subgraph_idxs:
    b, e = align_dict[idx]
    if " " in sentence[b:e]:
      continue
    if (b, e) not in align_list: align_list.append((b, e))
  align_list.sort(key=lambda y: y[0])
  token_list = [sentence[b:e] for b, e in align_list]
  return " ".join(token_list)


def get_random_linear_subgraph(penman_with_align,
                               sentence,
                               level=3,
                               prefix="x",
                               return_align_sent=True):
  """Gets random linearized subgraph in PENMAN and return aligned sentence if required."""
  dag = parse_string_to_dag(penman_with_align)
  dag.change_node_prefix(prefix)
  instances, attributes, relations = dag.get_triples()
  # Gets the random node index which will be the subgraph root.
  random_idx = instances[random.randrange(len(instances))][1]
  # Gets parent and child dict from relations.
  parent_dict, child_dict = _get_parent_and_child_dict(relations)
  # Initialization of subgraph indexes.
  subgraph_idxs = set()
  subgraph_idxs = _get_subgraph_idxs(subgraph_idxs, random_idx, parent_dict,
                                     child_dict, level)
  (subgraph_instances, subgraph_attributes,
   subgraph_relations) = _get_subgraph_triples(subgraph_idxs, instances,
                                               attributes, relations)
  linear_triples = _transfer_to_linear_triples(subgraph_instances,
                                               subgraph_attributes,
                                               subgraph_relations)
  subgraph_penman = graph_linear_utils.encode(
      linear_triples, random_idx, new_line=False)
  if not return_align_sent:
    return subgraph_penman
  align_dict = _get_align_dict_from_attributes(subgraph_attributes)
  align_sent = _get_align_sent(subgraph_idxs, align_dict, sentence)
  return subgraph_penman, align_sent

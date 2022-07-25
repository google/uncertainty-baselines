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

"""Tests for graph_utils."""
from absl import logging
from absl.testing import absltest
import graph_utils  # local file import from baselines.t5.data.deepbank


class DAGParseTest(absltest.TestCase):

  def test_dag_parse_result(self):
    dag_str = ('(x0 / _introduced_v_to :ARG2 (x1 / named :carg "James" :BV-of'
               ' (x2 / proper_q)))')
    dag = graph_utils.parse_string_to_dag(dag_str)
    dag.change_node_prefix('a')
    (instance, attribute, relation) = dag.get_triples()
    instance_set = set([i[2] for i in instance])
    attribute_set = set([a[2] for a in attribute])
    relation_set = set([r[0] for r in relation])
    self.assertEqual(instance_set,
                     set(['_introduced_v_to', 'named', 'proper_q']))
    self.assertEqual(attribute_set, set(['James_']))
    self.assertEqual(relation_set, set(['ARG2', 'BV']))
    logging.info('Instances: %s', instance)
    logging.info('Attributes: %s', attribute)
    logging.info('Relations: %s', relation)


class SmatchTest(absltest.TestCase):

  def test_smatch1(self):
    dag_str1 = ('(x0 / _introduce_v_to :ARG1 (x1 / pron) :ARG2 (x2 / named '
                ':carg "James" :BV-of (x3 / proper_q)))')
    dag_str2 = ('(x0 / _introduce_v_to :ARG1 (x1 / pron) :ARG2 (x2 / named '
                ':carg "James" :BV-of (x3 / proper_q)))')
    precision, recall, f_score = graph_utils.get_smatch(
        dag_str1, dag_str2, just_match_instance=True)
    self.assertEqual(precision, 1.00)
    self.assertEqual(recall, 1.00)
    self.assertEqual(f_score, 1.00)

    precision, recall, f_score = graph_utils.get_smatch(
        dag_str1, dag_str2, just_match_attribute=True)
    self.assertEqual(precision, 1.00)
    self.assertEqual(recall, 1.00)
    self.assertEqual(f_score, 1.00)

    precision, recall, f_score = graph_utils.get_smatch(
        dag_str1, dag_str2, just_match_relation=True)
    self.assertEqual(precision, 1.00)
    self.assertEqual(recall, 1.00)
    self.assertEqual(f_score, 1.00)

  def test_smatch2(self):
    dag_str1 = ('(x0 / _introduce_v_to :ARG1 (x1 / pron) :ARG2 (x2 / named '
                ':carg "James" :BV-of (x3 / proper_q)))')
    dag_str2 = ('(x0 / _introduce_v_to :ARG1 (x1 / pron) :ARG1 (x2 / named '
                ':carg "Jenny" :BV-of (x3 / _the_q)))')
    precision, recall, f_score = graph_utils.get_smatch(
        dag_str1, dag_str2, just_match_instance=True)
    self.assertEqual(precision, 0.75)
    self.assertEqual(recall, 0.75)
    self.assertEqual(f_score, 0.75)

    precision, recall, f_score = graph_utils.get_smatch(
        dag_str1, dag_str2, just_match_attribute=True)
    self.assertEqual(precision, 0.00)
    self.assertEqual(recall, 0.00)
    self.assertEqual(f_score, 0.00)

    precision, recall, f_score = graph_utils.get_smatch(
        dag_str1, dag_str2, just_match_relation=True)
    self.assertEqual(precision, 2/3)
    self.assertEqual(recall, 2/3)
    self.assertEqual(f_score, 2/3)

  def test_smatch3(self):
    dag_str1 = ('(x0 / _introduce_v_to :ARG1 (x1 / pron) :ARG2 (x2 / named '
                ':carg "James" :BV-of (x3 / proper_q)))')
    dag_str2 = ('(x0 / _introduced_v_to :ARG1 (x1 / pron) :ARG1 (x2 / named '
                ':carg "James"))')
    precision, recall, f_score = graph_utils.get_smatch(
        dag_str1, dag_str2, just_match_instance=True)
    self.assertEqual(precision, 0.5)
    self.assertEqual(recall, 2/3)
    self.assertEqual(f_score, 2/3  / (2/3 + 1/2))

    precision, recall, f_score = graph_utils.get_smatch(
        dag_str1, dag_str2, just_match_attribute=True)
    self.assertEqual(precision, 1.00)
    self.assertEqual(recall, 1.00)
    self.assertEqual(f_score, 1.00)

    precision, recall, f_score = graph_utils.get_smatch(
        dag_str1, dag_str2, just_match_relation=True)
    self.assertEqual(precision, 1/3)
    self.assertEqual(recall, 1/2)
    self.assertEqual(f_score, 1/3 / (1/3 + 1/2))

  def test_smatch4(self):
    """Test some bad cases."""
    dag_str1 = ('(_introduce_v_to (x1 / pron) :ARG2 (x2 / named '
                ':carg "James" :BV-of (x3 / proper_q)))')
    dag_str2 = (':ARG1 (x1 / pron) :ARG1 :ARG2 (x2 / named :carg "James"))')
    precision, recall, f_score = graph_utils.get_smatch(dag_str1, dag_str2)
    self.assertEqual(precision, 0.00)
    self.assertEqual(recall, 0.00)
    self.assertEqual(f_score, 0.00)


if __name__ == '__main__':
  absltest.main()

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

# Lint as: python3
"""Tests for MultiWoz rules."""

import tensorflow as tf
import constrained_evaluation as eval_model  # local file import
import data  # local file import
import psl_model_multiwoz as model  # local file import
import psl_model_multiwoz_test_util as test_util  # local file import


class PslRulesTest(tf.test.TestCase):

  def setUp(self):
    super(PslRulesTest, self).setUp()
    self.config = test_util.TEST_MULTIWOZ_CONFIG
    self.data = test_util.DATA

    tf.random.set_seed(self.config['default_seed'])

    train_dialogs = data.add_features(
        self.data['train_data'],
        vocab_mapping=self.data['vocab_mapping'],
        accept_words=self.config['accept_words'],
        cancel_words=self.config['cancel_words'],
        end_words=self.config['end_words'],
        greet_words=self.config['greet_words'],
        info_question_words=self.config['info_question_words'],
        insist_words=self.config['insist_words'],
        slot_question_words=self.config['slot_question_words'],
        includes_word=self.config['includes_word'],
        excludes_word=self.config['excludes_word'],
        accept_index=self.config['accept_index'],
        cancel_index=self.config['cancel_index'],
        end_index=self.config['end_index'],
        greet_index=self.config['greet_index'],
        info_question_index=self.config['info_question_index'],
        insist_index=self.config['insist_index'],
        slot_question_index=self.config['slot_question_index'],
        utterance_mask=self.config['utterance_mask'],
        pad_utterance_mask=self.config['pad_utterance_mask'],
        last_utterance_mask=self.config['last_utterance_mask'],
        mask_index=self.config['mask_index'])
    train_data = data.pad_dialogs(train_dialogs, self.config['max_dialog_size'],
                                  self.config['max_utterance_size'])
    raw_train_labels = data.one_hot_string_encoding(self.data['train_labels'],
                                                    self.config['class_map'])
    train_labels = data.pad_one_hot_labels(raw_train_labels,
                                           self.config['max_dialog_size'],
                                           self.config['class_map'])
    self.train_ds = data.list_to_dataset(train_data[0], train_labels[0],
                                         self.config['shuffle_train'],
                                         self.config['batch_size'])

    test_dialogs = data.add_features(
        self.data['test_data'],
        vocab_mapping=self.data['vocab_mapping'],
        accept_words=self.config['accept_words'],
        cancel_words=self.config['cancel_words'],
        end_words=self.config['end_words'],
        greet_words=self.config['greet_words'],
        info_question_words=self.config['info_question_words'],
        insist_words=self.config['insist_words'],
        slot_question_words=self.config['slot_question_words'],
        includes_word=self.config['includes_word'],
        excludes_word=self.config['excludes_word'],
        accept_index=self.config['accept_index'],
        cancel_index=self.config['cancel_index'],
        end_index=self.config['end_index'],
        greet_index=self.config['greet_index'],
        info_question_index=self.config['info_question_index'],
        insist_index=self.config['insist_index'],
        slot_question_index=self.config['slot_question_index'],
        utterance_mask=self.config['utterance_mask'],
        pad_utterance_mask=self.config['pad_utterance_mask'],
        last_utterance_mask=self.config['last_utterance_mask'],
        mask_index=self.config['mask_index'])
    test_data = data.pad_dialogs(test_dialogs, self.config['max_dialog_size'],
                                 self.config['max_utterance_size'])
    raw_test_labels = data.one_hot_string_encoding(self.data['test_labels'],
                                                   self.config['class_map'])
    self.test_labels = data.pad_one_hot_labels(raw_test_labels,
                                               self.config['max_dialog_size'],
                                               self.config['class_map'])
    self.test_ds = data.list_to_dataset(test_data[0], self.test_labels[0],
                                        self.config['shuffle_test'],
                                        self.config['batch_size'])

  def check_greet(self, predictions, mask, class_map):
    for dialog_pred, dialog_mask in zip(predictions, mask):
      first = True
      for utterance_pred, utterance_mask in zip(dialog_pred, dialog_mask):
        if first or utterance_mask == 0:
          first = False
          continue
        if utterance_pred == class_map['greet']:
          return False

    return True

  def test_psl_rule_1_run_model(self):
    rule_weights = (1.0,)
    rule_names = ('rule_1',)
    psl_constraints = model.PSLModelMultiWoZ(
        rule_weights, rule_names, config=self.config)

    constrained_model = test_util.build_constrained_model(
        [self.config['max_dialog_size'], self.config['max_utterance_size']])
    constrained_model.fit(self.train_ds, epochs=self.config['train_epochs'])

    logits = eval_model.evaluate_constrained_model(constrained_model,
                                                   self.test_ds,
                                                   psl_constraints)
    predictions = tf.math.argmax(logits[0], axis=-1)
    result = self.check_greet(predictions, self.test_labels[1],
                              self.config['class_map'])
    self.assertTrue(result)

  def test_psl_rule_1(self):
    rule_weights = (1.0,)
    rule_names = ('rule_1',)
    psl_constraints = model.PSLModelMultiWoZ(
        rule_weights, rule_names, config=self.config)
    logits = test_util.LOGITS

    loss = psl_constraints.rule_1(logits=tf.constant(logits))
    self.assertEqual(loss, 1.4)

  def test_psl_rule_2_run_model(self):
    rule_weights = (10.0,)
    rule_names = ('rule_2',)
    psl_constraints = model.PSLModelMultiWoZ(
        rule_weights, rule_names, config=self.config)

    constrained_model = test_util.build_constrained_model(
        [self.config['max_dialog_size'], self.config['max_utterance_size']])
    constrained_model.fit(self.train_ds, epochs=self.config['train_epochs'])

    logits = eval_model.evaluate_constrained_model(constrained_model,
                                                   self.test_ds,
                                                   psl_constraints)
    predictions = tf.math.argmax(logits[0], axis=-1)
    self.assertEqual(predictions[2][0], self.config['class_map']['greet'])
    self.assertEqual(predictions[3][0], self.config['class_map']['greet'])

  def test_psl_rule_2(self):
    rule_weights = (1.0,)
    rule_names = ('rule_2',)
    psl_constraints = model.PSLModelMultiWoZ(
        rule_weights, rule_names, config=self.config)
    logits = test_util.LOGITS

    loss = psl_constraints.rule_2(
        logits=tf.constant(logits), data=test_util.FEATURES)
    self.assertEqual(loss, 0.6)

  def test_psl_rule_3_run_model(self):
    rule_weights = (1.0,)
    rule_names = ('rule_3',)
    psl_constraints = model.PSLModelMultiWoZ(
        rule_weights, rule_names, config=self.config)

    constrained_model = test_util.build_constrained_model(
        [self.config['max_dialog_size'], self.config['max_utterance_size']])
    constrained_model.fit(self.train_ds, epochs=self.config['train_epochs'])

    logits = eval_model.evaluate_constrained_model(constrained_model,
                                                   self.test_ds,
                                                   psl_constraints)
    predictions = tf.math.argmax(logits[0], axis=-1)
    self.assertEqual(predictions[0][0],
                     self.config['class_map']['init_request'])
    self.assertEqual(predictions[1][0],
                     self.config['class_map']['init_request'])

  def test_psl_rule_3(self):
    rule_weights = (1.0,)
    rule_names = ('rule_3',)
    psl_constraints = model.PSLModelMultiWoZ(
        rule_weights, rule_names, config=self.config)
    logits = test_util.LOGITS

    loss = psl_constraints.rule_3(
        logits=tf.constant(logits), data=test_util.FEATURES)
    self.assertEqual(loss, 0.8)

  def test_psl_rule_4_run_model(self):
    rule_weights = (1.0,)
    rule_names = ('rule_4',)
    psl_constraints = model.PSLModelMultiWoZ(
        rule_weights, rule_names, config=self.config)

    constrained_model = test_util.build_constrained_model(
        [self.config['max_dialog_size'], self.config['max_utterance_size']])
    constrained_model.fit(self.train_ds, epochs=self.config['train_epochs'])

    logits = eval_model.evaluate_constrained_model(constrained_model,
                                                   self.test_ds,
                                                   psl_constraints)
    predictions = tf.math.argmax(logits[0], axis=-1)
    self.assertEqual(predictions[1][1],
                     self.config['class_map']['second_request'])
    self.assertEqual(predictions[2][1],
                     self.config['class_map']['second_request'])

  def test_psl_rule_4(self):
    rule_weights = (1.0,)
    rule_names = ('rule_4',)
    psl_constraints = model.PSLModelMultiWoZ(
        rule_weights, rule_names, config=self.config)
    logits = test_util.LOGITS

    loss = psl_constraints.rule_4(
        logits=tf.constant(logits), data=test_util.FEATURES)
    self.assertNear(loss, 1.8, err=1e-6)

  def test_psl_rule_5_run_model(self):
    rule_weights = (1.0,)
    rule_names = ('rule_5',)
    psl_constraints = model.PSLModelMultiWoZ(
        rule_weights, rule_names, config=self.config)

    constrained_model = test_util.build_constrained_model(
        [self.config['max_dialog_size'], self.config['max_utterance_size']])
    constrained_model.fit(self.train_ds, epochs=self.config['train_epochs'])

    logits = eval_model.evaluate_constrained_model(constrained_model,
                                                   self.test_ds,
                                                   psl_constraints)
    predictions = tf.math.argmax(logits[0], axis=-1)
    self.assertNotEqual(predictions[1][1],
                        self.config['class_map']['init_request'])
    self.assertNotEqual(predictions[2][1],
                        self.config['class_map']['init_request'])

  def test_psl_rule_5(self):
    rule_weights = (1.0,)
    rule_names = ('rule_5',)
    psl_constraints = model.PSLModelMultiWoZ(
        rule_weights, rule_names, config=self.config)
    logits = test_util.LOGITS

    loss = psl_constraints.rule_5(
        logits=tf.constant(logits), data=test_util.FEATURES)
    self.assertNear(loss, 1.4, err=1e-6)

  def test_psl_rule_6_run_model(self):
    rule_weights = (1.0,)
    rule_names = ('rule_6',)
    psl_constraints = model.PSLModelMultiWoZ(
        rule_weights, rule_names, config=self.config)

    constrained_model = test_util.build_constrained_model(
        [self.config['max_dialog_size'], self.config['max_utterance_size']])
    constrained_model.fit(self.train_ds, epochs=self.config['train_epochs'])

    logits = eval_model.evaluate_constrained_model(constrained_model,
                                                   self.test_ds,
                                                   psl_constraints)
    predictions = tf.math.argmax(logits[0], axis=-1)
    self.assertNotEqual(predictions[1][0], self.config['class_map']['greet'])
    self.assertNotEqual(predictions[2][0], self.config['class_map']['greet'])

  def test_psl_rule_6(self):
    rule_weights = (1.0,)
    rule_names = ('rule_6',)
    psl_constraints = model.PSLModelMultiWoZ(
        rule_weights, rule_names, config=self.config)
    logits = test_util.LOGITS

    loss = psl_constraints.rule_6(
        logits=tf.constant(logits), data=test_util.FEATURES)
    self.assertNear(loss, 1.4, err=1e-6)

  def test_psl_rule_7_run_model(self):
    rule_weights = (1.0,)
    rule_names = ('rule_7',)
    psl_constraints = model.PSLModelMultiWoZ(
        rule_weights, rule_names, config=self.config)

    constrained_model = test_util.build_constrained_model(
        [self.config['max_dialog_size'], self.config['max_utterance_size']])
    constrained_model.fit(self.train_ds, epochs=self.config['train_epochs'])

    logits = eval_model.evaluate_constrained_model(constrained_model,
                                                   self.test_ds,
                                                   psl_constraints)
    predictions = tf.math.argmax(logits[0], axis=-1)
    self.assertEqual(predictions[1][2], self.config['class_map']['end'])
    self.assertEqual(predictions[2][3], self.config['class_map']['end'])

  def test_psl_rule_7(self):
    rule_weights = (1.0,)
    rule_names = ('rule_7',)
    psl_constraints = model.PSLModelMultiWoZ(
        rule_weights, rule_names, config=self.config)
    logits = test_util.LOGITS

    loss = psl_constraints.rule_7(
        logits=tf.constant(logits), data=test_util.FEATURES)
    self.assertNear(loss, 1.1, err=1e-6)

  def test_psl_rule_8(self):
    rule_weights = (1.0,)
    rule_names = ('rule_8',)
    psl_constraints = model.PSLModelMultiWoZ(
        rule_weights, rule_names, config=self.config)
    logits = test_util.LOGITS

    loss = psl_constraints.rule_8(
        logits=tf.constant(logits), data=test_util.FEATURES)
    self.assertNear(loss, 0.9, err=1e-6)

  def test_psl_rule_9(self):
    rule_weights = (1.0,)
    rule_names = ('rule_9',)
    psl_constraints = model.PSLModelMultiWoZ(
        rule_weights, rule_names, config=self.config)
    logits = test_util.LOGITS

    loss = psl_constraints.rule_9(
        logits=tf.constant(logits), data=test_util.FEATURES)
    self.assertNear(loss, 0.8, err=1e-6)

  def test_psl_rule_10(self):
    rule_weights = (1.0,)
    rule_names = ('rule_10',)
    psl_constraints = model.PSLModelMultiWoZ(
        rule_weights, rule_names, config=self.config)
    logits = test_util.LOGITS

    loss = psl_constraints.rule_10(
        logits=tf.constant(logits), data=test_util.FEATURES)
    self.assertNear(loss, 0.3, err=1e-6)

  def test_psl_rule_11(self):
    rule_weights = (1.0,)
    rule_names = ('rule_11',)
    psl_constraints = model.PSLModelMultiWoZ(
        rule_weights, rule_names, config=self.config)
    logits = test_util.LOGITS

    loss = psl_constraints.rule_11(
        logits=tf.constant(logits), data=test_util.FEATURES)
    self.assertNear(loss, 0.7, err=1e-6)

  def test_psl_rule_12(self):
    rule_weights = (1.0,)
    rule_names = ('rule_12',)
    psl_constraints = model.PSLModelMultiWoZ(
        rule_weights, rule_names, config=self.config)
    logits = test_util.LOGITS

    loss = psl_constraints.rule_12(
        logits=tf.constant(logits), data=test_util.FEATURES)
    self.assertNear(loss, 0.1, err=1e-6)

if __name__ == '__main__':
  tf.test.main()

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

"""Tests for psl_utils."""

from typing import List
import tensorflow as tf
import tensorflow_datasets as tfds
import psl_model  # local file import from experimental.language_structure.psl
import psl_utils  # local file import from experimental.language_structure.vrnn


class _PSLTestModel(psl_model.PSLModel):

  def __init__(self):
    super(_PSLTestModel, self).__init__(rule_weights=[], rule_names=[])

  def compute_loss_per_rule(self, data: tf.Tensor,
                            logits: tf.Tensor) -> List[float]:
    return [tf.reduce_sum(logits - data)]

  def compute_loss(self, data: tf.Tensor, logits: tf.Tensor) -> float:
    return sum(self.compute_loss_per_rule(data, logits))


class PslUtilsTest(tfds.testing.TestCase):

  def test_multiwoz_synth_psl_feature_mixin(self):
    inputs = [{
        'input_word_ids': tf.constant([[[1, 2, 0], [0, 0, 0]]])
    }, {
        'input_word_ids': tf.constant([[[2, 1, 0], [0, 0, 0]]])
    }, {
        'input_word_ids': tf.constant([[[1, 2, 0], [0, 0, 0]]])
    }, {
        'input_word_ids': tf.constant([[[2, 1, 0], [0, 0, 0]]])
    },
              tf.constant([3]),
              tf.constant([4]),
              tf.constant([5]),
              tf.constant([6]),
              tf.constant([7])]
    fn = lambda _: inputs
    dataset = 'multiwoz_synth'
    config = {
        'accept_words': ['yes'],
        'cancel_words': [],
        'end_words': [],
        'greet_words': ['hello'],
        'info_question_words': [],
        'insist_words': [],
        'slot_question_words': [],
        'includes_word': -1,
        'excludes_word': -2,
        'utterance_mask': -3,
        'last_utterance_mask': -4,
        'pad_utterance_mask': -5,
    }
    vocab = ['<pad>', 'yes', 'hello']

    mixin_fn = psl_utils.psl_feature_mixin(fn, dataset, config, vocab)
    outputs = mixin_fn(inputs)

    self.assertLen(outputs, len(inputs) + 1)
    for i in range(len(inputs)):
      self.assertAllEqual(outputs[i], inputs[i])
    self.assertAllEqual(
        outputs[-1],
        tf.constant([[[-4, -1, -2, -2, -1, -2, -2, -2],
                      [-5, -2, -2, -2, -2, -2, -2, -2]]]))

  def test_dstc_synth_psl_feature_mixin(self):
    inputs = [{
        'input_word_ids': tf.constant([[[1, 0], [0, 0]]])
    }, {
        'input_word_ids': tf.constant([[[2, 1], [0, 0]]])
    }, {
        'input_word_ids': tf.constant([[[1, 0], [0, 0]]])
    }, {
        'input_word_ids': tf.constant([[[2, 1], [0, 0]]])
    },
              tf.constant([3]),
              tf.constant([4]),
              tf.constant([5]),
              tf.constant([6]),
              tf.constant([7])]
    fn = lambda _: inputs
    dataset = 'sgd_synth'
    config = {
        'num_batches': 1,
        'batch_size': 1,
        'max_dialog_size': 1,
        'max_utterance_size': 2,
        'num_labels': 39,
        'includes_word': -1,
        'excludes_word': -2,
        'utterance_mask': -3,
        'last_utterance_mask': -4,
        'pad_utterance_mask': -5,
        'mask_index': 0,
        'state_transitions': [[1, 3]],
        'words': {
            '1': {
                'usr': {
                    'index': 1,
                    'words': ['yes'],
                },
                'sys': {
                    'index': 2,
                    'words': [],
                },
            },
            '2': {
                'usr': {
                    'index': 3,
                    'words': [],
                },
                'sys': {
                    'index': 4,
                    'words': ['hello'],
                },
            },
        },
    }

    vocab = ['<pad>', 'yes', 'hello']

    mixin_fn = psl_utils.psl_feature_mixin(fn, dataset, config, vocab)
    outputs = mixin_fn(inputs)

    self.assertLen(outputs, len(inputs) + 1)
    for i in range(len(inputs)):
      self.assertAllEqual(outputs[i], inputs[i])
    self.assertAllEqual(
        outputs[-1], tf.constant([[[-4, -1, -2, -2, -1], [-5, -2, -2, -2,
                                                          -2]]]))

  def test_update_logits(self):
    model = tf.keras.Sequential(layers=[
        tf.keras.Input(shape=(1,)),
        tf.keras.layers.Dense(
            units=1, kernel_initializer='ones', bias_initializer='zeros')
    ])
    optimizer = tf.keras.optimizers.SGD(learning_rate=.5)
    model_inputs = tf.constant([1])
    weights = [tf.identity(weight) for weight in model.weights]

    logits = psl_utils.update_logits(
        model,
        optimizer,
        model_inputs,
        get_logits_fn=lambda x: x,
        psl_constraint=_PSLTestModel(),
        psl_inputs=tf.constant([0], dtype=tf.float32),
        grad_steps=2,
        alpha=1)

    self.assertEqual(len(model.weights), len(weights))
    for w1, w2 in zip(model.weights, weights):
      self.assertAllEqual(w1, w2)
    self.assertAllEqual(logits, tf.constant([[0.]]))


if __name__ == '__main__':
  tfds.testing.test_main()

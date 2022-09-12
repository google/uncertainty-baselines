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

"""Tests for utils."""

import tensorflow as tf
from vrnn import utils  # local file import from experimental.language_structure


class UtilsTest(tf.test.TestCase):

  def test_gumbel_softmax_sample_shape(self):
    sampler = utils.GumbelSoftmaxSampler(temperature=0.5)
    logits = tf.ones(shape=[5, 10])
    samples = sampler(logits)
    self.assertEqual([5, 10], samples.shape.as_list())

  def test_get_last_step(self):
    batch_size = 5
    hidden_size = 4
    inputs = tf.tile(
        tf.reshape(tf.range(1, 11), [1, 10, 1]), [batch_size, 1, hidden_size])
    seqlen = tf.constant([2, 3, 10, 1, 0], dtype=tf.int32)
    last_step = utils.get_last_step(inputs, seqlen)
    expected = tf.tile(tf.expand_dims(seqlen, axis=1), [1, hidden_size])
    self.assertAllEqual(last_step, expected)

  def test_to_one_hot(self):
    inputs = tf.constant([[0.6, 0.3, 0.1], [0.1, 0.8, 0.1]])
    expected = tf.constant([[1, 0, 0], [0, 1, 0]])
    self.assertAllEqual(utils.to_one_hot(inputs), expected)

  def test_to_one_hot_tie_inputs(self):
    inputs = tf.constant([[0.5, 0.5], [0.5, 0.5]])
    expected = tf.constant([[1, 0], [1, 0]])
    self.assertAllEqual(utils.to_one_hot(inputs), expected)

  def test_mlp_with_final_activitation(self):
    output_sizes = [5, 6]
    final_activation = tf.keras.layers.ReLU()
    test_model = utils.MLP(
        output_sizes=output_sizes, final_activation=final_activation)
    input_tensor = tf.keras.Input(shape=(8))
    output_tensor = test_model(input_tensor)

    expected_output_shape = [None, 6]
    self.assertEqual(expected_output_shape, output_tensor.shape.as_list())
    self.assertEqual(tf.float32, output_tensor.dtype)

  def test_sequential_word_loss_shape(self):
    max_dialog_length = 2
    y_true = tf.keras.Input(shape=(max_dialog_length, None))
    y_pred = tf.keras.Input(shape=(max_dialog_length, None, None))
    loss_fn = utils.SequentialWordLoss()
    loss = loss_fn(y_true=y_true, y_pred=y_pred)
    self.assertEqual([None, max_dialog_length, None], loss.shape.as_list())

  def test_sequential_word_loss(self):
    y_true = tf.constant([[1, 2], [1, 0]])
    y_pred = tf.constant([[[0, 0, 1], [1, 0, 0]], [[0, 1, 0], [0, 1, 0]]],
                         dtype=tf.float32)
    loss_fn = utils.SequentialWordLoss()
    unit_loss = tf.keras.losses.SparseCategoricalCrossentropy()(
        y_true=tf.constant([1]),
        y_pred=tf.constant([0, 0, 1], dtype=tf.float32)).numpy()
    expected = tf.constant([[unit_loss, unit_loss], [0, 0]])
    self.assertAllClose(
        loss_fn(y_true, y_pred, sample_weight=tf.sign(y_true)), expected)

  def test_sequential_word_loss_with_word_weights(self):
    y_true = tf.constant([[1, 2], [0, 0]])
    y_pred = tf.constant([[[0, 0, 1], [1, 0, 0]], [[0, 1, 0], [0, 1, 0]]],
                         dtype=tf.float32)
    word_weights = [0, 1, 2]
    loss_fn = utils.SequentialWordLoss(word_weights=word_weights)
    unit_loss = tf.keras.losses.SparseCategoricalCrossentropy()(
        y_true=tf.constant([1]),
        y_pred=tf.constant([0, 0, 1], dtype=tf.float32)).numpy()
    expected = tf.constant([[unit_loss, 2 * unit_loss], [0, 0]])
    self.assertAllClose(
        loss_fn(y_true, y_pred, sample_weight=tf.sign(y_true)), expected)

  def test_kl_loss_shape(self):
    max_dialog_length = 2
    p = tf.keras.Input(shape=(max_dialog_length, None, None))
    q = tf.keras.Input(shape=(max_dialog_length, None, None))
    loss_fn = utils.KlLoss(bpr=True, reduction=tf.keras.losses.Reduction.NONE)
    loss = loss_fn(p, q)
    self.assertEqual([max_dialog_length, None], loss.shape.as_list())

  def test_kl_loss(self):
    p = tf.constant([[[0, 1], [0, 1]], [[1, 0], [0, 1]]], dtype=tf.float32)
    q = tf.constant([[[1, 0], [0, 1]], [[0, 1], [0, 1]]], dtype=tf.float32)
    loss_fn = utils.KlLoss(bpr=False, reduction=tf.keras.losses.Reduction.NONE)
    unit_loss = tf.keras.losses.KLDivergence()(
        y_true=tf.constant([0, 1], dtype=tf.float32),
        y_pred=tf.constant([1, 0], dtype=tf.float32)).numpy()
    expected = tf.constant([[unit_loss, 0], [unit_loss, 0]])
    self.assertAllEqual(loss_fn(p, q), expected)

  def test_kl_loss_with_bpr(self):
    p = tf.constant([[[0, 1], [1, 0]], [[0, 1], [0, 1]]], dtype=tf.float32)
    q = tf.constant([[[1, 0], [0, 1]], [[1, 0], [1, 0]]], dtype=tf.float32)
    loss_fn = utils.KlLoss(bpr=True, reduction=tf.keras.losses.Reduction.NONE)
    unit_loss = tf.keras.losses.KLDivergence()(
        y_true=tf.constant([0, 1], dtype=tf.float32),
        y_pred=tf.constant([1, 0], dtype=tf.float32)).numpy()
    expected = tf.constant([unit_loss * 2, 0])
    self.assertAllEqual(loss_fn(p, q), expected)

  def test_bow_loss_shape(self):
    max_dialog_length = 2
    y_true = tf.keras.Input(shape=(max_dialog_length, None))
    y_pred = tf.keras.Input(shape=(max_dialog_length, None))
    loss_fn = utils.BowLoss(sequence_axis=2)
    loss = loss_fn(y_true=y_true, y_pred=y_pred)
    self.assertEqual([None, max_dialog_length, None], loss.shape.as_list())

  def test_bow_loss(self):
    y_true = tf.constant([[1, 2], [1, 0]])
    y_pred = tf.constant([[0, 1, 0], [0, 1, 0]], dtype=tf.float32)
    loss_fn = utils.BowLoss()
    unit_loss = tf.keras.losses.SparseCategoricalCrossentropy()(
        y_true=tf.constant([1]),
        y_pred=tf.constant([0, 0, 1], dtype=tf.float32)).numpy()
    expected = tf.constant([[0, unit_loss], [0, 0]])
    self.assertAllClose(
        loss_fn(y_true, y_pred, sample_weight=tf.sign(y_true)), expected)

  def test_create_mask(self):
    inputs = tf.constant([[1, 2], [2, 1], [3, 2]])
    masking_prob = {1: 1., 2: 0., 3: 0.8}
    self.assertAllEqual(
        tf.constant([[1, 0], [0, 1], [0, 0]]),
        utils.create_mask(inputs, masking_prob, seed=1))
    self.assertAllEqual(
        tf.constant([[1, 0], [0, 1], [1, 0]]),
        utils.create_mask(inputs, masking_prob, seed=2))

  def test_value_in_tensor(self):
    inputs = tf.constant([[1, 2], [2, 1], [3, 2]])
    tensor = tf.constant([1, 1, 2])
    expected = tf.constant([[True, True], [True, True], [False, True]])
    self.assertAllEqual(expected, utils.value_in_tensor(inputs, tensor))

  def test_bert_preprocessor(self):
    tfhub_url = 'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3'
    max_seq_length = 10
    batch_size = 12
    preprocessor = utils.BertPreprocessor(tfhub_url, max_seq_length)

    self.assertEqual(preprocessor.vocab_size, 30522)

    inputs = [
        tf.keras.layers.Input(shape=(), dtype=tf.string, batch_size=batch_size)
        for _ in range(2)
    ]
    outputs = preprocessor(inputs)

    self.assertLen(outputs, 2)
    for output in outputs:
      for key in ['input_word_ids', 'input_type_ids', 'input_mask']:
        self.assertEqual(output[key].shape.as_list(), [batch_size, 10])

    outputs = preprocessor(inputs, concat=True)

    for key in ['input_word_ids', 'input_type_ids', 'input_mask']:
      self.assertEqual(outputs[key].shape.as_list(), [batch_size, 10])

  def test_adjusted_mutual_info(self):
    a = tf.constant([[1, 2, 0], [2, 1, 0]])
    b = tf.constant([[1, 2, 1], [2, 1, 1]])

    self.assertEqual(utils.adjusted_mutual_info(a, b), 1.)

  def test_cluster_purity(self):
    a = tf.constant([[1, 0, 0], [1, 1, 0]])
    b = tf.constant([[1, 2, 3], [1, 1, 2]])

    self.assertEqual(utils.cluster_purity(a, b), 1.)

  def test_create_rebalanced_sample_weights(self):
    labels = tf.constant([[1, 2, 3], [1, 4, 0]])
    sample_weights = utils.create_rebalanced_sample_weights(labels)
    self.assertAllEqual(sample_weights,
                        tf.constant([[0.75, 1.5, 1.5], [0.75, 1.5, 0.]]))


if __name__ == '__main__':
  tf.test.main()

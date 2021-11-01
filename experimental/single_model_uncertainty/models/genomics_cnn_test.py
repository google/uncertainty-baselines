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

"""Tests for uncertainty_baselines.models.genomicscnn."""

from absl.testing import parameterized

import tensorflow as tf
import uncertainty_baselines as ub
import genomics_cnn  # local file import


class GenomicsCNNTest(tf.test.TestCase, parameterized.TestCase):

  def setUp(self):
    super().setUp()
    batch_size = 32
    num_classes = 10
    num_motifs = 16
    len_motifs = 20
    num_denses = 6
    seq_len = 250

    # generate random input data composed by {0, 1, 2, 3}
    rand_nums = tf.random.categorical(
        tf.math.log([[0.2, 0.3, 0.3, 0.2]]), batch_size * seq_len)
    self.rand_data = tf.reshape(rand_nums, (batch_size, seq_len))
    self.rand_labels = tf.math.round(tf.random.uniform([batch_size]))

    self.params = {
        'batch_size': batch_size,
        'num_classes': num_classes,
        'num_motifs': num_motifs,
        'len_motifs': len_motifs,
        'num_denses': num_denses,
        'seq_len': seq_len,
        'optimizer_name': 'adam',
    }

  def testCreateModel(self):
    model = genomics_cnn.genomics_cnn(
        batch_size=self.params['batch_size'],
        len_seqs=self.params['seq_len'],
        num_classes=self.params['num_classes'],
        num_motifs=self.params['num_motifs'],
        len_motifs=self.params['len_motifs'],
        num_denses=self.params['num_denses'])

    logits = model(self.rand_data)
    loss = tf.reduce_mean(
        tf.keras.losses.sparse_categorical_crossentropy(
            y_true=self.rand_labels, y_pred=logits, from_logits=True))
    self.assertGreater(loss, 0)

  @parameterized.named_parameters(('with_weight_decay', 1e-4),
                                  ('without_weight_decay', 0.0))
  def testCreateOptimizer(self, weight_decay):
    model = genomics_cnn.genomics_cnn(
        batch_size=self.params['batch_size'],
        len_seqs=self.params['seq_len'],
        num_classes=self.params['num_classes'],
        num_motifs=self.params['num_motifs'],
        len_motifs=self.params['len_motifs'],
        num_denses=self.params['num_denses'])
    optimizer = ub.optimizers.get(
        optimizer_name=self.params['optimizer_name'],
        learning_rate=0.001,
        weight_decay=weight_decay,
        model=model)

    with tf.GradientTape() as tape:
      logits = model(self.rand_data)
      loss = tf.reduce_mean(
          tf.keras.losses.sparse_categorical_crossentropy(
              y_true=self.rand_labels, y_pred=logits, from_logits=True))
      grads = tape.gradient(loss, model.trainable_variables)
      optimizer.apply_gradients(list(zip(grads, model.trainable_variables)))

  @parameterized.named_parameters(
      # test onehot vs embedding
      ('Onehot', True, False, False, False),
      ('Embedding', False, False, False, False),
      # test mc dropout
      ('Onehot_mcdropout', True, True, False, False),
      # test sn
      ('Onehot_sn', True, False, True, False),
      # test gp
      ('Onehot_gp', True, False, False, True),
      # test sn+gp
      ('Onehot_sn_gp', True, False, True, True),
      # test mcdropout+sn+gp
      ('Onehot_mcdropout_sn_gp', True, True, True, True),
  )
  def testCreateDifferentModels(self, one_hot, use_mc_dropout, use_spec_norm,
                                use_gp_layer):

    if use_spec_norm:
      spec_norm_hparams = {'spec_norm_bound': 6.0, 'spec_norm_iteration': 1}
    else:
      spec_norm_hparams = None

    if use_gp_layer:
      gp_layer_hparams = {
          'gp_input_dim': 128,
          'gp_hidden_dim': 1024,
          'gp_scale': 2.0,
          'gp_bias': 0.0,
          'gp_input_normalization': True,
          'gp_cov_discount_factor': 0.999,
          'gp_cov_ridge_penalty': 1e-3,
      }
    else:
      gp_layer_hparams = None

    model = genomics_cnn.genomics_cnn(
        batch_size=self.params['batch_size'],
        len_seqs=self.params['seq_len'],
        num_classes=self.params['num_classes'],
        num_motifs=self.params['num_motifs'],
        len_motifs=self.params['len_motifs'],
        num_denses=self.params['num_denses'],
        one_hot=one_hot,
        use_mc_dropout=use_mc_dropout,
        spec_norm_hparams=spec_norm_hparams,
        gp_layer_hparams=gp_layer_hparams)

    logits = model(self.rand_data)
    if isinstance(logits, (tuple, list)):
      logits, covmat = logits
      self.assertEqual(covmat.shape,
                       (self.params['batch_size'], self.params['batch_size']))
    self.assertEqual(logits.shape,
                     (self.params['batch_size'], self.params['num_classes']))


if __name__ == '__main__':
  tf.test.main()

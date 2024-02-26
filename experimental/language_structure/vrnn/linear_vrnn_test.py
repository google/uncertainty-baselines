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

"""Tests for linear_vae_cell."""

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from psl import psl_model_multiwoz  # local file import from experimental.language_structure
from psl import psl_model_multiwoz_test_util as psl_test_util  # local file import from experimental.language_structure
from vrnn import linear_vrnn  # local file import from experimental.language_structure
from vrnn import model_config  # local file import from experimental.language_structure


class LinearVrnnTest(tfds.testing.TestCase):

  def _create_test_vrnn_config(self):
    config = model_config.vanilla_linear_vrnn_config(
        vae_cell=model_config.vanilla_linear_vae_cell_config(
            encoder_embedding=model_config.embedding_config(
                vocab_size=5,
                embed_size=2,
            ),
            decoder_embedding=model_config.embedding_config(
                vocab_size=5,
                embed_size=2,
            ),
            max_seq_length=3,
            encoder_hidden_size=2,
            encoder_projection_sizes=(2,),
            sampler_post_processor_output_sizes=(2,),
            num_states=3),
        max_dialog_length=3,
        num_states=3,
        vocab_size=5,
        prior_latent_state_updater_hidden_size=(6,),
        bow_hidden_sizes=(4,))
    return config

  @tfds.testing.run_in_graph_and_eager_modes
  def test_vanilla_linear_vrnn_shape(self):
    config = self._create_test_vrnn_config()
    test_model = linear_vrnn.VanillaLinearVRNN(config)

    inputs = [{
        'input_word_ids':
            tf.keras.Input(
                shape=(config.max_dialog_length, None), dtype=tf.int32),
        'input_mask':
            tf.keras.Input(
                shape=(config.max_dialog_length, None), dtype=tf.int32),
    }] * 4 + [
        tf.keras.Input(shape=(config.num_states,)),
        tf.keras.Input(shape=(config.num_states,)),
        tf.keras.Input(shape=(config.max_dialog_length,), dtype=tf.int32),
        tf.keras.Input(shape=(config.max_dialog_length,), dtype=tf.int32),
    ]
    outputs = test_model(inputs)
    self.assertLen(outputs, 8)
    (latent_state, latent_samples, p_z_logits, q_z_logits, decoder_outputs_1,
     decoder_outputs_2, bow_logits_1, bow_logits_2) = outputs

    self.assertEqual([
        None, config.max_dialog_length,
        config.vae_cell.encoder_projection_sizes[-1]
    ], latent_state.shape.as_list())

    for output in [latent_samples, p_z_logits, q_z_logits]:
      self.assertEqual([None, config.max_dialog_length, config.num_states],
                       output.shape.as_list())

    for output in [bow_logits_1, bow_logits_2]:
      self.assertEqual([
          None, config.max_dialog_length,
          config.vae_cell.decoder_embedding.vocab_size
      ], output.shape.as_list())

    for output in [decoder_outputs_1, decoder_outputs_2]:
      self.assertEqual([
          None, config.max_dialog_length, None,
          config.vae_cell.decoder_embedding.vocab_size
      ], output.shape.as_list())

  @tfds.testing.run_in_graph_and_eager_modes
  def test_loss_shape(self):
    config = self._create_test_vrnn_config()
    batch_size = 2
    labels_1 = tf.keras.Input(
        shape=(config.max_dialog_length, None), batch_size=batch_size)
    labels_2 = tf.keras.Input(
        shape=(config.max_dialog_length, None), batch_size=batch_size)
    labels_1_mask = tf.keras.Input(
        shape=(config.max_dialog_length, None), batch_size=batch_size)
    labels_2_mask = tf.keras.Input(
        shape=(config.max_dialog_length, None), batch_size=batch_size)
    model_outputs = [
        tf.keras.Input(
            shape=(config.max_dialog_length, None), batch_size=batch_size),
        tf.keras.Input(
            shape=(config.max_dialog_length, None), batch_size=batch_size),
        tf.keras.Input(
            shape=(config.max_dialog_length, config.num_states),
            batch_size=batch_size),
        tf.keras.Input(
            shape=(config.max_dialog_length, config.num_states),
            batch_size=batch_size),
        tf.keras.Input(
            shape=(config.max_dialog_length, config.vae_cell.max_seq_length,
                   None),
            batch_size=batch_size),
        tf.keras.Input(
            shape=(config.max_dialog_length, config.vae_cell.max_seq_length,
                   None),
            batch_size=batch_size),
        tf.keras.Input(
            shape=(config.max_dialog_length, None), batch_size=batch_size),
        tf.keras.Input(
            shape=(config.max_dialog_length, None), batch_size=batch_size),
    ]
    latent_label_id = tf.keras.Input(
        shape=(config.max_dialog_length,),
        batch_size=batch_size,
        dtype=tf.int32)
    latent_label_mask = tf.keras.Input(
        shape=(config.max_dialog_length,), batch_size=batch_size)
    word_weights = np.ones((config.vae_cell.decoder_embedding.vocab_size),
                           dtype=np.float32)

    psl_config = psl_test_util.TEST_MULTIWOZ_CONFIG
    rule_weights = (1.0,)
    rule_names = ('rule_1',)
    psl_constraints = psl_model_multiwoz.PSLModelMultiWoZ(
        rule_weights, rule_names, config=psl_config)
    psl_inputs = tf.keras.Input(
        shape=(config.max_dialog_length, 8), batch_size=batch_size)
    psl_constraint_loss_weight = 0.1

    outputs = linear_vrnn.compute_loss(
        labels_1,
        labels_2,
        labels_1_mask,
        labels_2_mask,
        model_outputs,
        latent_label_id,
        latent_label_mask,
        word_weights,
        with_bpr=True,
        kl_loss_weight=0.5,
        with_bow=True,
        bow_loss_weight=0.3,
        num_latent_states=config.num_states,
        classification_loss_weight=0.8,
        psl_constraint_model=psl_constraints,
        psl_inputs=psl_inputs,
        psl_constraint_loss_weight=psl_constraint_loss_weight)

    self.assertLen(outputs, 8)
    for loss in outputs[:7]:
      self.assertEqual([], loss.shape.as_list())

    loss_per_rule = outputs[7]
    self.assertLen(loss_per_rule, len(rule_names))
    for loss in loss_per_rule:
      self.assertEqual([], loss.shape.as_list())


if __name__ == '__main__':
  tfds.testing.test_main()

# coding=utf-8
# Copyright 2023 The Uncertainty Baselines Authors.
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

"""VRNN.

## References:

[1]: Qiu Liang et al. Structured Attention for Unsupervised Dialogue Structure
     Induction.
     _arXiv preprint arXiv:2009.08552, 2020.
     https://arxiv.org/pdf/2009.08552.pdf
[2]: https://github.com/Liang-Qiu/SVRNN-dialogues
"""

from typing import Any, Dict, Optional, Sequence

import tensorflow as tf
from psl import psl_model  # local file import from experimental.language_structure
from vrnn import linear_vae_cell  # local file import from experimental.language_structure
from vrnn import model_config  # local file import from experimental.language_structure
from vrnn import utils  # local file import from experimental.language_structure


class _VRNN(tf.keras.Model):
  """VRNN base class.

  It iterates through a sequence of sentence pairs (with masks) and for each
  pair, it
    - runs vae_cell to encode->sample->decode inputs.
    - updates the latent state.
    - (optional) calculate BOW outputs.
  """

  def __init__(self,
               vae_cell,
               prior_latent_state_updater,
               num_states: int,
               max_dialog_length: int,
               with_direct_transition: bool,
               bow_layer_1: Optional[Any] = None,
               bow_layer_2: Optional[Any] = None):
    """VRNN base class constructor.

    Args:
      vae_cell: the VAE model to process a sentence pair (e.g., one round of
        dialog).
      prior_latent_state_updater: the model to update the prior of latent state.
      num_states: the number of latent states.
      max_dialog_length: the maxium size of the sequence (e.g., the number of
        dialog rounds).
      with_direct_transition: whether to update the prior of latent state by the
        hidden state (DD-VRNN).
      bow_layer_1: the model to compute BOW logit for the first sentence.
      bow_layer_2: the model to compute BOW logit for the second sentence. The
        two layers must be both provided or both None.
    """
    super(_VRNN, self).__init__()
    if bow_layer_1 is None != bow_layer_2 is None:
      raise ValueError('Two BOW layers must be both provided or both None.')

    self._num_states = num_states
    self._is_tuple_state = vae_cell.is_tuple_state()
    self._max_dialog_length = max_dialog_length
    self._with_direct_transition = with_direct_transition

    self.vae_cell = vae_cell
    self.prior_latent_state_updater = prior_latent_state_updater
    self.bow_layer_1 = bow_layer_1
    self.bow_layer_2 = bow_layer_2

  def _verify_and_prepare_inputs(self, inputs: Sequence[Any]):
    if len(inputs) not in (6, 8):
      raise ValueError(
          'Expect inputs to be a sequence of length 6 (encoder_input_1, '
          'encoder_input_2, decoder_input_1, decoder_input_2, state, sample) or'
          ' 8 (encoder_input_1, encoder_input_2, decoder_input_1, '
          'decoder_input_2, state, sample, label_id, label_mask), found %s.' %
          len(inputs))

    (encoder_input_1, encoder_input_2, decoder_input_1, decoder_input_2, state,
     sample) = inputs[:6]
    if len(inputs) == 8:
      label_id, label_mask = inputs[6:]
      label = tf.one_hot(label_id, depth=self._num_states)

      label = self._split_sequence(label)
      label_mask = self._split_sequence(label_mask)
    else:
      label = None
      label_mask = None

    encoder_input_1 = self._split_sequence_dict(encoder_input_1)
    encoder_input_2 = self._split_sequence_dict(encoder_input_2)
    decoder_input_1 = self._split_sequence_dict(decoder_input_1)
    decoder_input_2 = self._split_sequence_dict(decoder_input_2)

    if self._is_tuple_state:
      state = (state, state)

    return (encoder_input_1, encoder_input_2, decoder_input_1, decoder_input_2,
            state, sample, label, label_mask)

  def _split_sequence(self, tensor: tf.Tensor) -> Sequence[tf.Tensor]:
    return tf.unstack(tensor, axis=1)

  def _merge_to_sequence(self, tensors: Sequence[tf.Tensor]) -> tf.Tensor:
    return tf.stack(tensors, axis=1)

  def _split_sequence_dict(
      self, inputs: Dict[str, tf.Tensor]) -> Sequence[Dict[str, tf.Tensor]]:
    """Splits the inputs dictionary to a sequence of dictionary."""
    outputs = []
    for key in inputs.keys():
      for i, value in enumerate(self._split_sequence(inputs[key])):
        if i >= len(outputs):
          outputs.append({})
        outputs[i][key] = value
    return outputs

  def call(self, inputs: Sequence[Any]):
    (encoder_input_1, encoder_input_2, decoder_input_1, decoder_input_2, state,
     sample, label, label_mask) = self._verify_and_prepare_inputs(inputs)

    with_bow = self.bow_layer_1 is not None
    latent_samples = []
    p_z_logits = []
    q_z_logits = []
    bow_logits_1 = []
    bow_logits_2 = []
    decoder_outputs_1 = []
    decoder_outputs_2 = []
    latent_states = []
    for i in range(self._max_dialog_length):
      if self._with_direct_transition:
        p_z_logit = self.prior_latent_state_updater(sample)
      else:
        hidden_state = state[0] if self._is_tuple_state else state
        p_z_logit = self.prior_latent_state_updater(hidden_state)

      vae_cell_inputs = [
          encoder_input_1[i], encoder_input_2[i], decoder_input_1[i],
          decoder_input_2[i], state
      ]
      if label is not None:
        vae_cell_inputs.extend([label[i], label_mask[i]])
      (q_z_logit, decoder_output_1, decoder_output_2, state, latent_state,
       decoder_initial_state, decoder_state_1, _, sample) = self.vae_cell(
           vae_cell_inputs, return_states=True, return_samples=True)

      if with_bow:
        bow_logits_1.append(self.bow_layer_1(decoder_initial_state))
        bow_logits_2.append(
            self.bow_layer_2(
                tf.concat([decoder_initial_state, decoder_state_1], axis=1)))

      sample = utils.to_one_hot(sample)
      latent_samples.append(sample)
      p_z_logits.append(p_z_logit)
      q_z_logits.append(q_z_logit)
      decoder_outputs_1.append(decoder_output_1)
      decoder_outputs_2.append(decoder_output_2)
      latent_states.append(latent_state)

    latent_samples = self._merge_to_sequence(latent_samples)
    p_z_logits = self._merge_to_sequence(p_z_logits)
    q_z_logits = self._merge_to_sequence(q_z_logits)
    decoder_outputs_1 = self._merge_to_sequence(decoder_outputs_1)
    decoder_outputs_2 = self._merge_to_sequence(decoder_outputs_2)
    latent_states = self._merge_to_sequence(latent_states)
    outputs = [
        latent_states, latent_samples, p_z_logits, q_z_logits,
        decoder_outputs_1, decoder_outputs_2
    ]
    if with_bow:
      bow_logits_1 = self._merge_to_sequence(bow_logits_1)
      bow_logits_2 = self._merge_to_sequence(bow_logits_2)
      outputs += [bow_logits_1, bow_logits_2]
    return outputs


class _MlpWithProjector(tf.keras.layers.Layer):
  """A layer adding a dense layer on top of utils.MLP."""

  def __init__(self,
               hidden_sizes: Sequence[int],
               output_size: int,
               dropout: float,
               final_activation: Optional[Any] = None):
    super(_MlpWithProjector, self).__init__()
    self.mlp = utils.MLP(
        hidden_sizes, dropout=dropout, final_activation=final_activation)
    self.projector = tf.keras.layers.Dense(output_size)

  def call(self, inputs):
    outputs = self.mlp(inputs)
    return self.projector(outputs)


class VanillaLinearVRNN(_VRNN):
  """Vanilla Linear VRNN model."""

  def __init__(self, config: model_config.VanillaLinearVRNNConfig):
    vae_cell = linear_vae_cell.VanillaLinearVAECell(config.vae_cell)
    prior_latent_state_updater = _MlpWithProjector(
        config.prior_latent_state_updater_hidden_size,
        config.num_states,
        dropout=0.)

    bow_layer_1 = bow_layer_2 = None
    if config.with_bow:
      bow_layer_kwargs = dict(
          hidden_sizes=config.bow_hidden_sizes,
          output_size=config.vae_cell.decoder_embedding.vocab_size,
          dropout=config.dropout,
          final_activation=tf.nn.tanh)
      bow_layer_1 = _MlpWithProjector(**bow_layer_kwargs)
      bow_layer_2 = _MlpWithProjector(**bow_layer_kwargs)

    super().__init__(
        vae_cell=vae_cell,
        prior_latent_state_updater=prior_latent_state_updater,
        num_states=config.num_states,
        max_dialog_length=config.max_dialog_length,
        with_direct_transition=config.with_direct_transition,
        bow_layer_1=bow_layer_1,
        bow_layer_2=bow_layer_2)


def compute_loss(labels_1: tf.Tensor,
                 labels_2: tf.Tensor,
                 labels_1_mask: tf.Tensor,
                 labels_2_mask: tf.Tensor,
                 model_outputs: Sequence[tf.Tensor],
                 latent_label_id: tf.Tensor,
                 latent_label_mask: tf.Tensor,
                 word_weights: Any,
                 with_bpr: bool,
                 kl_loss_weight: float,
                 with_bow: bool,
                 bow_loss_weight: float,
                 num_latent_states: int,
                 classification_loss_weight: float,
                 psl_constraint_model: Optional[psl_model.PSLModel] = None,
                 psl_inputs: Optional[tf.Tensor] = None,
                 psl_constraint_loss_weight: Optional[float] = 0):
  """Computes total loss and its components.

  Args:
    labels_1: targeted sequence 1 to reconstruct, of shape [batch_size,
      max_dialog_length, max_seq_length].
    labels_2: targeted sequence 2 to reconstruct, of shape [batch_size,
      max_dialog_length, max_seq_length].
    labels_1_mask: mask of `labels_1`, of shape [batch_size, max_dialog_length,
      max_seq_length].
    labels_2_mask: mask of `labels_2`, of shape [batch_size, max_dialog_length,
      max_seq_length].
    model_outputs: the outputs of VRNN.
    latent_label_id: the label id of the dialog latent state.
    latent_label_mask: the label mask of the dialog latent state.
    word_weights: of shape [vocab_size], the weights of each token, used to
      rescale loss.
    with_bpr: enable batch prior regularization when computing KL divergence.
      See utils.KlLoss for details.
    kl_loss_weight: the weight of KL loss when computing elbo.
    with_bow: enable bag-of-word loss.
    bow_loss_weight: the weight of bow loss when computing elbo.
    num_latent_states: number of latent states.
    classification_loss_weight: the weight of the latent label classification
      loss when computing the total loss.
    psl_constraint_model: the PSL contraint model to compute the constraint
      loss.
    psl_inputs: the input data for `psl_constraint_model`.
    psl_constraint_loss_weight: the weight of the PSL constraint loss when
      computing the total loss.

  Returns:
    total_loss: total loss.
    rc_loss: sequence reconstruction loss.
    kl_loss: KL divergence between the prior and posterior distribution.
    bow_loss: bag-of-word loss.
    classification_loss: dialog latent state classification loss.
    elbo: the evidence lower bound.
  """
  if with_bow:
    (_, _, p_z_logits, q_z_logits, outputs_1, outputs_2, bow_logits_1,
     bow_logits_2) = model_outputs
  else:
    (_, _, p_z_logits, q_z_logits, outputs_1, outputs_2) = model_outputs

  labels_1 = tf.cast(labels_1, dtype=tf.float32)
  labels_2 = tf.cast(labels_2, dtype=tf.float32)
  seq_length = tf.cast(
      tf.reduce_sum(labels_1_mask + labels_2_mask), dtype=tf.float32)

  # reconstruction loss
  rc_loss_fn = utils.SequentialWordLoss(
      word_weights=word_weights, from_logits=True)

  rc_loss = tf.reduce_sum(
      rc_loss_fn(labels_1, outputs_1, sample_weight=labels_1_mask) +
      rc_loss_fn(labels_2, outputs_2, sample_weight=labels_2_mask)) / seq_length

  # kl divergence
  kl_loss_fn = utils.KlLoss(
      with_bpr, from_logits=True, reduction=tf.keras.losses.Reduction.SUM)
  kl_loss = kl_loss_fn(p_z_logits, q_z_logits) / seq_length

  # bow loss
  bow_loss = 0
  if with_bow:
    bow_loss_fn = utils.BowLoss(
        sequence_axis=2, word_weights=word_weights, from_logits=True)
    bow_loss = tf.reduce_sum(
        bow_loss_fn(labels_1, bow_logits_1, sample_weight=labels_1_mask) +
        bow_loss_fn(labels_2, bow_logits_2, sample_weight=labels_2_mask)
    ) / seq_length

  elbo = rc_loss + kl_loss_weight * kl_loss + bow_loss_weight * bow_loss

  logits = get_logits(model_outputs)
  # classification loss
  classification_loss_function = tf.keras.losses.CategoricalCrossentropy(
      from_logits=True, reduction=tf.keras.losses.Reduction.SUM)
  latent_label = tf.one_hot(latent_label_id, depth=num_latent_states)
  classification_loss = classification_loss_function(
      latent_label, logits, sample_weight=latent_label_mask)

  if psl_constraint_model is None or psl_inputs is None:
    constraint_loss = 0
    constraint_loss_weight = 0
    constraint_loss_per_rule = None
  else:
    probability = tf.keras.activations.softmax(logits)
    constraint_loss_weight = psl_constraint_loss_weight
    psl_constraint_model.generate_predicates(psl_inputs)
    constraint_loss_per_rule = psl_constraint_model.compute_loss_per_rule(
        psl_inputs, probability)
    constraint_loss = sum(constraint_loss_per_rule)

  total_loss = (
      elbo + classification_loss_weight * classification_loss +
      constraint_loss_weight * constraint_loss)
  return (total_loss, rc_loss, kl_loss, bow_loss, classification_loss,
          constraint_loss, elbo, constraint_loss_per_rule)


def get_logits(model_outputs: Sequence[tf.Tensor]) -> tf.Tensor:
  return model_outputs[3]


def get_prediction(logits: tf.Tensor) -> tf.Tensor:
  """Gets the model prediction from the model outputs."""
  return tf.argmax(logits, axis=2, output_type=tf.int32)

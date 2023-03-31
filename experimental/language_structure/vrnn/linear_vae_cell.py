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

"""VAE Cell in VRNN.

VAE Cell is the core component of VRNN [1]. It's migrated from the Pytorch
version [2].

## References:

[1]: Qiu Liang et al. Structured Attention for Unsupervised Dialogue Structure
     Induction.
     _arXiv preprint arXiv:2009.08552, 2020.
     https://arxiv.org/pdf/2009.08552.pdf
[2]: https://github.com/Liang-Qiu/SVRNN-dialogues
"""

from typing import Any, Dict, Optional, Sequence, Union

import numpy as np
import tensorflow as tf
import bert_utils  # local file import from baselines.clinc_intent
from vrnn import model_config  # local file import from experimental.language_structure
from vrnn import utils  # local file import from experimental.language_structure

from official.nlp.bert import bert_models
from official.nlp.bert import configs

INPUT_ID_NAME = 'input_word_ids'
INPUT_MASK_NAME = 'input_mask'

_TensorMapList = Sequence[Dict[str, tf.Tensor]]


class _BERT(tf.keras.Model):
  """BERT model ."""

  def __init__(self, max_seq_length: int, bert_config: configs.BertConfig,
               trainable: bool):
    """BERT class constructor.

    Args:
      max_seq_length: the maximum input sequence length.
      bert_config: Configuration for a BERT model.
      trainable: whether the model is trainable.
    """

    super(_BERT, self).__init__()

    self.bert_model = bert_models.get_transformer_encoder(
        bert_config, max_seq_length)
    self._trainable = trainable
    self._vocab_size = bert_config.vocab_size

  def call(self,
           inputs: Dict[str, tf.Tensor],
           return_sequence: bool = True) -> tf.Tensor:

    sequence_output, cls_output = self.bert_model(inputs)
    if not self._trainable:
      sequence_output = tf.stop_gradient(sequence_output)
      cls_output = tf.stop_gradient(cls_output)
    if return_sequence:
      return sequence_output
    else:
      return cls_output

  @property
  def vocab_size(self):
    return self._vocab_size


class _Embedding(tf.keras.layers.Embedding):
  """Word embedding layer.

  A wrapper class of tf.keras.layers.Embedding. It receives a
  Dict[str, tf.Tensor] containing key ${input_id_key} and returns the embedding.
  """

  def __init__(self, input_id_key: str, vocab_size: int, embed_size: int,
               trainable: bool, **kwargs: Dict[str, Any]):
    super(_Embedding, self).__init__(vocab_size, embed_size, **kwargs)
    self._input_id_key = input_id_key
    self._trainable = trainable

    self._vocab_size = vocab_size

  def call(self, inputs: Dict[str, tf.Tensor]) -> tf.Tensor:
    outputs = super().call(inputs[self._input_id_key])
    if not self._trainable:
      outputs = tf.stop_gradient(outputs)
    return outputs

  @property
  def vocab_size(self):
    return self._vocab_size


def _build_embedding_layer(config: model_config.EmbeddingConfig,
                           max_seq_length: int):
  """Creates embedding layer of the specific `embedding_type`."""
  if config.embedding_type == model_config.GLOVE_EMBED:
    # If word_embedding_path is specified, use the embedding size of the
    # pre-trained embeddings.
    if config.word_embedding_path:
      with tf.io.gfile.GFile(config.word_embedding_path,
                             'rb') as embedding_file:
        word_embedding = np.load(embedding_file)
      vocab_size, embed_size = word_embedding.shape
      if config.vocab_size != vocab_size:
        raise ValueError(
            'Expected consistent vocab size between vocab.txt and the '
            'embedding, found {} and {}.'.format(vocab_size, config.vocab_size))
      config.embed_size = embed_size
      embeddings_initializer = (tf.keras.initializers.Constant(word_embedding))
    else:
      embeddings_initializer = None

    return _Embedding(
        INPUT_ID_NAME,
        config.vocab_size,
        config.embed_size,
        embeddings_initializer=embeddings_initializer,
        input_length=max_seq_length,
        trainable=config.trainable_embedding)
  elif config.embedding_type == model_config.BERT_EMBED:
    return _BERT(
        max_seq_length,
        bert_config=configs.BertConfig(**config.bert_config),
        trainable=config.trainable_embedding)
  raise ValueError('Invalid embedding type {}, expected {} or {}'.format(
      config.embedding_type, model_config.GLOVE_EMBED, model_config.BERT_EMBED))


class _DualRNN(tf.keras.Model):
  """Dual RNN base class.

  It receives two sentences (with masks) and the initial state of RNN, returns
    the outputs of two RNNs.

  To use the class, one needs to create an inheriting class implementing
  _run_dual_rnn().

  """

  def __init__(self,
               hidden_size: int,
               embedding_layer: Union[_BERT, _Embedding],
               num_layers: int = 1,
               dropout: float = 0.5,
               cell_type: Optional[str] = 'lstm',
               return_state: Optional[bool] = False,
               **kwargs: Dict[str, Any]):
    """Dual RNN base class constructor.

    Args:
      hidden_size: the hidden layer size of the RNN.
      embedding_layer: an embedding layer to be used.
      num_layers: number of layers of the RNN.
      dropout: dropout rate.
      cell_type: the RNN cell type.
      return_state: whether to include the final state in the outputs.
      **kwargs: optional arguments from childern class to be passed to
        _run_dual_rnn
    """
    super(_DualRNN, self).__init__()
    self._hidden_size = hidden_size
    self._num_layers = num_layers
    self._dropout = dropout
    self._cell_type = cell_type
    self._return_state = return_state

    self.embedding_layer = embedding_layer

  def build(self, input_shape):
    self.dropout = tf.keras.layers.Dropout(self._dropout)

  def call(self, input_1, input_2, initial_state, **kwargs):
    embed_1 = self.embedding_layer(input_1)
    embed_2 = self.embedding_layer(input_2)

    input_mask_1 = self._get_input_mask(input_1)
    input_mask_2 = self._get_input_mask(input_2)

    output_1, output_2, state_1, state_2 = self._run_dual_rnn(
        embed_1, embed_2, input_mask_1, input_mask_2, initial_state, **kwargs)
    if self._return_state:
      return output_1, output_2, state_1, state_2
    return output_1, output_2

  def _get_input_mask(self, inputs: Dict[str, tf.Tensor]) -> tf.Tensor:
    return inputs.get(INPUT_MASK_NAME,
                      tf.ones_like(inputs[INPUT_ID_NAME], dtype=tf.int32))

  def _run_dual_rnn(self, input_1, input_2, input_mask_1, input_mask_2,
                    initial_state, **kwargs):
    raise NotImplementedError('Must implement method _run_dual_rnn.')

  def _split_hidden_state_and_cell_state(self, state: Sequence[Any]):
    """Split the state into the hidden state (`h`) and optionally the cell state (`c`).

    When the cell state is a tuple (e.g., LSTM), `state` is of format
      [(h_1, c_1), (h_2, c_2), ..., (h_n, c_n)]
    where n is the num_layers.

    Otherwise `state` is of format
      [h_1, h_2, ..., h_n]

    Args:
      state: the cell state of each layer of the RNN.

    Returns:
      a tuple of the hidden state and (the cell state or None).

    """
    if utils.state_is_tuple(self._cell_type):
      return [s[0] for s in state], [s[1] for s in state]
    return state, None


class DualRNNEncoder(_DualRNN):
  """Dual RNN encoder."""

  def _create_rnn(self):
    cells = [
        utils.get_rnn_cell(self._cell_type)(units=self._hidden_size)
        for _ in range(self._num_layers)
    ]
    return tf.keras.layers.RNN(cells, return_state=True, return_sequences=True)

  def build(self, input_shape):
    super().build(input_shape)
    self.sent_rnn = self._create_rnn()

  def _run_dual_rnn(self, input_1, input_2, input_mask_1, input_mask_2,
                    initial_state, **unused_kwargs):
    del initial_state  # Not used
    output_1, state_1 = self._run_rnn(input_1, input_mask_1)
    output_2, state_2 = self._run_rnn(input_2, input_mask_2)

    hidden_state_1, _ = self._split_hidden_state_and_cell_state(state_1)
    hidden_state_2, _ = self._split_hidden_state_and_cell_state(state_2)

    return output_1, output_2, hidden_state_1, hidden_state_2

  def _run_rnn(self, inputs, input_mask):
    outputs = self.sent_rnn(inputs)
    output = outputs[0]
    state = outputs[1:]
    seqlen = tf.reduce_sum(input_mask, axis=1)
    final_step_output = utils.get_last_step(output, seqlen)
    final_step_output = self.dropout(final_step_output)
    return final_step_output, state


class DualRNNDecoder(_DualRNN):
  """Dual RNN decoder."""

  def _create_rnn(self, hidden_size):
    cells = [
        utils.get_rnn_cell(self._cell_type)(
            units=hidden_size, dropout=self._dropout)
        for _ in range(self._num_layers)
    ]
    return tf.keras.layers.RNN(cells, return_state=True, return_sequences=True)

  def build(self, input_shape):
    super().build(input_shape)
    self.dec_rnn_1 = self._create_rnn(self._hidden_size)
    self.project_1 = tf.keras.layers.Dense(self.embedding_layer.vocab_size)

    self.dec_rnn_2 = self._create_rnn(self._hidden_size * 2)
    self.project_2 = tf.keras.layers.Dense(self.embedding_layer.vocab_size)

  def _run_dual_rnn(self, input_1, input_2, input_mask_1, input_mask_2,
                    initial_state, **unused_kwargs):
    initial_state_1 = self._rnn1_initial_state(initial_state)
    output_1, state_1 = self._run_rnn(input_1, input_mask_1, initial_state_1,
                                      self.dec_rnn_1, self.dropout,
                                      self.project_1)

    initial_state_2 = self._rnn2_initial_state(initial_state, state_1)
    output_2, state_2 = self._run_rnn(input_2, input_mask_2, initial_state_2,
                                      self.dec_rnn_2, self.dropout,
                                      self.project_2)

    hidden_state_1, _ = self._split_hidden_state_and_cell_state(state_1)
    hidden_state_2, _ = self._split_hidden_state_and_cell_state(state_2)
    return output_1, output_2, hidden_state_1, hidden_state_2

  def _run_rnn(self, inputs, input_mask, initial_state, rnn, dropout,
               projection_layer):
    del input_mask  # Not used
    outputs = rnn(inputs, initial_state=initial_state)
    final_state = outputs[1:]
    outputs = dropout(outputs[0])
    outputs = projection_layer(outputs)
    return outputs, final_state

  def _concat_states(self, state_1: Sequence[tf.Tensor],
                     state_2: Sequence[tf.Tensor]) -> Sequence[tf.Tensor]:
    return [tf.concat([s1, s2], axis=1) for s1, s2 in zip(state_1, state_2)]

  def _rnn1_initial_state(self,
                          initial_state: Sequence[tf.Tensor]) -> Sequence[Any]:
    if utils.state_is_tuple(self._cell_type):
      return list(zip(initial_state, initial_state))
    return initial_state

  def _rnn2_initial_state(self, initial_state: Sequence[tf.Tensor],
                          rnn1_final_state: Sequence[Any]) -> Sequence[Any]:
    (rnn1_final_hidden_state, rnn1_final_cell_state
    ) = self._split_hidden_state_and_cell_state(rnn1_final_state)
    initial_hidden_state = self._concat_states(initial_state,
                                               rnn1_final_hidden_state)
    if utils.state_is_tuple(self._cell_type):
      initial_cell_state = self._concat_states(initial_state,
                                               rnn1_final_cell_state)
      return list(zip(initial_hidden_state, initial_cell_state))
    return initial_hidden_state


class _VAECell(tf.keras.layers.Layer):
  """VAE Cell base class.

  It receives two sentences (with masks) and the initial state as inputs and
  returns multiple
  outputs (see call() for details).

  It encodes->samples->decodes the inputs and updates the state in the
  meanwhile. However, the connection between any of two sequential components
  are component dependent. For example, the output of the sampler may not be
  fitted as the input of decoder. So, to use the class, one needs to create an
  inheriting class implementing
  the "glue" methods such as `_post_process_samples()`.

  """

  def __init__(self, encoder, sampler, decoder, state_updater):
    super(_VAECell, self).__init__()
    self.encoder = encoder
    self.sampler = sampler
    self.decoder = decoder
    self.state_updater = state_updater

  def _verify_and_prepare_inputs(self, inputs: Sequence[Any]):
    if len(inputs) not in (5, 7):
      raise ValueError(
          'Inputs should be a sequence of length 5 (encoder_input_1, '
          'encoderinput_2, decoder_input_1, decoder_input_2, state) or 7 '
          '(encoder_input_1, encoderinput_2, decoder_input_1, decoder_input_2, '
          'state, label, label_mask), found %s' % len(inputs))
    (encoder_input_1, encoder_input_2, decoder_input_1, decoder_input_2,
     state) = inputs[:5]
    if len(inputs) == 7:
      label, label_mask = inputs[5:]
    else:
      label = None
      label_mask = None
    return (encoder_input_1, encoder_input_2, decoder_input_1, decoder_input_2,
            state, label, label_mask)

  def _may_extract_from_tuple_state(self, state):
    if isinstance(state, (list, tuple)):
      return state[0]
    return state

  def _post_process_samples(self, samples: tf.Tensor) -> tf.Tensor:
    raise NotImplementedError('Must implement method to post-process samples.')

  def _project_encoder_outputs(self, inputs: Sequence[tf.Tensor]):
    raise NotImplementedError(
        'Must implement method to project encoder outputs.')

  def _prepare_encoder_initial_state(self,
                                     inputs: Sequence[tf.Tensor]) -> tf.Tensor:
    raise NotImplementedError(
        'Must implement method to prepare encoder initial state.')

  def _prepare_decoder_initial_state(
      self, inputs: Sequence[tf.Tensor]) -> Sequence[tf.Tensor]:
    raise NotImplementedError(
        'Must implement method to prepare decoder initial state.')

  def _prepare_decoder_inputs(
      self, inputs: Sequence[tf.Tensor]) -> Sequence[tf.Tensor]:
    return inputs

  def _prepare_state_updater_inputs(self, inputs: Sequence[Any]):
    raise NotImplementedError(
        'Must implement method to prepare state updater inputs.')

  def _post_process_decoder_state(self,
                                  state: Sequence[tf.Tensor]) -> tf.Tensor:
    return state[0]

  def _prepare_sample_logits(self, logits: tf.Tensor, label: Any,
                             label_mask: Any) -> tf.Tensor:
    del self, label, label_mask  # Unused.
    return logits

  def is_tuple_state(self):
    return self.state_updater.is_tuple_state()

  def call(self,
           inputs: Sequence[Any],
           return_states: Optional[bool] = False,
           return_samples: Optional[bool] = False):
    (encoder_input_1, encoder_input_2, decoder_input_1, decoder_input_2, state,
     label, label_mask) = self._verify_and_prepare_inputs(inputs)

    initial_state = self._may_extract_from_tuple_state(state)

    encoder_initial_state = self._prepare_encoder_initial_state([initial_state])
    encoder_outputs_1, encoder_outputs_2 = self.encoder(encoder_input_1,
                                                        encoder_input_2,
                                                        encoder_initial_state)

    latent_state, sampler_inputs = self._project_encoder_outputs(
        [encoder_outputs_1, encoder_outputs_2, initial_state])
    sample_logits = self._prepare_sample_logits(sampler_inputs, label,
                                                label_mask)
    samples = self.sampler(sample_logits)
    samples_processed = self._post_process_samples(samples)

    decoder_initial_state = self._prepare_decoder_initial_state(
        [initial_state, samples_processed])
    decoder_input_1, decoder_input_2 = self._prepare_decoder_inputs(
        [decoder_input_1, decoder_input_2])
    (decoder_outputs_1, decoder_outputs_2, decoder_state_1,
     decoder_state_2) = self.decoder(decoder_input_1, decoder_input_2,
                                     decoder_initial_state)
    decoder_state_1 = self._post_process_decoder_state(decoder_state_1)
    decoder_state_2 = self._post_process_decoder_state(decoder_state_2)
    decoder_initial_state = self._post_process_decoder_state(
        decoder_initial_state)

    state_updater_inputs = self._prepare_state_updater_inputs([
        samples_processed,
        encoder_outputs_1,
        encoder_outputs_2,
        state,
    ])
    next_state = self.state_updater(state_updater_inputs)

    outputs = [
        sample_logits, decoder_outputs_1, decoder_outputs_2, next_state,
        latent_state
    ]
    if return_states:
      outputs += [decoder_initial_state, decoder_state_1, decoder_state_2]
    if return_samples:
      outputs.append(samples)

    return outputs


class _VanillaStateUpdater(tf.keras.layers.Layer):
  """Vanilla hidden state updater."""

  def __init__(self, cell_type, units, dropout):
    del dropout
    super(_VanillaStateUpdater, self).__init__()

    self._cell_type = cell_type
    self._units = units
    self.cell = utils.get_rnn_cell(cell_type)(units)

  def call(self, inputs):
    if not isinstance(inputs, (list, tuple)) or len(inputs) != 2:
      raise ValueError('Expect inputs to be the sequence of length 2.')
    inputs, state = inputs
    _, state = self.cell(inputs, state)
    return state

  def is_tuple_state(self):
    return utils.state_is_tuple(self._cell_type)


class _VanillaEncoderOutputProjector(tf.keras.layers.Layer):
  """The layer to project vanilla vae cell's encoder outputs."""

  def __init__(self, hidden_sizes: Sequence[int], output_size: int,
               dropout: float):
    super(_VanillaEncoderOutputProjector, self).__init__()
    self.mlp = utils.MLP(hidden_sizes, dropout=dropout)
    self.project_layer = tf.keras.layers.Dense(output_size)

  def call(self, inputs: Sequence[Any]):
    if len(inputs) < 3:
      raise ValueError(
          'Expect inputs to be the sequence of length greater than 2, found {}.'
          .format(len(inputs)))
    encoder_input_1, encoder_input_2, initial_state = inputs[:3]
    inputs = tf.concat([initial_state, encoder_input_1, encoder_input_2],
                       axis=1)
    hidden = self.mlp(inputs)
    return hidden, self.project_layer(hidden)


class VanillaLinearVAECell(_VAECell):
  """Vanilla linear VAE Cell class."""

  def __init__(self, config: model_config.VanillaLinearVAECellConfig):
    model_config.verify_embedding_configs(config.encoder_embedding,
                                          config.decoder_embedding,
                                          config.shared_embedding)

    # Creates embedding layers for encoder and decoder.
    self.encoder_embedding_layer = _build_embedding_layer(
        config.encoder_embedding, config.max_seq_length)

    if config.shared_embedding:
      self.decoder_embedding_layer = self.encoder_embedding_layer
      self.shared_embedding_layer = self.encoder_embedding_layer
    else:
      self.decoder_embedding_layer = _build_embedding_layer(
          config.decoder_embedding, config.max_seq_length)
      self.shared_embedding_layer = None

    encoder = DualRNNEncoder(
        hidden_size=config.encoder_hidden_size,
        embedding_layer=self.encoder_embedding_layer,
        num_layers=config.num_ecnoder_rnn_layers,
        dropout=config.dropout,
        cell_type=config.encoder_cell_type)
    sampler = utils.GumbelSoftmaxSampler(config.temperature, hard=False)

    decoder = DualRNNDecoder(
        hidden_size=config.decoder_hidden_size,
        embedding_layer=self.decoder_embedding_layer,
        # Hardcoded to be 1 layer to align with pytorch version. Otherwise, we
        # need to define the initial state for each layer in
        # _prepare_decoder_initial_state and change _post_process_decoder_state
        num_layers=1,
        dropout=config.dropout,
        cell_type=config.decoder_cell_type,
        return_state=True)
    state_updater = _VanillaStateUpdater(config.state_updater_cell_type,
                                         config.num_states, config.dropout)

    self._gumbel_softmax_label_adjustment_multiplier = (
        config.gumbel_softmax_label_adjustment_multiplier)
    self.encoder_output_projector = _VanillaEncoderOutputProjector(
        hidden_sizes=list(config.encoder_projection_sizes),
        output_size=config.num_states,
        dropout=config.dropout)
    self.sample_post_processor = utils.MLP(
        config.sampler_post_processor_output_sizes, dropout=config.dropout)

    super(VanillaLinearVAECell, self).__init__(
        encoder=encoder,
        sampler=sampler,
        decoder=decoder,
        state_updater=state_updater)

  def init_bert_embedding_layers(
      self, config: model_config.VanillaLinearVAECellConfig):
    if config.encoder_embedding.embedding_type == model_config.BERT_EMBED:
      (self.encoder_embedding_layer, _,
       _) = bert_utils.load_bert_weight_from_ckpt(
           bert_model=self.encoder_embedding_layer,
           bert_ckpt_dir=config.encoder_embedding.bert_ckpt_dir)
    if config.decoder_embedding.embedding_type == model_config.BERT_EMBED:
      (self.decoder_embedding_layer, _,
       _) = bert_utils.load_bert_weight_from_ckpt(
           bert_model=self.decoder_embedding_layer,
           bert_ckpt_dir=config.encoder_embedding.bert_ckpt_dir)

  def _post_process_samples(self, samples: tf.Tensor) -> tf.Tensor:
    return self.sample_post_processor(samples)

  def _project_encoder_outputs(self, inputs: Sequence[tf.Tensor]):
    return self.encoder_output_projector(inputs)

  def _prepare_encoder_initial_state(self, inputs: Sequence[tf.Tensor]):
    # Encoder don't use external initial state.
    return None

  def _prepare_decoder_initial_state(
      self, inputs: Sequence[tf.Tensor]) -> Sequence[tf.Tensor]:
    if len(inputs) < 2:
      raise ValueError(
          'Expect inputs to be the sequence of length greater than 1, found {}.'
          .format(len(inputs)))
    initial_state, samples_processed = inputs[0], inputs[1]
    return [tf.concat([initial_state, samples_processed], axis=1)]

  def _prepare_decoder_inputs(  # pytype: disable=signature-mismatch  # overriding-parameter-type-checks
      self, inputs: _TensorMapList) -> _TensorMapList:
    last_step_removed = [
        {key: value[:, :-1] for key, value in input.items()} for input in inputs
    ]
    return last_step_removed

  def _prepare_state_updater_inputs(self, inputs: Sequence[Any]):
    if len(inputs) < 4:
      raise ValueError(
          'Expect inputs to be the sequence of length greater than 3, found {}.'
          .format(len(inputs)))
    samples_processed, encoder_inputs_1, encoder_inputs_2, state = inputs[:4]
    inputs = tf.concat([samples_processed, encoder_inputs_1, encoder_inputs_2],
                       axis=1)
    return [inputs, state]

  def _prepare_sample_logits(self, logits: tf.Tensor,
                             label: Optional[tf.Tensor],
                             label_mask: Any) -> tf.Tensor:
    if label is None and label_mask is None:
      return super()._prepare_sample_logits(logits, label, label_mask)
    if label is None or label_mask is None:
      raise ValueError(
          'label and label_mask must be both specified, found one is None')

    # Add weighted one-hot label to the sample logits.
    # See https://aclanthology.org/2021.naacl-main.374.pdf for details.

    # Expand the dimension for broadcast multiply.
    label_mask = tf.expand_dims(label_mask, axis=-1)
    logits_label_adjument = tf.norm(
        logits, axis=-1, keepdims=True) * tf.cast(
            label, logits.dtype) * tf.cast(label_mask, logits.dtype)

    return logits + self._gumbel_softmax_label_adjustment_multiplier * logits_label_adjument

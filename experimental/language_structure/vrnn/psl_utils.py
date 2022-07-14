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

"""PSL utils to load PSL constraint model."""

from typing import Any, Dict, List, Sequence

import tensorflow as tf
import data  # local file import from experimental.language_structure.psl as data_utils
import psl_model  # local file import from experimental.language_structure.psl
import psl_model_dstc_synthetic  # local file import from experimental.language_structure.psl
import psl_model_multiwoz  # local file import from experimental.language_structure.psl
import data_preprocessor  # local file import from experimental.language_structure.vrnn as preprocessor

_INPUT_ID_NAME = preprocessor.INPUT_ID_NAME

_MULTIWOZ_SYNTH = 'multiwoz_synth'
_SGD_SYNTH = 'sgd_synth'
_SGD = 'sgd'
_SGD_DOMAIN_ADAPATION = 'sgd_domain_adapation'


def get_psl_model(dataset: str, rule_names: List[str],
                  rule_weights: List[float], **kwargs) -> psl_model.PSLModel:
  """Constraints PSL constraint model."""
  psl_model_cls_map = {
      _MULTIWOZ_SYNTH: psl_model_multiwoz.PSLModelMultiWoZ,
      _SGD_SYNTH: psl_model_dstc_synthetic.PSLModelDSTCSynthetic,
      _SGD: psl_model_dstc_synthetic.PSLModelDSTCSynthetic,
      _SGD_DOMAIN_ADAPATION: psl_model_dstc_synthetic.PSLModelDSTCSynthetic,
  }

  if dataset in psl_model_cls_map:
    psl_model_cls = psl_model_cls_map[dataset]
    return psl_model_cls(rule_weights, rule_names, **kwargs)

  raise ValueError('Supported PSL constraint for dataset {}, found {}.'.format(
      ', '.join(psl_model_cls_map.keys()), dataset))


def _get_keyword_ids_per_class(dataset: str, config: Dict[str, Any],
                               vocab: Sequence[str]) -> Sequence[Sequence[int]]:
  """Gets keyword ids for each class in the PSL constraint model."""
  vocab_mapping = {word: word_id for word_id, word in enumerate(vocab)}
  keyword_ids_per_class = []

  if dataset == _MULTIWOZ_SYNTH:
    keys = [
        'accept_words', 'cancel_words', 'end_words', 'greet_words',
        'info_question_words', 'insist_words', 'slot_question_words'
    ]
    for key in keys:
      keyword_ids = [
          vocab_mapping[word] for word in config[key] if word in vocab_mapping
      ]
      keyword_ids_per_class.append(keyword_ids)
  return keyword_ids_per_class


def _create_psl_features(
    user_utterance_ids: tf.Tensor, system_utterance_ids: tf.Tensor,
    config: Dict[str, Any], dataset: str,
    keyword_ids_per_class: Sequence[Sequence[int]]) -> tf.Tensor:
  """Creates features for PSL constraint model."""
  if dataset not in (_MULTIWOZ_SYNTH):
    return tf.concat([user_utterance_ids, system_utterance_ids], axis=-1)
  features = data_utils.create_features(
      user_utterance_ids,
      system_utterance_ids,
      keyword_ids_per_class,
      check_keyword_by_utterance=dataset == 'sgd_synth',
      include_keyword_value=config['includes_word'],
      exclude_keyword_value=config['excludes_word'],
      pad_utterance_mask_value=config['pad_utterance_mask'],
      utterance_mask_value=config['utterance_mask'],
      last_utterance_mask_value=config['last_utterance_mask'])
  return features


def psl_feature_mixin(fn: Any, dataset: str, psl_config: Dict[str, Any],
                      vocab: Sequence[str]):
  """Creates PSL feature generation mixin.

  Args:
    fn: dataset processing function converting the dataset into VRNN features.
    dataset: dataset name.
    psl_config: PSL config to create features.
    vocab: vocabulary list.

  Returns:
    decorated `fn` to include PSL input features generation.
  """
  keyword_ids_per_class = _get_keyword_ids_per_class(dataset, psl_config, vocab)

  def _run(inputs: Sequence[tf.Tensor]):
    encoder_input_1, encoder_input_2 = inputs[:2]

    psl_inputs = _create_psl_features(encoder_input_1[_INPUT_ID_NAME],
                                      encoder_input_2[_INPUT_ID_NAME],
                                      psl_config, dataset,
                                      keyword_ids_per_class)
    return (*inputs, psl_inputs)

  return lambda inputs: _run(fn(inputs))


def _copy_model_weights(weights: List[tf.Tensor]) -> List[tf.Tensor]:
  """Copies a list of model weights."""
  weights_copy = []
  for layer in weights:
    weights_copy.append(tf.identity(layer))

  return weights_copy


def update_logits(model: tf.keras.Model,
                  optimizer: tf.keras.optimizers.Optimizer, model_inputs: Any,
                  get_logits_fn: Any, psl_constraint: psl_model.PSLModel,
                  psl_inputs: tf.Tensor, grad_steps: int,
                  alpha: float) -> tf.Tensor:
  """Test step for gradient based weight updates.

  Args:
    model: keras model generating the logits
    optimizer: keras optimizer
    model_inputs: model input features
    get_logits_fn: the function deriving the logits from the model outputs.
    psl_constraint: differentable psl constraints
    psl_inputs: psl input features
    grad_steps: number of gradient steps taken to try and satisfy the
      constraints
    alpha: parameter to determine how important it is to keep the constrained
      weights close to the trained unconstrained weights

  Returns:
    Logits after satisfiying constraints.
  """

  @tf.function
  def test_step(model_inputs: Any, psl_inputs: tf.Tensor,
                weights: Sequence[tf.Tensor]):
    """Update weights by satisfing test constraints."""
    with tf.GradientTape() as tape:
      model_outputs = model(model_inputs, training=False)
      logits = get_logits_fn(model_outputs)
      constraint_loss = psl_constraint.compute_loss(psl_inputs, logits)
      weight_loss = tf.reduce_sum([
          tf.reduce_mean(tf.math.squared_difference(w, w_h))
          for w, w_h in zip(weights, model.trainable_weights)
      ])
      loss = constraint_loss + alpha * weight_loss

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

  weights_copy = _copy_model_weights(model.trainable_weights)
  for _ in tf.range(tf.cast(grad_steps, dtype=tf.int32)):
    test_step(model_inputs, psl_inputs, weights=weights_copy)

  model_outputs = model(model_inputs)
  logits = get_logits_fn(model_outputs)
  for var, weight in zip(model.trainable_variables, weights_copy):
    var.assign(weight)

  return logits

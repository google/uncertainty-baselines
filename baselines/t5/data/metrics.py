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

"""Metrics for Seq2Seq parsing."""
import collections
from typing import Any, Dict, List, Optional, Sequence, Text, Tuple, Union

from absl import logging

import numpy as np

import robustness_metrics.metrics as rm_metrics
import scipy.special as sp_special
import seqio
import sklearn.metrics as sk_metrics
import t5.data
import t5x.decoding as t5x_decoding
import tensorflow as tf

from data import metrics_utils  # local file import from baselines.t5
from data.deepbank import graph_utils  # local file import from baselines.t5
from data.deepbank import penman_utils  # local file import from baselines.t5


NEG_INF = t5x_decoding.NEG_INF
tf_metrics = tf.keras.metrics

DEFAULT_VOCAB = t5.data.get_default_vocabulary()

GreedyScore = float
TopkPrediction = Sequence[int]
TopkScore = Union[Sequence[float], Sequence[Sequence[float]]]

BeamScore = Tuple[GreedyScore, TopkPrediction, TopkScore]

_SEQUENCE_BEAM_TYPES = ('all_beam', 'non_top1_beam', 'top1_beam')
_SEQUENCE_UNCERTAINTY_TYPES = ('probability', 'margin', 'entropy')


def _safe_divide(x, y):
  return x / y if y != 0 else 0.0


def _list_to_count_dict(node_list: List[Text]) -> Dict[Text, int]:
  count_dict = collections.defaultdict(int)
  for node in node_list:
    count_dict[node] += 1
  return count_dict


def _overlap_node_num(targets: List[Text], predictions: List[Text]) -> int:
  target_count_dict = _list_to_count_dict(targets)
  predicted_count_dict = _list_to_count_dict(predictions)
  overlap_num = 0
  for node, count in target_count_dict.items():
    overlap_num += min(count, predicted_count_dict[node])
  return overlap_num


def seq2seq_metrics(targets: List[Text],
                    predictions: List[Text]) -> Dict[Text, float]:
  """Returns eval metrics for seq2seq without assuming any structure."""
  num_correct = 0
  num_total = 0

  for target, predicted in zip(targets, predictions):
    if target == predicted:
      num_correct += 1
    num_total += 1

  return dict(sequence_accuracy=_safe_divide(num_correct, num_total))


def _seq2seq_uncertainty_metrics(targets: List[Text],
                                 predictions: List[Text],
                                 log_probs: np.ndarray,
                                 num_ece_bins: int = 15,
                                 metric_prefix: str = '') -> Dict[Text, float]:
  """Returns uncertainty metrics for seq2seq tasks."""

  # Convert to numpy.
  targets = np.array(targets)
  predictions = np.array(predictions)

  model_pred_confs = np.exp(log_probs)
  model_pred_confs_ece = np.stack([1. - model_pred_confs, model_pred_confs],
                                  axis=-1)

  # Compute sequence-level accuracy (i.e., full match).
  correct_predictions = np.array(predictions == targets)
  correct_predictions_fl = np.array(correct_predictions, dtype=np.float32)

  # Define sequence-level uncertainty metrics (ECE and calibration AUC).
  ece = rm_metrics.ExpectedCalibrationError(num_bins=num_ece_bins)
  calib_auroc = rm_metrics.CalibrationAUC(
      curve='ROC', correct_pred_as_pos_label=False)
  calib_auprc = rm_metrics.CalibrationAUC(
      curve='PR', correct_pred_as_pos_label=False)

  ece.add_batch(model_pred_confs_ece, label=correct_predictions)
  calib_auroc.add_batch(
      np.ones_like(correct_predictions_fl),
      confidence=model_pred_confs,
      label=correct_predictions_fl)
  calib_auprc.add_batch(
      np.ones_like(correct_predictions_fl),
      confidence=model_pred_confs,
      label=correct_predictions_fl)

  return {
      f'{metric_prefix}ece': ece.result()['ece'],
      f'{metric_prefix}calib_auroc': calib_auroc.result()['calibration_auc'],
      f'{metric_prefix}calib_auprc': calib_auprc.result()['calibration_auc']
  }


def seq2seq_uncertainty_metrics(
    targets: List[Text],
    predictions: List[Text],
    aux_values: Dict[Text, Any],
    vocab: seqio.SentencePieceVocabulary = DEFAULT_VOCAB,
    num_ece_bins: int = 15) -> Dict[Text, float]:
  """Returns uncertainty metrics for seq2seq without assuming any structure.

  This function takes `predictions` (a list of strings) and `aux_values`
  (a dictionary of token score values) from `EncoderDecoderBeamScoreModel`'s
  `predict_batch_with_aux` function, and use them to compute model's calibration
  performance. The token level implementation follows [1], formula (5).

  Note that to ensure seqio to correctly feed the output of
  `predict_batch_with_aux` to this metric. The first three positional metrics
  must be ('targets', 'predictions', 'aux_values').

  ## Reference

  [1]: Aviral Kumar, Sunita Sarawagi. Calibration of Encoder Decoder Models for
       Neural Machine Translation. https://arxiv.org/abs/1903.00802

  Args:
    targets: target strings for model prediction, shape (batch_size,).
    predictions: predicted strings for model prediction, shape (batch_size,).
    aux_values: A dictionary of auxillary information provided by the
      `predict_batch_with_aux` function. It must have a field `scores` which
      contains sequence-level or token-level log probabilities with shape
      (batch_size,) or (batch_size, max_seq_length) respectively.
    vocab: The SentencePieceVocabulary used for model training.
    num_ece_bins: The number of bins to use for expected calibration error (ECE)
      metric.

  Returns:
    A dictionary of metric values.
  """
  log_probs = aux_values.get('scores', None)

  if log_probs is None:
    raise ValueError('Cannot find the field `scores` from `aux_values`.')

  # Convert to numpy.
  log_probs = np.array(log_probs, dtype=np.float32)

  # Compute sequence-level probability.
  # If log probability is on token level with shape (batch_size, seq_len),
  # Reduce it back to sequence-level score (batch_size, ).
  metrics = {}
  if log_probs.ndim == 2:
    token_targets = []
    token_predictions = []
    token_log_probs = []
    seq_log_probs = []
    for (target, pred, log_prob) in zip(targets, predictions, log_probs):
      # Note that the token at position = `seq_length` is an EOS symbol, whose
      # probability should count towards the sequence probability.
      pred_encoded = list(vocab.encode(pred)) + [vocab.eos_id]
      target_encoded = list(vocab.encode(target)) + [vocab.eos_id]
      pred_length = len(pred_encoded)
      target_length = len(target_encoded)
      if target_length > pred_length:
        target_encoded = target_encoded[:pred_length]
      elif target_length < pred_length:
        target_encoded = target_encoded + [vocab.pad_id] * (
            pred_length - target_length)

      # The sum of `log_prob` is only up to the length of sequence.
      log_prob = log_prob[:pred_length]

      seq_log_probs.append(np.sum(log_prob))
      # Length of log_prob might be less than length of the encoded sequence.
      # This rarely happens and we will skip those sequences when computing
      # token metrics.
      if len(log_prob) == pred_length:
        token_targets.extend(target_encoded)
        token_predictions.extend(pred_encoded)
        token_log_probs.extend(log_prob.tolist())

    token_log_probs = np.array(token_log_probs, dtype=np.float32)
    log_probs = np.array(seq_log_probs, dtype=np.float32)
    # TODO(phandu): Add weighted token ECE metric following
    # Section 2.2.2 of [1].
    metrics = _seq2seq_uncertainty_metrics(
        token_targets,
        token_predictions,
        token_log_probs,
        num_ece_bins=num_ece_bins,
        metric_prefix='token_')

  sequence_metrics = _seq2seq_uncertainty_metrics(
      targets,
      predictions,
      log_probs,
      num_ece_bins=num_ece_bins,
      metric_prefix='sequence_')
  metrics.update(sequence_metrics)
  return metrics


def deepbank_metrics(targets: List[Text],
                     predictions: List[Text],
                     data_version: str = 'v0') -> Dict[Text, float]:
  """Returns eval metrics for seq2seq semantic parsing."""
  num_correct = 0
  num_correct_root = 0
  num_total = 0

  precision_by_node = collections.defaultdict(int)
  recall_by_node = collections.defaultdict(int)

  for t, p in zip(targets, predictions):
    target = penman_utils.reverse_tokened_graph_str(
        t, data_version=data_version)
    predicted = penman_utils.reverse_tokened_graph_str(
        p, data_version=data_version)
    if target == predicted:
      num_correct += 1
    if metrics_utils.find_root(target) == metrics_utils.find_root(predicted):
      num_correct_root += 1
    num_total += 1

    target_nodeseq_dict = metrics_utils.graph_to_nodeseq(
        target, data_version=data_version)
    predicted_nodeseq_dict = metrics_utils.graph_to_nodeseq(
        predicted, data_version=data_version)

    for node_type, target_nodeseq in target_nodeseq_dict.items():
      predicted_nodeseq = predicted_nodeseq_dict[node_type]
      precision_by_node[node_type] += _safe_divide(
          _overlap_node_num(target_nodeseq, predicted_nodeseq),
          len(predicted_nodeseq))
      recall_by_node[node_type] += _safe_divide(
          _overlap_node_num(target_nodeseq, predicted_nodeseq),
          len(target_nodeseq))

  analysis = dict(
      sequence_accuracy=_safe_divide(num_correct, num_total),
      root_accuracy=_safe_divide(num_correct_root, num_total))

  for node_type, _ in precision_by_node.items():
    analysis['Precision/' + node_type] = _safe_divide(
        precision_by_node[node_type], num_total)
    analysis['Recall/' + node_type] = _safe_divide(recall_by_node[node_type],
                                                   num_total)

  return analysis


def deepbank_metrics_v2(targets: List[Text],
                        predictions: List[Text],
                        data_version: str = 'v0') -> Dict[Text, float]:
  """Returns eval metrics for seq2seq semantic parsing with SMATCH metrics."""
  total_match_node_num, total_test_node_num, total_gold_node_num = 0, 0, 0
  total_match_edge_num, total_test_edge_num, total_gold_edge_num = 0, 0, 0
  total_match_attr_num, total_test_attr_num, total_gold_attr_num = 0, 0, 0

  for t, p in zip(targets, predictions):
    # Transfers the original input/output to PENMAN object.
    target_penman = penman_utils.PENMANStr(
        t, variable_free=True, retokenized=True,
        data_version=data_version).penman
    predicted_penman = penman_utils.PENMANStr(
        p, variable_free=True, retokenized=True,
        data_version=data_version).penman
    # Computes info for Smatch scores.
    match_node_num, test_node_num, gold_node_num, _ = graph_utils.get_dag_match(
        predicted_penman, target_penman,
        just_match_instance=True)
    match_edge_num, test_edge_num, gold_edge_num, _ = graph_utils.get_dag_match(
        predicted_penman, target_penman,
        just_match_relation=True)
    match_attr_num, test_attr_num, gold_attr_num, _ = graph_utils.get_dag_match(
        predicted_penman, target_penman,
        just_match_attribute=True)

    total_match_node_num += match_node_num
    total_test_node_num += test_node_num
    total_gold_node_num += gold_node_num

    total_match_edge_num += match_edge_num
    total_test_edge_num += test_edge_num
    total_gold_edge_num += gold_edge_num

    total_match_attr_num += match_attr_num
    total_test_attr_num += test_attr_num
    total_gold_attr_num += gold_attr_num

  node_p = _safe_divide(total_match_node_num, total_test_node_num)
  node_r = _safe_divide(total_match_node_num, total_gold_node_num)

  edge_p = _safe_divide(total_match_edge_num, total_test_edge_num)
  edge_r = _safe_divide(total_match_edge_num, total_gold_edge_num)

  attr_p = _safe_divide(total_match_attr_num, total_test_attr_num)
  attr_r = _safe_divide(total_match_attr_num, total_gold_attr_num)

  total_match_num = total_match_node_num + total_match_edge_num + total_match_attr_num
  total_test_num = total_test_node_num + total_test_edge_num + total_test_attr_num
  total_gold_num = total_gold_node_num + total_gold_edge_num + total_gold_attr_num
  total_p = _safe_divide(total_match_num, total_test_num)
  total_r = _safe_divide(total_match_num, total_gold_num)

  analysis = dict(
      node_precision=node_p,
      node_recall=node_r,
      node_smatch=_safe_divide(2 * node_p * node_r, node_p + node_r),
      edge_precision=edge_p,
      edge_recall=edge_r,
      edge_smatch=_safe_divide(2 * edge_p * edge_r, edge_p + edge_r),
      attribute_precision=attr_p,
      attribute_recall=attr_r,
      attribute_smatch=_safe_divide(2 * attr_p * attr_r, attr_p + attr_r),
      total_precision=total_p,
      total_recall=total_r,
      total_smatch=_safe_divide(2 * total_p * total_r, total_p + total_r))

  return analysis


def dataflow_metrics(targets: List[Text],
                     predictions: List[Text],
                     dataset_name: str = 'snips') -> Dict[Text, float]:
  """Returns eval metrics for seq2seq semantic parsing with SMATCH metrics."""
  total_match_node_num, total_test_node_num, total_gold_node_num = 0, 0, 0
  total_match_edge_num, total_test_edge_num, total_gold_edge_num = 0, 0, 0
  total_match_attr_num, total_test_attr_num, total_gold_attr_num = 0, 0, 0

  for t, p in zip(targets, predictions):
    # Transfers the original input/output to PENMAN object.
    target_penman = penman_utils.PENMANStr(
        t, variable_free=True, retokenized=True,
        data_version=dataset_name).penman
    predicted_penman = penman_utils.PENMANStr(
        p, variable_free=True, retokenized=True,
        data_version=dataset_name).penman
    # Computes info for Smatch scores.
    match_node_num, test_node_num, gold_node_num, _ = graph_utils.get_dag_match(
        predicted_penman, target_penman,
        just_match_instance=True)
    match_edge_num, test_edge_num, gold_edge_num, _ = graph_utils.get_dag_match(
        predicted_penman, target_penman,
        just_match_relation=True)
    match_attr_num, test_attr_num, gold_attr_num, _ = graph_utils.get_dag_match(
        predicted_penman, target_penman,
        just_match_attribute=True)

    total_match_node_num += match_node_num
    total_test_node_num += test_node_num
    total_gold_node_num += gold_node_num

    total_match_edge_num += match_edge_num
    total_test_edge_num += test_edge_num
    total_gold_edge_num += gold_edge_num

    total_match_attr_num += match_attr_num
    total_test_attr_num += test_attr_num
    total_gold_attr_num += gold_attr_num

  node_p = _safe_divide(total_match_node_num, total_test_node_num)
  node_r = _safe_divide(total_match_node_num, total_gold_node_num)

  edge_p = _safe_divide(total_match_edge_num, total_test_edge_num)
  edge_r = _safe_divide(total_match_edge_num, total_gold_edge_num)

  attr_p = _safe_divide(total_match_attr_num, total_test_attr_num)
  attr_r = _safe_divide(total_match_attr_num, total_gold_attr_num)

  total_match_num = total_match_node_num + total_match_edge_num + total_match_attr_num
  total_test_num = total_test_node_num + total_test_edge_num + total_test_attr_num
  total_gold_num = total_gold_node_num + total_gold_edge_num + total_gold_attr_num
  total_p = _safe_divide(total_match_num, total_test_num)
  total_r = _safe_divide(total_match_num, total_gold_num)

  analysis = dict(
      node_precision=node_p,
      node_recall=node_r,
      node_smatch=_safe_divide(2 * node_p * node_r, node_p + node_r),
      edge_precision=edge_p,
      edge_recall=edge_r,
      edge_smatch=_safe_divide(2 * edge_p * edge_r, edge_p + edge_r),
      attribute_precision=attr_p,
      attribute_recall=attr_r,
      attribute_smatch=_safe_divide(2 * attr_p * attr_r, attr_p + attr_r),
      total_precision=total_p,
      total_recall=total_r,
      total_smatch=_safe_divide(2 * total_p * total_r, total_p + total_r))

  return analysis


def deepbank_uncertainty_metrics(
    targets: List[Text],
    predictions: List[Text],
    aux_values: Dict[Text, Any],
    vocab: seqio.SentencePieceVocabulary = DEFAULT_VOCAB,
    data_version: str = 'v0',
    num_ece_bins: int = 15,
    smatch_threshold: float = 0.85) -> Dict[Text, float]:
  """Returns compositional uncertainty metrics for DeepBank graphs.

  This function takes `predictions` (a list of strings) and `aux_values`
  (a dictionary of token score values) from `EncoderDecoderBeamScoreModel`'s
  `predict_batch_with_aux` function, and use them to compute model's calibration
  performance.

  Note that to ensure seqio to correctly feed the output of
  `predict_batch_with_aux` to this metric. The first three positional metrics
  must be ('targets', 'predictions', 'aux_values').

  Args:
    targets: target strings for model prediction, shape (batch_size,).
    predictions: predicted strings for model prediction, shape (batch_size,).
    aux_values: A dictionary of auxillary information provided by the
      `predict_batch_with_aux` function. It must have a field `scores` which
      contains token-level log probabilities with shape (batch_size,
      max_seq_length).
    vocab: The SentencePieceVocabulary used for model training.
    data_version: DeepBank version.
    num_ece_bins: The number of bins to use for expected calibration error (ECE)
      metric.
    smatch_threshold: Threshold for defining a good smatch score.

  Returns:
    A dictionary of metric values.
  """
  log_probs = aux_values.get('scores', None)
  log_probs = np.array(log_probs, dtype=np.float32)

  if log_probs is None:
    raise ValueError('Cannot find the field `scores` from `aux_values`.')
  # The `log_probs` should be token-level probabilties,
  # with shape (batch_size, max_seq_length)
  if log_probs.ndim != 2:
    raise ValueError('Incorrect dimension for `log_probs`, which should be 2.'
                     f'Got {log_probs.ndim}.')

  all_node_prob_match_list = []
  all_attr_prob_match_list = []
  all_edge_prob_match_list = []
  all_smatch_score_list = []
  seq_log_probs = []
  for pred, target, log_prob in zip(predictions, targets, log_probs):
    token_ids = vocab.encode(pred)
    tokens = [
        metrics_utils.single_token_transfer(vocab.decode([id]), data_version)
        for id in token_ids
    ]
    seq_log_probs.append(np.sum(log_prob[:len(tokens)+1]))
    log_prob = log_prob[:len(tokens)]

    pred_penman_with_prob = penman_utils.assign_prob_to_penman(
        tokens, log_prob, data_version)
    try:
      _ = graph_utils.parse_string_to_dag(pred_penman_with_prob)
    except LookupError:
      # The predicted graph here is an ill-formed graph. Skip.
      logging.warning('Fail to parse DAG: %s', pred_penman_with_prob)
      continue

    pred_penman = penman_utils.PENMANStr(
        pred, variable_free=True, retokenized=True,
        data_version=data_version).penman
    gold_penman = penman_utils.PENMANStr(
        target, variable_free=True, retokenized=True,
        data_version=data_version).penman
    _, _, smatch_score = graph_utils.get_smatch(pred_penman, gold_penman)
    all_smatch_score_list.append(smatch_score)

    # Generates a list of predictive probability and predictive accuracy for
    # node/attribute/edge predictions.
    (node_prob_match_list, attr_prob_match_list,
     edge_prob_match_list) = graph_utils.get_dag_match_for_calibration(
         pred_penman_with_prob, gold_penman)
    all_node_prob_match_list += node_prob_match_list
    all_attr_prob_match_list += attr_prob_match_list
    all_edge_prob_match_list += edge_prob_match_list

  def compositional_eval(prob_match_list):
    model_pred_confs = np.array([p for (p, _) in prob_match_list],
                                dtype=np.float32)
    model_pred_confs_ece = np.stack([1. - model_pred_confs, model_pred_confs],
                                    axis=-1)
    correct_predictions = np.array([c for (_, c) in prob_match_list])
    correct_predictions_fl = np.array(correct_predictions, dtype=np.float32)
    ece = rm_metrics.ExpectedCalibrationError(num_bins=num_ece_bins)
    ece.add_batch(model_pred_confs_ece, label=correct_predictions)
    calib_auroc = rm_metrics.CalibrationAUC(
        curve='ROC', correct_pred_as_pos_label=False)
    calib_auprc = rm_metrics.CalibrationAUC(
        curve='PR', correct_pred_as_pos_label=False)
    calib_auroc.add_batch(
        np.ones_like(correct_predictions_fl),
        confidence=model_pred_confs,
        label=correct_predictions_fl)
    calib_auprc.add_batch(
        np.ones_like(correct_predictions_fl),
        confidence=model_pred_confs,
        label=correct_predictions_fl)
    return ece, calib_auroc, calib_auprc

  # Node evaluation
  node_ece, node_calib_auroc, node_calib_auprc = compositional_eval(
      all_node_prob_match_list)
  # Node evaluation
  attr_ece, attr_calib_auroc, attr_calib_auprc = compositional_eval(
      all_attr_prob_match_list)
  # Edge evaluation
  edge_ece, edge_calib_auroc, edge_calib_auprc = compositional_eval(
      all_edge_prob_match_list)
  # Uncertainty metrics based on smtach
  model_pred_confs = np.exp(np.array(seq_log_probs, dtype=np.float32))
  model_pred_confs_ece = np.stack([1. - model_pred_confs, model_pred_confs],
                                  axis=-1)
  correct_predictions = np.array(
      [s >= smatch_threshold for s in all_smatch_score_list])
  correct_predictions_fl = np.array(correct_predictions, dtype=np.float32)

  smatch_ece = rm_metrics.ExpectedCalibrationError(num_bins=num_ece_bins)
  smatch_calib_auroc = rm_metrics.CalibrationAUC(
      curve='ROC', correct_pred_as_pos_label=False)
  smatch_calib_auprc = rm_metrics.CalibrationAUC(
      curve='PR', correct_pred_as_pos_label=False)

  smatch_ece.add_batch(model_pred_confs_ece, label=correct_predictions)
  smatch_calib_auroc.add_batch(
      np.ones_like(correct_predictions_fl),
      confidence=model_pred_confs,
      label=correct_predictions_fl)
  smatch_calib_auprc.add_batch(
      np.ones_like(correct_predictions_fl),
      confidence=model_pred_confs,
      label=correct_predictions_fl)

  analysis = dict(
      node_ece=node_ece.result()['ece'],
      node_calib_auroc=node_calib_auroc.result()['calibration_auc'],
      node_calib_auprc=node_calib_auprc.result()['calibration_auc'],
      attr_ece=attr_ece.result()['ece'],
      attr_calib_auroc=attr_calib_auroc.result()['calibration_auc'],
      attr_calib_auprc=attr_calib_auprc.result()['calibration_auc'],
      edge_ece=edge_ece.result()['ece'],
      edge_calib_auroc=edge_calib_auroc.result()['calibration_auc'],
      edge_calib_auprc=edge_calib_auprc.result()['calibration_auc'],
      smatch_ece=smatch_ece.result()['ece'],
      smatch_calib_auroc=smatch_calib_auroc.result()['calibration_auc'],
      smatch_calib_auprc=smatch_calib_auprc.result()['calibration_auc'],
  )
  return analysis


def dataflow_uncertainty_metrics(
    targets: List[Text],
    predictions: List[Text],
    aux_values: Dict[Text, Any],
    vocab: seqio.SentencePieceVocabulary = DEFAULT_VOCAB,
    dataset_name: str = 'snips',
    num_ece_bins: int = 15,
    smatch_threshold: float = 0.85) -> Dict[Text, float]:
  """Returns compositional uncertainty metrics for DataFlow graphs.

  This function takes `predictions` (a list of strings) and `aux_values`
  (a dictionary of token score values) from `EncoderDecoderBeamScoreModel`'s
  `predict_batch_with_aux` function, and use them to compute model's calibration
  performance.

  Note that to ensure seqio to correctly feed the output of
  `predict_batch_with_aux` to this metric. The first three positional metrics
  must be ('targets', 'predictions', 'aux_values').

  Args:
    targets: target strings for model prediction, shape (batch_size,).
    predictions: predicted strings for model prediction, shape (batch_size,).
    aux_values: A dictionary of auxillary information provided by the
      `predict_batch_with_aux` function. It must have a field `scores` which
      contains token-level log probabilities with shape (batch_size,
      max_seq_length).
    vocab: The SentencePieceVocabulary used for model training.
    dataset_name: name of the dataflow dataset.
    num_ece_bins: The number of bins to use for expected calibration error (ECE)
      metric.
    smatch_threshold: Threshold for defining a good smatch score.

  Returns:
    A dictionary of metric values.
  """
  log_probs = aux_values.get('scores', None)
  log_probs = np.array(log_probs, dtype=np.float32)

  if log_probs is None:
    raise ValueError('Cannot find the field `scores` from `aux_values`.')
  # The `log_probs` should be token-level probabilties,
  # with shape (batch_size, max_seq_length)
  if log_probs.ndim != 2:
    raise ValueError('Incorrect dimension for `log_probs`, which should be 2.'
                     f'Got {log_probs.ndim}.')

  all_node_prob_match_list = []
  all_attr_prob_match_list = []
  all_edge_prob_match_list = []
  all_smatch_score_list = []
  seq_log_probs = []
  for pred, target, log_prob in zip(predictions, targets, log_probs):
    token_ids = vocab.encode(pred)
    tokens = [
        metrics_utils.single_token_transfer(vocab.decode([id]), dataset_name)
        for id in token_ids
    ]
    seq_log_probs.append(np.sum(log_prob[:len(tokens)+1]))
    log_prob = log_prob[:len(tokens)]

    pred_penman_with_prob = penman_utils.assign_prob_to_penman_for_dataflow(
        tokens, log_prob, dataset_name)
    try:
      _ = graph_utils.parse_string_to_dag(pred_penman_with_prob)
    except LookupError:
      # The predicted graph here is an ill-formed graph. Skip.
      logging.warning('Fail to parse DAG: %s', pred_penman_with_prob)
      continue

    pred_penman = penman_utils.PENMANStr(
        pred, variable_free=True, retokenized=True,
        data_version=dataset_name).penman
    gold_penman = penman_utils.PENMANStr(
        target, variable_free=True, retokenized=True,
        data_version=dataset_name).penman
    _, _, smatch_score = graph_utils.get_smatch(pred_penman, gold_penman)
    all_smatch_score_list.append(smatch_score)

    # Generates a list of predictive probability and predictive accuracy for
    # node/attribute/edge predictions.
    (node_prob_match_list, attr_prob_match_list,
     edge_prob_match_list) = graph_utils.get_dag_match_for_calibration(
         pred_penman_with_prob, gold_penman)
    all_node_prob_match_list += node_prob_match_list
    all_attr_prob_match_list += attr_prob_match_list
    all_edge_prob_match_list += edge_prob_match_list

  def compositional_eval(prob_match_list):
    model_pred_confs = np.array([p for (p, _) in prob_match_list],
                                dtype=np.float32)
    model_pred_confs_ece = np.stack([1. - model_pred_confs, model_pred_confs],
                                    axis=-1)
    correct_predictions = np.array([c for (_, c) in prob_match_list])
    correct_predictions_fl = np.array(correct_predictions, dtype=np.float32)
    ece = rm_metrics.ExpectedCalibrationError(num_bins=num_ece_bins)
    ece.add_batch(model_pred_confs_ece, label=correct_predictions)
    calib_auroc = rm_metrics.CalibrationAUC(
        curve='ROC', correct_pred_as_pos_label=False)
    calib_auprc = rm_metrics.CalibrationAUC(
        curve='PR', correct_pred_as_pos_label=False)
    calib_auroc.add_batch(
        np.ones_like(correct_predictions_fl),
        confidence=model_pred_confs,
        label=correct_predictions_fl)
    calib_auprc.add_batch(
        np.ones_like(correct_predictions_fl),
        confidence=model_pred_confs,
        label=correct_predictions_fl)
    return ece, calib_auroc, calib_auprc

  # Node evaluation
  node_ece, node_calib_auroc, node_calib_auprc = compositional_eval(
      all_node_prob_match_list)
  # Node evaluation
  attr_ece, attr_calib_auroc, attr_calib_auprc = compositional_eval(
      all_attr_prob_match_list)
  # Edge evaluation
  edge_ece, edge_calib_auroc, edge_calib_auprc = compositional_eval(
      all_edge_prob_match_list)
  # Uncertainty metrics based on smtach
  model_pred_confs = np.exp(np.array(seq_log_probs, dtype=np.float32))
  model_pred_confs_ece = np.stack([1. - model_pred_confs, model_pred_confs],
                                  axis=-1)
  correct_predictions = np.array(
      [s >= smatch_threshold for s in all_smatch_score_list])
  correct_predictions_fl = np.array(correct_predictions, dtype=np.float32)

  smatch_ece = rm_metrics.ExpectedCalibrationError(num_bins=num_ece_bins)
  smatch_calib_auroc = rm_metrics.CalibrationAUC(
      curve='ROC', correct_pred_as_pos_label=False)
  smatch_calib_auprc = rm_metrics.CalibrationAUC(
      curve='PR', correct_pred_as_pos_label=False)

  smatch_ece.add_batch(model_pred_confs_ece, label=correct_predictions)
  smatch_calib_auroc.add_batch(
      np.ones_like(correct_predictions_fl),
      confidence=model_pred_confs,
      label=correct_predictions_fl)
  smatch_calib_auprc.add_batch(
      np.ones_like(correct_predictions_fl),
      confidence=model_pred_confs,
      label=correct_predictions_fl)

  analysis = dict(
      node_ece=node_ece.result()['ece'],
      node_calib_auroc=node_calib_auroc.result()['calibration_auc'],
      node_calib_auprc=node_calib_auprc.result()['calibration_auc'],
      attr_ece=attr_ece.result()['ece'],
      attr_calib_auroc=attr_calib_auroc.result()['calibration_auc'],
      attr_calib_auprc=attr_calib_auprc.result()['calibration_auc'],
      edge_ece=edge_ece.result()['ece'],
      edge_calib_auroc=edge_calib_auroc.result()['calibration_auc'],
      edge_calib_auprc=edge_calib_auprc.result()['calibration_auc'],
      smatch_ece=smatch_ece.result()['ece'],
      smatch_calib_auroc=smatch_calib_auroc.result()['calibration_auc'],
      smatch_calib_auprc=smatch_calib_auprc.result()['calibration_auc'],
  )
  return analysis


def binary_classification(
    targets: Sequence[str],
    scores: Sequence[Union[GreedyScore, BeamScore]],
    label_tokens: Sequence[Union[str, int]],
    prediction_threshold: float = 0.5,
    num_ece_bins: int = 15,
    auc_temperatures: Sequence[float] = (0.5, 1., 1.5, 2.)
) -> Dict[str, float]:
  """Computes binary classification metrics.

  This metric is intended to be used with the EncoderDecoderClassifierModel that
  scores each example with predictive logits for all the possible classes.

  Args:
    targets: A sequence of strings for predicted token ids. Shape (batch_size,).
    scores: A sequence of float. A nested array of logit scores for each
      possible labels. Shape (batch_size, output_len, num_classes).
      Alternatively, it can be a sequence of 3-tuples, where the first element
      is the logit scores described previously, and the second and third
      elements are the top-k predictions (a sequence of strings with shape
      (beam_size, )) and the associated token-level log probabilities (an array
      of shape (beam_size, output_len)). They will be ignored in the score
      computation for this function.
    label_tokens: A list of label tokens (e.g.,'<extra_id_0>') for the output
      classes. Shape (num_classes,).
    prediction_threshold: The numeric threshold to convert the predictive
      probability to the predicted class, i.e., pred = I(prob > threshold).
    num_ece_bins: The number of bins to be used for computing expected
      calibration error.
    auc_temperatures: A list of temperature parameters to be used for computing
      the optimal AUC based on temperature-adjusted logits.

  Raises:
    ValueError: raise when the length of the label tokens is not 2.
    ValueError: raise when the rank of the scores is not 3.
    ValueError: raise when the output_len (i.e., score_shape[1]) is not 1.
    ValueError: raise when the num_classes (i.e., score_shape[2]) is not 2.

  Returns:
    Predictive metrics (Accuracy, F1, AUC scores), calibration metrics
    (ECE and Calibration AUC), and selective prediction metrics
    (Oracle Collaborative AUCs).
  """
  if isinstance(scores[0], tuple):
    # If input is a sequence of BeamScore (i.e., a tuple whose first element
    # is the GreedyScore), convert it to List[GreedyScore].
    scores = [beam_score[0] for beam_score in scores]

  if isinstance(scores, tuple):
    scores = scores[0]

  scores = np.array(scores, dtype=np.float32)

  # Check score shape.
  score_shape = scores.shape
  if len(label_tokens) != 2:
    raise ValueError('`label_tokens` should only contain 2 labels. '
                     f'Got {len(label_tokens)}: {label_tokens}')
  if len(score_shape) != 3:
    raise ValueError(
        '`scores` should be a 3-D tensor with shape (batch_size, output_len, '
        f'vocab_size). Got shape {score_shape}')
  if score_shape[1] != 1:
    raise ValueError('For binary classification, the output len should be 1.'
                     f' Got {score_shape[1]}')
  if score_shape[2] != 2:
    raise ValueError('For binary classification, the num_class should be 2.'
                     f' Got {score_shape[2]}')

  # Converts target tokens to binary scores (i.e., '<extra_id_0>' to 0 and
  # '<extra_id_1>' to 1).
  label_tokens_to_ids = {
      token: token_id for token_id, token in enumerate(label_tokens)
  }
  target_labels = np.array([label_tokens_to_ids[token] for token in targets])

  # Converts logit scores to class probabilities. This is done by squeeze score
  # to (batch_size, num_classes) and then apply softmax to compute the
  # predictive probability for all classes (`model_class_probs`) and also for
  # the predicted class (`model_pred_probs`).
  scores = np.squeeze(scores, axis=1)
  model_class_probs = sp_special.softmax(scores, axis=-1)
  model_pred_probs = model_class_probs[:, 1]

  predicted_labels = (model_pred_probs > prediction_threshold).astype(int)
  model_pred_confs = model_class_probs[np.arange(score_shape[0]),
                                       predicted_labels]

  # Creates float32 versions of result tensors to ensure compatibility with the
  # tensorflow metrics.
  target_labels_fl = np.array(target_labels).astype(np.float32)
  predicted_labels_fl = np.array(predicted_labels).astype(np.float32)
  model_class_probs = np.array(model_class_probs).astype(np.float32)
  model_pred_probs = np.array(model_pred_probs).astype(np.float32)
  model_pred_confs = np.array(model_pred_confs).astype(np.float32)

  # Computes the accuracy, F1 and AUC metrics.
  acc = sk_metrics.accuracy_score(target_labels, predicted_labels)
  nll = sk_metrics.log_loss(target_labels, model_pred_probs)
  f1 = sk_metrics.f1_score(target_labels, predicted_labels)
  auc_roc = tf_metrics.AUC(curve='ROC')(target_labels_fl, model_pred_probs)
  auc_prc = tf_metrics.AUC(curve='PR')(target_labels_fl, model_pred_probs)

  # Computes the optimal AUC based on temperature-adjusted logits.
  auc_rocs_temperature_adjusted = []
  auc_prcs_temperature_adjusted = []
  for temperature in auc_temperatures:
    temp_adjusted_logits = scores / temperature
    temp_adjusted_probs = sp_special.softmax(
        temp_adjusted_logits, axis=-1)[:, 1]
    temp_adjusted_probs = temp_adjusted_probs.astype(np.float32)

    auc_roc_temp = tf_metrics.AUC(curve='ROC')(target_labels_fl,
                                               temp_adjusted_probs)
    auc_prc_temp = tf_metrics.AUC(curve='PR')(target_labels_fl,
                                              temp_adjusted_probs)

    auc_rocs_temperature_adjusted.append(auc_roc_temp.numpy())
    auc_prcs_temperature_adjusted.append(auc_prc_temp.numpy())

  auc_roc_optimal_temperature = np.max(auc_rocs_temperature_adjusted)
  auc_prc_optimal_temperature = np.max(auc_prcs_temperature_adjusted)

  # Defines and computes calibration metrics.
  ece = rm_metrics.ExpectedCalibrationError(num_bins=num_ece_bins)
  calib_auroc = rm_metrics.CalibrationAUC(
      curve='ROC', correct_pred_as_pos_label=False)
  calib_auprc = rm_metrics.CalibrationAUC(
      curve='PR', correct_pred_as_pos_label=False)

  ece.add_batch(model_class_probs, label=target_labels)
  calib_auroc.add_batch(
      predicted_labels_fl, confidence=model_pred_confs, label=target_labels_fl)
  calib_auprc.add_batch(
      predicted_labels_fl, confidence=model_pred_confs, label=target_labels_fl)

  # Defines and computes selective prediction metrics.
  collab_auprc_1 = rm_metrics.OracleCollaborativeAUC(
      oracle_fraction=0.01, curve='PR')
  collab_auroc_1 = rm_metrics.OracleCollaborativeAUC(
      oracle_fraction=0.01, curve='ROC')

  collab_auprc_2 = rm_metrics.OracleCollaborativeAUC(
      oracle_fraction=0.02, curve='PR')
  collab_auroc_2 = rm_metrics.OracleCollaborativeAUC(
      oracle_fraction=0.02, curve='ROC')

  collab_auprc_5 = rm_metrics.OracleCollaborativeAUC(
      oracle_fraction=0.05, curve='PR')
  collab_auroc_5 = rm_metrics.OracleCollaborativeAUC(
      oracle_fraction=0.05, curve='ROC')

  collab_auprc_10 = rm_metrics.OracleCollaborativeAUC(
      oracle_fraction=0.1, curve='PR')
  collab_auroc_10 = rm_metrics.OracleCollaborativeAUC(
      oracle_fraction=0.1, curve='ROC')

  collab_auprc_1.add_batch(model_pred_probs, label=target_labels_fl)
  collab_auprc_2.add_batch(model_pred_probs, label=target_labels_fl)
  collab_auprc_5.add_batch(model_pred_probs, label=target_labels_fl)
  collab_auprc_10.add_batch(model_pred_probs, label=target_labels_fl)

  collab_auroc_1.add_batch(model_pred_probs, label=target_labels_fl)
  collab_auroc_2.add_batch(model_pred_probs, label=target_labels_fl)
  collab_auroc_5.add_batch(model_pred_probs, label=target_labels_fl)
  collab_auroc_10.add_batch(model_pred_probs, label=target_labels_fl)

  return {
      'accuracy': acc * 100,
      'nll': nll,
      'f1': f1 * 100,
      'auroc': auc_roc.numpy(),
      'auprc': auc_prc.numpy(),
      'auroc_temperature_adjusted': auc_roc_optimal_temperature,
      'auprc_temperature_adjusted': auc_prc_optimal_temperature,
      'ece': ece.result()['ece'],
      'calib_auroc': calib_auroc.result()['calibration_auc'],
      'calib_auprc': calib_auprc.result()['calibration_auc'],
      'collab_auroc_1%': collab_auroc_1.result()['collaborative_auc'],
      'collab_auroc_2%': collab_auroc_2.result()['collaborative_auc'],
      'collab_auroc_5%': collab_auroc_5.result()['collaborative_auc'],
      'collab_auroc_10%': collab_auroc_10.result()['collaborative_auc'],
      'collab_auprc_1%': collab_auprc_1.result()['collaborative_auc'],
      'collab_auprc_2%': collab_auprc_2.result()['collaborative_auc'],
      'collab_auprc_5%': collab_auprc_5.result()['collaborative_auc'],
      'collab_auprc_10%': collab_auprc_10.result()['collaborative_auc'],
  }


def _log_softmax(x, axis=-1):
  x_max = np.max(x, axis=axis, keepdims=True)
  x_shifted = x - x_max
  return x_shifted - sp_special.logsumexp(x_shifted, axis=axis, keepdims=True)


def sequence_classification(
    targets: Sequence[str],
    scores: Sequence[Union[GreedyScore, BeamScore]],
    label_tokens: Sequence[Union[str, int]],
    num_ece_bins: int = 15,
    uncertainty_type: str = 'conditional_entropy',
    return_collab_accuracy: bool = False,
    oos_token_id: Optional[int] = None,
) -> Dict[str, float]:
  """Computes sequence-level uncertainty metrics.

  This metric is intended to be used with the EncoderDecoderClassifierModel that
  scores each example with predictive logits for all the possible classes.

  Notice that when the classification problem is compositional (i.e., a sequence
  label problem where targets has shape (batch_size, output_len) where
  output_len > 1), the logit scores is assumed to be the top-1 logits (i.e., the
  logit prediction p(x_k+1|x_k) is conditioned on the top-1 prediction for x_k).

  Given the log probability score matrix log P( x_k+1 | x_k ) (shape
  [batch_size, output_len, num_class]), the uncertainty for the whole sequence
  (which has length = output_len) can be summarized as one of the two ways [1]:

    * Entropy:  -(1/L) * sum[ log P( x_k+1 | x_k ) ].
    * Conditional Entropy:  -(1/L) * sum[ H( x_k+1 | x_k ) ].

  where H( x_k+1 | x_k ) is the conditional entropy computed based on the
  conditional distribution P( x_k+1 | x_k ).

  Note:

    1. The entropy can be understood as a special case of conditional entropy,
    where H is computed by integrating over a Delta distribution concentrated
    at argmax P(x_k+1|x_k) (instead of the original distribution
    P( x_k+1 | x_k )).
    2. Between the two estimators, the conditional entropy is suggested to be
    more stable in the small samples [1].
    3. Entropy as a uncertainty metric is approperiate when the output_len is
    known and fixed. For problems where the output sequence has varying length,
    there may exist a undesired correlation between sequence length and entropy
    magnitude (i.e., shorter sequence tend to have lower entropy), making
    entropy less effective as a calibration metric.

  ## Reference

  [1]: Andrey Malinin, Mark Gales. Uncertainty Estimation in Autoregressive
       Structured Prediction. In _International Conference on Learning
       Representations_, 2020.
       https://arxiv.org/abs/2002.07650

  Args:
    targets: A sequence of strings for the target strings. In the case of
      sequence prediction, the string takes the form '<token_1> <token_2>
      <token_3> ...'. Shape (batch_size,).
    scores: A sequence of float. A nested array of logit scores for each
      possible labels. Shape (batch_size, output_len, num_classes).
      Alternatively, it can be a sequence of 3-tuples, where the first element
      is the logit scores described previously, and the second and third
      elements are the top-k predictions (a sequence of strings with shape
      (beam_size, )) and the associated token-level log probabilities
      (an array of shape (beam_size, output_len)). They will be ignored in the
      score computation for this function.
    label_tokens: A list of label tokens for the output classes. Shape
      (num_classes,).
    num_ece_bins: The number of bins to be used for computing expected
      calibration error.
    uncertainty_type: The type of sequence-level uncertainty to compute. Must be
      one of (`entropy`, `conditional_entropy`).
    return_collab_accuracy: Whether to report collaborative accuracy.
    oos_token_id: The token ID for the intent label `OOS` (out-of-domain). If
      oos_token_id is not None, OOD evaluation will be turned on.

  Raises:
    ValueError: raise when the rank of the scores is not 3.
    ValueError: raise when the num_classes (i.e., score_shape[2]) does not
      match the number of tokens in the label_tokens.
    ValueError: raise when the uncertainty_type does not belong to
      (`entropy`, `conditional_entropy`).
    ValueError: raise when the number of tokens in a target string does not
      equal to the output_len of the `scores` tensor.

  Returns:
    Predictive metrics (Accuracy, F1, AUC scores), calibration metrics
    (ECE and Calibration AUC), and selective prediction metrics
    (Oracle Collaborative AUCs).
  """
  if isinstance(scores[0], tuple):
    # If input is a sequence of BeamScore (i.e., a tuple whose first element
    # is the GreedyScore), convert it to List[GreedyScore].
    scores = [beam_score[0] for beam_score in scores]

  scores = np.array(scores, dtype=np.float32)

  # Validates input shapes.
  score_shape = scores.shape  # (batch_size, output_len, num_class)
  output_len = score_shape[1]

  if len(score_shape) != 3:
    raise ValueError(
        '`scores` should be a 3-D tensor with shape (batch_size, output_len, '
        f'num_class). Got shape {score_shape}')

  if score_shape[2] != len(label_tokens):
    raise ValueError(
        f'The number of classes in score tensor ({score_shape[2]}) does not '
        f'match the number of label tokens ({len(label_tokens)}).')

  if uncertainty_type not in ('entropy', 'conditional_entropy'):
    raise ValueError(
        'The uncertainty type must be one of ("entropy", "conditional_entropy")'
        f'. Got "{uncertainty_type}".')

  # Converts target tokens into label ids.
  label_tokens_to_ids = {
      token: token_id for token_id, token in enumerate(label_tokens)
  }

  target_labels = []
  for token_seq in targets:
    tokens = token_seq.split(' ')
    target_labels.append([label_tokens_to_ids[token] for token in tokens])

    if len(tokens) != output_len:
      raise ValueError(f'Expects {output_len} tokens in the target string '
                       f'"{token_seq}". Got {len(tokens)}: {tokens}.')

  target_labels = np.array(target_labels)  # Shape (batch_size, output_len).

  # Computes model prediction.
  model_class_log_probs = _log_softmax(scores, axis=-1)
  logging.info('The shape of model_class_log_probs %s',
               model_class_log_probs.shape)
  predicted_labels = np.argmax(model_class_log_probs, axis=-1)

  correct_prediction = np.all(predicted_labels == target_labels, axis=-1)

  # Computes model uncertainty as sequence-level entropy.
  # Shape (batch_size, output_len)
  if uncertainty_type == 'conditional_entropy':
    token_entropy = np.sum(
        np.exp(model_class_log_probs) * model_class_log_probs, axis=-1)
  else:
    token_entropy = np.max(model_class_log_probs, axis=-1)

  sequence_entropy = np.mean(token_entropy, axis=-1)
  # Exponentiate so uncertainty stays within range [0, 1].
  model_pred_confs = np.exp(sequence_entropy)

  # Prepares result tensor for metric computation.
  correct_predictions_fl = np.array(correct_prediction, dtype=np.float32)
  model_pred_confs_ece = np.stack([1. - model_pred_confs, model_pred_confs],
                                  axis=-1)

  # Performance metrics.
  acc = np.mean(correct_predictions_fl)
  auc_roc = tf_metrics.AUC(curve='ROC')(correct_predictions_fl,
                                        model_pred_confs)
  auc_prc = tf_metrics.AUC(curve='PR')(correct_predictions_fl, model_pred_confs)

  # Calibration metrics.
  # To utilize the classification metrics, we evaluate sequence-level
  # uncertainty as a binary problem where the model prediction is always 1, and
  # the target label is 1 if model prediction is correct, and 0 otherwise.
  labels = [False, True]
  nll = sk_metrics.log_loss(correct_prediction, model_pred_confs, labels=labels)
  ece = rm_metrics.ExpectedCalibrationError(num_bins=num_ece_bins)
  calib_auroc = rm_metrics.CalibrationAUC(
      curve='ROC', correct_pred_as_pos_label=False)
  calib_auprc = rm_metrics.CalibrationAUC(
      curve='PR', correct_pred_as_pos_label=False)

  ece.add_batch(model_pred_confs_ece, label=correct_prediction)
  calib_auroc.add_batch(
      np.ones_like(correct_predictions_fl),
      confidence=model_pred_confs,
      label=correct_predictions_fl)
  calib_auprc.add_batch(
      np.ones_like(correct_predictions_fl),
      confidence=model_pred_confs,
      label=correct_predictions_fl)

  result_dict = {
      'accuracy': float(acc) * 100,
      'nll': nll,
      'auroc': auc_roc.numpy(),
      'auprc': auc_prc.numpy(),
      'ece': ece.result()['ece'],
      'calib_auroc': calib_auroc.result()['calibration_auc'],
      'calib_auprc': calib_auprc.result()['calibration_auc'],
  }

  # Defines and computes selective prediction metrics.
  if return_collab_accuracy:
    collab_acc_1 = rm_metrics.OracleCollaborativeAccuracy(fraction=0.01)
    collab_acc_2 = rm_metrics.OracleCollaborativeAccuracy(fraction=0.02)
    collab_acc_5 = rm_metrics.OracleCollaborativeAccuracy(fraction=0.05)
    collab_acc_10 = rm_metrics.OracleCollaborativeAccuracy(fraction=0.1)

    collab_acc_kwargs = dict(
        model_predictions=np.ones_like(correct_predictions_fl),
        label=correct_predictions_fl,
        custom_binning_score=model_pred_confs)

    collab_acc_1.add_batch(**collab_acc_kwargs)
    collab_acc_2.add_batch(**collab_acc_kwargs)
    collab_acc_5.add_batch(**collab_acc_kwargs)
    collab_acc_10.add_batch(**collab_acc_kwargs)

    result_dict.update({
        'collab_acc_1%': collab_acc_1.result()['collaborative_accuracy'],
        'collab_acc_2%': collab_acc_2.result()['collaborative_accuracy'],
        'collab_acc_5%': collab_acc_5.result()['collaborative_accuracy'],
        'collab_acc_10%': collab_acc_10.result()['collaborative_accuracy']
    })

  if oos_token_id is not None:
    # OOD metrics
    is_oos = np.any(target_labels == oos_token_id, axis=-1)
    logging.info('The bool is_oos %s', is_oos)
    auc_roc_ood = tf_metrics.AUC(curve='ROC')(is_oos, 1 - model_pred_confs)
    auc_prc_ood = tf_metrics.AUC(curve='PR')(is_oos, 1 - model_pred_confs)

    result_dict.update({
        'ood_auc_roc': auc_roc_ood.numpy(),
        'ood_auc_prc': auc_prc_ood.numpy(),
    })

  return result_dict


def sequence_classification_beam_metrics(
    targets: Sequence[str],
    scores: Sequence[BeamScore],
    vocab: seqio.SentencePieceVocabulary,
    beam_type: str = 'all_beam',
    uncertainty_type: str = 'probability',
    num_ece_bins: int = 15,
    return_accuracy: bool = False,
    return_collab_accuracy: bool = False) -> Dict[str, float]:
  """Computes sequence-level classification metrics for beam prediction.

  This function computes the uncertainty metrics of a top-K beam prediction
  from a seq2seq model, which is analogous to top-K prediction in
  classification. In particular, we first compute the cumulative accuracy
  and probability of a top-K beam prediction (either including or excluding
  top-1), and use them to compute the downstream metrics, including accuracy,
  calibration, and collaborative accuracy.


  Args:
    targets: A sequence of strings for the target strings. In the case of
      sequence prediction, the string takes the form '<token_1> <token_2>
      <token_3> ...'. Shape (batch_size,).
    scores: A sequence of 3-tuples (greedy_score, beam_prediction, beam_score).
      The `greedy_score` is a nested array of logit scores for each possible
      labels, shape (batch_size, output_len, num_classes). The second element is
      the top-k predictions (a sequence of token ids with shape (batch_size,
      beam_size)). The third element is the associated token-level log
      probabilities (an array of shape (batch_size, beam_size, output_len)). For
      more detail, see the `score_batch` function of
      `models.EncoderDecoderClassifierModel`.
    vocab: The SentencePieceVocabulary used for model training.
    beam_type: Type of beam prediction to compute metric for, must be one of
      ('all_beam', 'non_top1_beam', 'top1_beam').
    uncertainty_type: The type of sequence-level uncertainty to compute. Must be
      one of ('probability', 'margin', 'entropy').
    num_ece_bins: Number of bins to use for Expected calibration error.
    return_accuracy: Whether to report accuracy.
    return_collab_accuracy: Whether to report collaborative accuracy.

  Returns:
    A dictionary of top-K beam prediction metrics, including accuracy,
    calibration, and collaborative AUC of the whole beam, the non-top-1 beam,
    or the token-level probability.
  """
  # Extracts beam result from scores.
  beam_predictions, beam_scores = _extract_beam_results_from_scores(scores)

  # Computes beam accuracy, shape (batch_size, ).
  correct_predictions = _compute_beam_correctness(
      targets, beam_predictions, vocab=vocab, beam_type=beam_type)

  correct_predictions_fl = np.array(correct_predictions, dtype=np.float32)

  # Computes beam uncertainty, shape (batch_size, ).
  if np.any(beam_scores > 0):
    raise ValueError('`beam_scores` should to be log probability. However, '
                     'some element of `beam_scores` is positive.')

  beam_confidences = _compute_beam_uncertainty(
      beam_predictions,
      beam_scores,
      beam_type=beam_type,
      uncertainty_type=uncertainty_type)
  beam_confidences_ece = np.stack([1. - beam_confidences, beam_confidences],
                                  axis=-1)

  if np.any(beam_confidences > 1.):
    max_conf_id = np.argmax(beam_confidences)
    logging.info(
        '`beam_confidences` must be within range (0, 1) when '
        'uncertainty_type=%s and beam_type=%s. '
        'However, max(beam_confidences) is %s '
        'Applying clipping with max=1.', uncertainty_type, beam_type,
        beam_confidences[max_conf_id])
    beam_confidences = np.clip(beam_confidences, a_min=0., a_max=1.)
    beam_confidences_ece = np.clip(beam_confidences_ece, a_min=0., a_max=1.)

  # Computes calibration metrics.
  ece = rm_metrics.ExpectedCalibrationError(num_bins=num_ece_bins)
  calib_auroc = rm_metrics.CalibrationAUC(
      curve='ROC', correct_pred_as_pos_label=False)
  calib_auprc = rm_metrics.CalibrationAUC(
      curve='PR', correct_pred_as_pos_label=False)

  ece.add_batch(beam_confidences_ece, label=correct_predictions)
  calib_auroc.add_batch(
      np.ones_like(correct_predictions_fl),
      confidence=beam_confidences,
      label=correct_predictions_fl)
  calib_auprc.add_batch(
      np.ones_like(correct_predictions_fl),
      confidence=beam_confidences,
      label=correct_predictions_fl)

  result_dict = {
      f'ece_{beam_type}_{uncertainty_type}':
          ece.result()['ece'],
      f'calib_auroc_{beam_type}_{uncertainty_type}':
          calib_auroc.result()['calibration_auc'],
      f'calib_auprc_{beam_type}_{uncertainty_type}':
          calib_auprc.result()['calibration_auc'],
  }

  # Optionally, computes (collaborative) accuracy metrics.
  if return_accuracy:
    acc = np.mean(correct_predictions_fl)
    result_dict.update({
        f'accuracy_{beam_type}': float(acc) * 100,
    })

  if return_collab_accuracy:
    collab_acc_1 = rm_metrics.OracleCollaborativeAccuracy(fraction=0.01)
    collab_acc_2 = rm_metrics.OracleCollaborativeAccuracy(fraction=0.02)
    collab_acc_5 = rm_metrics.OracleCollaborativeAccuracy(fraction=0.05)
    collab_acc_10 = rm_metrics.OracleCollaborativeAccuracy(fraction=0.1)

    collab_acc_kwargs = dict(
        model_predictions=np.ones_like(correct_predictions_fl),
        label=correct_predictions_fl,
        custom_binning_score=beam_confidences)

    collab_acc_1.add_batch(**collab_acc_kwargs)
    collab_acc_2.add_batch(**collab_acc_kwargs)
    collab_acc_5.add_batch(**collab_acc_kwargs)
    collab_acc_10.add_batch(**collab_acc_kwargs)

    result_dict.update({
        f'collab_acc_1%_{beam_type}_{uncertainty_type}':
            collab_acc_1.result()['collaborative_accuracy'],
        f'collab_acc_2%_{beam_type}_{uncertainty_type}':
            collab_acc_2.result()['collaborative_accuracy'],
        f'collab_acc_5%_{beam_type}_{uncertainty_type}':
            collab_acc_5.result()['collaborative_accuracy'],
        f'collab_acc_10%_{beam_type}_{uncertainty_type}':
            collab_acc_10.result()['collaborative_accuracy']
    })

  return result_dict


def topk_collaborative_accuracy(
    targets: Sequence[str],
    scores: Sequence[BeamScore],
    vocab: seqio.SentencePieceVocabulary,
    beam_type: str = 'all_beam',
    uncertainty_type: str = 'probability') -> Dict[str, float]:
  """Computes uncertainty-based collaborative accuracy among topk predictions.

  The Top-K collaborative accuracy measures the effectiveness of uncertainty
  in improving a collaborative system's decision in the below game:

  Consider a collaborative game, where the model is allowed to either directly
  predict using its top-1 prediction, or to issue a clarification among its
  top k predictions. The decision between prediction and clarification is
  based on its uncertainty, where issuing a clarification costs -0.5.

  Then, the reward for different scenarios:

   * Right top1 pred, no clarification: +1.
   * Wrong top1 pred, no clarification: +0.
   * Right top1 pred, issue clarification: -0.5 + 1. = +0.5
   * Wrong top1 pred, issue clarification, right topk guess: -0.5+1.0 = +0.5
   * Wrong top1 pred, issue clarification, wrong topk guess: -0.5+0.75= +0.25

  Notice that the last scenario (wrong topk & issue clarification) still gets a
  small positive reward. This is because the system is still partially correct
  (i.e., correct clarification, incorrect top-K guess). In this case, the user
  can still take advantage of the clarification oppurtunity to guide the AI
  system toward the correct action. In this sense, it is better than wrong top1
  prediction with no clarification, since in that case, the AI system is
  directly generating a wrong response without creating an opportunity for human
  intervation.

  Args:
    targets: A sequence of strings for the target strings. In the case of
      sequence prediction, the string takes the form '<token_1> <token_2>
      <token_3> ...'. Shape (batch_size,).
    scores: A sequence of 3-tuples (greedy_score, beam_prediction, beam_score).
      The `greedy_score` is a nested array of logit scores for each possible
      labels, shape (batch_size, output_len, num_classes). The second element is
      the top-k predictions (a sequence of token ids with shape (batch_size,
      beam_size)). The third element is the associated token-level log
      probabilities (an array of shape (batch_size, beam_size, output_len)). For
      more detail, see the `score_batch` function of
      `models.EncoderDecoderClassifierModel`.
    vocab: The SentencePieceVocabulary used for model training.
    beam_type: Type of beam prediction to compute metric for, must be one of
      ('all_beam', 'non_top1_beam', 'top1_beam').
    uncertainty_type: The type of sequence-level uncertainty to compute. Must be
      one of ('probability', 'margin', 'entropy').

  Returns:
    A dictionary of metrics for the top-K collaborative performance:

      * False Discovery Rate (fdr): Percentage of unnecessary clarification
          among all clarifications.
      * False Negative Rate (fnr): Percentage of false negative (didn't issue
          clarification when should) among all cases that needs clarification.
      * Collaboration Rate (rate): Percentage of cases where a clarification is
          issued.
      * Collaboration Reward (reward): Total reward of the aforementioned
          collaborative game.
      * Uncertainty threshold (threshold): The best threshold that is used
          for making the uncertainty-based clarification decision.
  """
  # Extracts beam result from scores.
  beam_predictions, beam_scores = _extract_beam_results_from_scores(scores)

  # Computes beam confidence.
  beam_confidences = _compute_beam_uncertainty(
      beam_predictions,
      beam_scores,
      beam_type=beam_type,
      uncertainty_type=uncertainty_type)

  # Computes top-1 and top-K correctness.
  correct_predictions_top1 = _compute_beam_correctness(
      targets, beam_predictions, vocab=vocab, beam_type='top1_beam')
  correct_predictions_topk = _compute_beam_correctness(
      targets, beam_predictions, vocab=vocab, beam_type='all_beam')

  correct_predictions_top1 = np.array(
      correct_predictions_top1, dtype=np.float32)
  correct_predictions_topk = np.array(
      correct_predictions_topk, dtype=np.float32)

  # Computes collaboration decision based on model confidence, shape
  # (batch_size, num_thresholds).
  decision_threshold = np.arange(0.01, 1., 0.01)
  if beam_type == 'non_top1_beam':
    issue_collab = beam_confidences[:, None] > decision_threshold[None, :]
  else:
    issue_collab = beam_confidences[:, None] < decision_threshold[None, :]

  # Computes results for collaboration performance and the total rewards, shapes
  # (batch_size, num_thresholds).
  correct_top1 = correct_predictions_top1[:, None]
  correct_topk = correct_predictions_topk[:, None]

  collab_tp = np.sum((1. - correct_top1) * issue_collab, axis=0)
  collab_fn = np.sum((1. - correct_top1) * (1. - issue_collab), axis=0)
  collab_fp = np.sum(correct_top1 * issue_collab, axis=0)

  collab_fdr = collab_fp / (collab_fp + collab_tp)
  collab_fnr = collab_fn / (collab_fn + collab_tp)

  collab_count = np.mean(issue_collab, axis=0)
  collab_accuracy = np.mean(
      (correct_top1 + issue_collab * correct_topk) > 0, axis=0)
  collab_rewards = np.mean(
      1.0 * correct_top1 - 0.5 * issue_collab + 1.0 * issue_collab *
      (1. - correct_top1) * correct_topk + 0.75 * issue_collab *
      (1. - correct_top1) * (1. - correct_topk),
      axis=0)

  # Selects best threshold based on reward.
  collab_thresh_id = np.argmax(collab_rewards)

  return {
      f'topk_collab_acc_{beam_type}_{uncertainty_type}':
          float(collab_accuracy[collab_thresh_id]) * 100,
      f'topk_collab_fdr_{beam_type}_{uncertainty_type}':
          float(collab_fdr[collab_thresh_id]) * 100,
      f'topk_collab_fnr_{beam_type}_{uncertainty_type}':
          float(collab_fnr[collab_thresh_id]) * 100,
      f'topk_collab_rate_{beam_type}_{uncertainty_type}':
          float(collab_count[collab_thresh_id]) * 100,
      f'topk_collab_reward_{beam_type}_{uncertainty_type}':
          float(collab_rewards[collab_thresh_id]) * 100,
      f'topk_collab_threshold_{beam_type}_{uncertainty_type}':
          float(decision_threshold[collab_thresh_id])
  }


def _extract_beam_results_from_scores(
    scores: Sequence[BeamScore]) -> Tuple[List[TopkPrediction], np.ndarray]:
  """Extracts Beam predictions and log probability from model scores.

  Args:
    scores: A sequence of 3-tuples, where the first element is a nested array of
      logit scores for each possible labels, and the second and third elements
      are the top-k predictions (a sequence of strings with shape (beam_size, ))
      and the associated token-level log probabilities (an array of shape
      (beam_size, output_len)). They will be ignored in the score computation
      for this function.

  Returns:
    beam_predictions: A list of token ids for top-K predictions, shape
      (batch_size, beam_size, output_len).
    beam_scores: A float np.ndarray for token-level log probabilities.

  Raises:
    ValueError: If score is not a list of tuples.
    ValueError: If score's tuple element does not contain three elements.
  """
  # Verifies types and lengths of score elements.
  if not isinstance(scores[0], tuple):
    raise ValueError('`scores` must be a sequence of tuples. '
                     f'But `type(scores[0])`={type(scores[0])}')
  elif len(scores[0]) != 3:
    raise ValueError('`scores` must be a sequence of 3-tuples. '
                     f'But `len(scores[0])`={len(scores[0])}')

  # Parses `scores` into `beam_predictions` and `beam_scores`, with shapes
  # List[int32] with shape (batch_size, beam_size) and
  # ndarray[float32] with shape (batch_size, beam_size, output_len).
  # Both scores and predictions are sorted in the increasing order of `scores`
  # (see t5x.decoding.beam_search for detail), and scores for null predictions
  # are set to NEG_INF.
  beam_predictions = [beam_score[1] for beam_score in scores]
  beam_scores = [beam_score[2] for beam_score in scores]
  beam_scores = np.array(beam_scores, dtype=np.float32)

  if np.any(beam_scores > 0):
    logging.info(
        '`beam_scores` should to be log probability. However, '
        'some element of `beam_scores` is positive. (Percent positive:'
        '%s, range: (%s, %s). Thresholding at 0.',
        np.mean(beam_scores > 0.) * 100, np.min(beam_scores),
        np.max(beam_scores))
    beam_scores = np.clip(beam_scores, a_min=None, a_max=0.)

  return beam_predictions, beam_scores


def _process_beam_by_type(
    beam_predictions: List[TopkPrediction],
    beam_scores: Optional[np.ndarray] = None,
    beam_type: str = 'all_beam') -> Tuple[List[TopkPrediction], np.ndarray]:
  """Truncates beam predictions and scores according to desired beam type.

  If beam_type = 'all_beam', then returns original predictions and scores.
  If beam_type = 'top1_beam', then truncates the predictions and scores
    to the highest-score elements.
  If beam_type = 'non_top1_beam', then truncates the predictions and scores
    to the non-highest-score elements.

  Args:
    beam_predictions: An array of integer token ids for model predictions, shape
      (batch_size, beam_size, output_len). Null tokens will be marked with '0'.
    beam_scores: An array of float log probability for each tokens, shape
      (batch_size, beam_size, output_len).
    beam_type: Type of truncation to apply. Must be one of ('all_beam',
      'non_top1_beam', 'top1_beam').

  Returns:
    Truncated beam_predictions and beam_scores with shape
    (batch_size, truncated_beam_size, output_len).

  Raises:
    ValueError: If beam_type is not one of ('all_beam', 'non_top1_beam',
      'top1_beam').
  """
  # Checks beam_type.
  if beam_type not in _SEQUENCE_BEAM_TYPES:
    raise ValueError(f'`beam_type` must be one of {_SEQUENCE_BEAM_TYPES}. '
                     f'Got {beam_type}.')

  if beam_scores is None:
    beam_scores = np.zeros(shape=(1, 1), dtype=np.float32)

  if beam_type == 'top1_beam':
    beam_predictions = [prediction[-1:] for prediction in beam_predictions]
    beam_scores = beam_scores[:, -1:]

  if beam_type == 'non_top1_beam':
    beam_predictions = [prediction[:-1] for prediction in beam_predictions]
    beam_scores = beam_scores[:, :-1]

  return beam_predictions, beam_scores


def _compute_beam_correctness(targets: Sequence[str],
                              beam_predictions: List[TopkPrediction],
                              vocab: seqio.SentencePieceVocabulary,
                              beam_type: str) -> np.ndarray:
  """Computes correctness of top-K beam predictions.

  Given a collection of beam results [y_1, ..., y_K], the
  accuracy of the top-K beams is defined as whether y_true is in the list of
  beam predictions.

  Args:
    targets: A sequence of strings for the target strings. In the case of
      sequence prediction, the string takes the form '<token_1> <token_2>
      <token_3> ...'. Shape (batch_size,).
    beam_predictions: An array of integer token ids for model predictions, shape
      (batch_size, beam_size, output_len). Null tokens will be marked with '0'.
    vocab: The SentencePieceVocabulary used for model training.
    beam_type: Type of truncation to apply. Must be one of ('all_beam',
      'non_top1_beam', 'top1_beam').

  Returns:
    An int array indicating correctness of predictions (1 for correct prediction
    and 0 for incorrect prediction), shape (batch_size, ).
  """

  beam_predictions, _ = _process_beam_by_type(
      beam_predictions, beam_scores=None, beam_type=beam_type)

  correct_predictions = [
      vocab.encode(target) in predictions
      for target, predictions in zip(targets, beam_predictions)
  ]

  return np.array(correct_predictions, dtype=np.int32)


def _compute_beam_uncertainty(beam_predictions: List[TopkPrediction],
                              beam_scores: np.ndarray, beam_type: str,
                              uncertainty_type: str) -> np.ndarray:
  """Computes the predictive uncertainty of top-K beam predictions.

  Given a collection of beam results [y_1, ..., y_K], the uncertainty can be
  computed in several ways:

  (1) Cumulative probability, i.e., the sum of the probability of the
      individual beams:

      log p([y_1, .., y_K]) = LogSumExp [ log p(y_1), ... log p(y_K) ].

  (2) Predictive margin, i.e., the difference in the log probability between the
      top two beams:

      margin = p(y_1) - p(y_2)

  (3) Predictive entropy, i.e., the predictive entropy among the model's
      predictive distribution as characterized by the top-K beams. Here the
      top-K beam is interpreted as a sample from the model's predictive
      distribution using importance sampling (Eq. 14 of [1]):

      entropy = integrate_{y_k} log p( y_k ) * p( p_k ) dy_k.

  ## Reference

  [1]: Andrey Malinin, Mark Gales. Uncertainty Estimation in Autoregressive
       Structured Prediction. In _International Conference on Learning
       Representations_, 2020.
       https://arxiv.org/abs/2002.07650


  Args:
    beam_predictions: A list of integer token ids for model predictions, shape
      (batch_size, beam_size, output_len). Null tokens will be marked with '0'.
    beam_scores: An array of float log probability for each tokens, shape
      (batch_size, beam_size, output_len).
    beam_type: Type of truncation to apply. Must be one of ('all_beam',
      'non_top1_beam', 'top1_beam').
    uncertainty_type: Type of model uncertainty to compute. Must be one of
      ('probability', 'margin', 'entropy').

  Returns:
    beam_confidences. A float list of uncertainty scores for each prediction,
      shape (batch_size, ).

  Raises:
    ValueError: If `uncertainty_type` is not one of ('probability', 'margin',
      'entropy').
  """
  # Checks uncertainty type.
  if uncertainty_type not in _SEQUENCE_UNCERTAINTY_TYPES:
    raise ValueError(
        f'`uncertainty_type` must be one of {_SEQUENCE_UNCERTAINTY_TYPES}.'
        f' Got {uncertainty_type}.')

  # Prepares beam predictions and scores.
  beam_predictions, beam_scores = _process_beam_by_type(beam_predictions,
                                                        beam_scores, beam_type)

  # Prepares sequence-level (normalized) log probs for uncertainty computation
  # by setting null tokens / sequences to approperiate values, shape
  # (batch_size, beam_size).

  # Preprocess token-level beam_scores, setting null token scores to NEG_INF.
  null_tokens_mask = np.array(beam_predictions, dtype=np.int32) == 0
  beam_scores = np.where(null_tokens_mask, NEG_INF, beam_scores)

  # Computes sequence-level log_probs, setting null sequence score to NEG_INF.
  beam_log_probs = np.sum(beam_scores * (beam_scores > NEG_INF), axis=2)
  beam_seq_len = np.sum(beam_scores > NEG_INF, axis=2, dtype=np.float32)
  beam_log_probs_normalized = beam_log_probs / beam_seq_len

  process_null_beam = (
      lambda log_prob: log_prob * (beam_seq_len > 0) +  # pylint:disable=g-long-lambda
      NEG_INF * (beam_seq_len == 0).astype(np.float32))
  beam_log_probs = process_null_beam(beam_log_probs)
  beam_log_probs_normalized = process_null_beam(beam_log_probs_normalized)

  # Computes uncertainty, shape (batch_size, ).
  if uncertainty_type == 'probability':
    # Computes the log of mean predictive probability.
    beam_confidences = sp_special.logsumexp(beam_log_probs, axis=1)
  elif uncertainty_type == 'margin':
    # Computes the log margin between probability of top1 and top2 predictions.
    beam_probs = np.exp(beam_log_probs_normalized)
    beam_confidences = beam_probs[:, -1] - beam_probs[:, -2]
    if np.any(beam_confidences < 0.):
      neg_conf_id = np.argmin(beam_confidences)
      logging.info(
          'negative margin (%s perc), e.g., id=%d: beam_confidences: %s, '
          'beam_probs: %s, beam_log_probs: %s. beam_scores: %s. '
          'beam_predictions: %s.',
          np.mean(beam_confidences < 0.) * 100, neg_conf_id,
          beam_confidences[neg_conf_id], beam_probs[neg_conf_id],
          beam_log_probs_normalized[neg_conf_id], beam_scores[neg_conf_id],
          beam_predictions[neg_conf_id])
      beam_confidences = np.clip(beam_confidences, a_min=0., a_max=None)
    beam_confidences = np.log(beam_confidences)
  else:
    # Compute predictive entropy on sequence level (Eq. 14 of [1]).
    beam_weights = sp_special.softmax(beam_log_probs, axis=1)
    beam_log_probs_masked = beam_log_probs_normalized * (beam_seq_len > 0)
    beam_confidences = np.sum(beam_weights * beam_log_probs_masked, axis=1)

  # Exponentiate so uncertainty stays within range (0, 1).
  return np.exp(beam_confidences)

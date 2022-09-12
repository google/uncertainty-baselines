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

"""Utility functions for t5x training and inference."""

import json
import os
import queue
import re

from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Type, Union

import jax
import jax.numpy as jnp
import numpy as np
import seqio
from t5x import adafactor
from t5x import optimizers
from t5x import partitioning
from t5x import train_state as train_state_lib
from t5x import utils

import tensorflow as tf
from tensorflow.io import gfile

_DeviceArray = jnp.DeviceArray
_BeamPrediction = List[int]
_BeamScore = List[float]

_BeamPredType = Union[_BeamPrediction, List[_BeamPrediction]]
_BeamScoreType = Union[_BeamScore, Dict[str, _BeamScore]]
_BeamOutputTupleType = Tuple[_BeamPredType, _BeamScoreType]


# TODO(jereliu): Update type annotation for `inferences` after t5x.infer becomes
# stable.
# TODO(phandu): Support `output_ids` argument to match upstream implementation.
def write_beam_inferences_to_file(
    path: str,
    inferences: Tuple[Sequence[Any], Mapping[str, Any]],
    task_ds: tf.data.Dataset,
    mode: str,
    vocabulary: Optional[seqio.Vocabulary] = None,
    json_encoder_cls: Type[json.JSONEncoder] = seqio.TensorAndNumpyEncoder,
    include_all_inputs: bool = False,
    input_fields_to_include: Optional[Sequence[str]] = None,
    output_beam_scores: bool = True) -> None:
  """Writes beam predictions, along with pretokenized inputs, to JSONL file.

  This function is an improved version of t5x.infer.write_inferences_to_file()
  that can write Top-K beam decoding predictions to jsonl, and (optionally)
  outputs the negative log likelihood scores for each prediction. It is intended
  to be used with a t5x.models.EncoderDecoderModel whose the arguments for
  model.predict_batch_with_aux() are set to `num_decodes=K` and
  `return_all_decodes=True`. (See eval.gin files under deepbank task for
  examples.)

  The only difference between this function and the original function is in the
  `mode == 'predict'` section, where the prediction output is processed by the
  `process_beam_prediction_outputs` function instead.

  Args:
    path: File path to write to.
    inferences: A tuple containing (predictions, aux_values). If mode is
      'predict' then the `predictions` will be token IDs. If it's
      'scores' then it'll be a collection of scores. `aux_values` will be an
      empty dictionary unless mode is 'predict_with_aux', in which case it'll
      contain the model's auxiliary outputs.
    task_ds: Original task dataset. Features from task with suffix
      `_pretokenized` are added to the outputs.
    mode: Prediction mode, either 'predict', 'score' or 'predict_with_aux'.
    vocabulary: Task output vocabulary. Only used in `predict` mode in order to
      decode predicted outputs into string.
    json_encoder_cls: a JSON encoder class used to customize JSON serialization
      via json.dumps.
    include_all_inputs: if True, will include all model inputs in the output
      JSONL file (including raw tokens) in addition to the pretokenized inputs.
    input_fields_to_include: List of input fields to include in the output JSONL
      file. This list should be None if `include_all_inputs` is set to True.
    output_beam_scores: Whether to also write beam prediction scores when it is
      available.
  """
  all_predictions, all_aux_values = inferences

  if mode in ('predict', 'predict_with_aux') and vocabulary is None:
    raise ValueError('The `vocabulary` parameter is required in `predict` and '
                     '`predict_with_aux` modes')

  def _json_compat(value):
    if isinstance(value, bytes):
      return value.decode('utf-8')
    elif isinstance(value, (jnp.bfloat16, jnp.floating)):
      return float(value)
    elif isinstance(value, jnp.integer):
      return float(value)
    elif isinstance(value, (jnp.ndarray, np.ndarray)):
      # Flatten array features.
      return value.tolist()
    else:
      return value

  if include_all_inputs and input_fields_to_include is not None:
    raise ValueError(
        'include_all_inputs and input_fields_to_include should not be set'
        ' simultaneously.')
  with gfile.GFile(path, 'w') as f:
    for i, inp in task_ds.enumerate().as_numpy_iterator():
      predictions = all_predictions[i]
      aux_values = {aux_field: v[i] for aux_field, v in all_aux_values.items()}

      if include_all_inputs:
        inputs = inp
      elif input_fields_to_include is not None:
        inputs = {
            k: v for k, v in inp.items() if k in input_fields_to_include or
            (k.endswith('_pretokenized') and
             k[:-len('_pretokenized')] in input_fields_to_include)
        }
      else:
        inputs = {k: v for k, v in inp.items() if k.endswith('_pretokenized')}

      json_dict = {}
      json_dict['inputs'] = {k: _json_compat(v) for k, v in inputs.items()}

      if mode == 'predict':
        assert vocabulary is not None
        # This is a section where we deviate from the original function.
        predict_dict = process_beam_prediction_outputs(predictions, vocabulary,
                                                       output_beam_scores)

        # Converts to json compatible format and add to json_dict.
        predict_dict = {k: _json_compat(v) for k, v in predict_dict.items()}
        json_dict.update(predict_dict)
      elif mode == 'score':
        json_dict['score'] = _json_compat(predictions)
        # This is a section where we deviate from the original function.
        if aux_values:
          json_dict['intermediates'] = jax.tree_map(_json_compat, aux_values)
      elif mode == 'predict_batch_with_aux':
        assert vocabulary is not None
        # This is a section where we deviate from the original function.
        predict_dict = process_beam_prediction_outputs(predictions, vocabulary,
                                                       output_beam_scores)

        # Converts to json compatible format and add to json_dict.
        predict_dict = {k: _json_compat(v) for k, v in predict_dict.items()}
        json_dict.update(predict_dict)

        json_dict['aux'] = jax.tree_map(_json_compat, aux_values)
      else:
        raise ValueError(f'Invalid mode: {mode}')

      json_str = json.dumps(json_dict, cls=json_encoder_cls)
      f.write(json_str + '\n')


def process_beam_prediction_outputs(
    output: Union[_BeamOutputTupleType, _BeamPredType],
    vocabulary: Optional[seqio.Vocabulary] = None,
    output_beam_scores: bool = True) -> Dict[str, Any]:
  """Prepares the beam prediction output to be written into json dictionary.

  Args:
    output: Prediction output from model.predict_batch(). It can be either a
      list of `beam_prediction`, or a tuple of (`beam_prediction`,
      `beam_scores`). The `beam_prediction` can be a single list (i.e.,
      List[int]) for Top-1 prediction, or a nested list for Top-K prediction.
      The `beam_scores` can be a float list for the log-likelihood of all beam
      predictions (i.e., List[float]), a dictionary containing the
      scores, i.e., {'scores': List[float]}, or a jax DeviceArray (
      shape (beam_size, max_len)) that contains token-level log probabilities.
    vocabulary: Task output vocabulary. Only used in `predict` mode in order to
      decode predicted outputs into string.
    output_beam_scores: Whether to also write beam prediction scores when it is
      available.

  Returns:
    A dictionary whose fields contains Top-K predictions of the format
    {'prediction_0': List[int], 'prediction_1': List[int], ...}, and optionally
    the negative log-likelihood score for each beam prediction of the format
    {'beam_scores': List[float]}.

  Raises:
    ValueError: If `output` is not a list or 2-tuple.
    ValueError: If beam_scores is not a list or a dict with key `scores`.
    ValueError: If beam_predictions is not a nested list of integers.
    ValueError: If lengths of beam_predictions and beam_scores not equal.
  """
  # Check `output` data type. Must be either a list of beam predictions, or a
  # 2-tuple (beam_predictions, beam_scores).
  if not isinstance(output, (list, tuple)):
    raise ValueError(
        'Output of predict_batch() should be either a list or a tuple. '
        f'Got {type(output)}.')
  if isinstance(output, tuple) and len(output) != 2:
    raise ValueError(
        'Output tuple of predict_batch() should be a 2-tuple (predictions, '
        f'scores). Got {len(output)}.')

  # Extracts the beam predictions (and beam scores) from output.
  beam_predictions = output
  output_contains_scores = isinstance(output, tuple) and len(output) == 2

  if output_contains_scores:
    beam_predictions, beam_scores = beam_predictions

    # Validate data types for `beam_scores`. Also converts it to a list if
    # it is a dictionary.
    if not isinstance(beam_scores, (list, dict, _DeviceArray)):
      raise ValueError(
          'scores output from predict_batch() should be either list, dict, or a'
          f' jax device array. Got {type(output[1])}.')
    if isinstance(beam_scores, dict):
      if 'scores' not in beam_scores.keys():
        raise ValueError(
            'score dictionary from predict_batch() must contain key '
            f'`scores`. Got keys {beam_scores.keys()}.')
      beam_scores = beam_scores['scores']
    if isinstance(beam_scores, (_DeviceArray, jnp.ndarray)):
      if len(beam_scores.shape) != 2:
        raise ValueError(
            'score array from predict_batch() must be a rank-2 tensor with '
            f'shape (beam_size, max_len). Got {beam_scores.shape}.')
      beam_scores = beam_scores.tolist()

  # Validate data types for `beam_predictions`. Also converts it to a nested
  # list if it is a single list (List[int]) that only contains the top-1
  # prediction.
  if not isinstance(beam_predictions, list):
    raise ValueError('prediction output from predict_batch() must be a list. '
                     f'Got {type(beam_predictions)}.')
  if not isinstance(beam_predictions[0], list):
    beam_predictions = [beam_predictions]

  if not isinstance(beam_predictions[0][0], int):
    raise ValueError('prediction output from predict_batch() must be a list of'
                     f' integer token ids. Got {type(beam_predictions[0][0])}')

  # Starts processing outputs.
  beam_outputs_dict = dict()

  # First processes the beam predictions. Write predictions in reverse order
  # since we'd like to write higher quality prediction (i.e., the later sequence
  # in beam_predictions) first.
  for beam_id, beam_prediction in enumerate(reversed(beam_predictions)):
    output_decoded = vocabulary.decode_tf(beam_prediction).numpy()  # pytype: disable=attribute-error
    beam_outputs_dict[f'prediction_{beam_id}'] = output_decoded
    beam_outputs_dict[f'prediction_{beam_id}_ids'] = beam_prediction

  # Optionally, also processes the beam scores.
  if output_beam_scores and output_contains_scores:
    # Make sure length of beam_scores matches with beam_predictions.
    if len(beam_scores) != len(beam_predictions):
      raise ValueError(
          'Lengths of beam_predictions and beam_scores should equal. Got '
          f'len(beam_predictions)={len(beam_predictions)}, '
          f'len(beam_scores)={len(beam_scores)}.')

    beam_outputs_dict['beam_scores'] = beam_scores[::-1]

  return beam_outputs_dict


class AdafactorGP(adafactor.Adafactor):
  """A wrapper of t5x.adafactor.Adafactor to support mutable updates."""

  def apply_param_gradient(self, step, hyper_params, param, state, grad, path):
    if 'gp_head_state' in path:
      # For head_state parameters, we will use grad as the new value.
      return grad.astype(param.dtype), state

    return super().apply_param_gradient(step, hyper_params, param, state, grad,
                                        path)


def latest_checkpoint_paths(model_sweep_dir: str,
                            ckpt_pattern: str = r'checkpoint_\d+$',
                            ckpt_type: str = 'first'):
  r"""Returns a list of paths to the latest checkpoints.

  This function assumes the checkpoints are stored under paths of the format

  '/{model_sweep_dir}/{model_dir}/{ckpt_pattern}/'

  and finds the best checkpoint under each checkpoint path.

  When there are multiple checkpoints under the directory, it assumes the best
  checkpoint is the second to the largest checkpoint (which is usually a result
  of using t5x.checkpoints.SaveBestCheckpointer with keep=1).

  Args:
    model_sweep_dir: String of full path to the sweep directory.
    ckpt_pattern: String of regex pattern for the checkpoint directories.
      The default pattern assumes the directory name starts with `checkpoint_`
      and is then followed by arbitrary number of integers.
    ckpt_type: String indicate the type of checkpoints to extract, it can be
      'latest' (i.e., the checkpoint correspond to the largest checkpoint id)
      or 'first' (i.e., the checkpoint correspond to the first checkpoint id).
      The latter option is useful when using the
      `t5x.checkpoints.SaveBestCheckpointer`, which will save the best
      checkpoint (as its best checkpoint), the second-to-last checkpoint,
      and the latest checkpoint.

  Returns:
    A list of paths to the latest checkpoints.
  """
  ckpt_pattern_regex = re.compile(ckpt_pattern)

  ckpt_paths = []

  for model_dir in gfile.listdir(model_sweep_dir):
    model_path = os.path.join(model_sweep_dir, model_dir)

    # Skips if the path is not a directory.
    if not gfile.isdir(model_path):
      continue

    # Finds all checkpoints under `model_path` that matches `ckpt_pattern`.
    ckpt_names = gfile.listdir(model_path)
    ckpt_names = [path for path in ckpt_names if ckpt_pattern_regex.match(path)]

    # Finds the latest checkpoint by their number.
    ckpt_numbers = [
        re.findall(r'\d+$', ckpt_path)[0] for ckpt_path in ckpt_names
    ]
    ckpt_numbers = [int(number) if number else 0 for number in ckpt_numbers]

    if len(ckpt_numbers) == 1:
      latest_checkpoint_id = 0
    elif ckpt_type == 'first':
      # Return the first checkpoint id. Since the `BestCheckpointer` always keep
      # (at least) 3 checkpoints: the best checkpoint, the second-to-last
      # checkpoint, and the final checkpoint.
      latest_checkpoint_id = np.argsort(ckpt_numbers)[0]
    elif ckpt_type == 'latest':
      latest_checkpoint_id = np.argmax(ckpt_numbers)
    else:
      raise ValueError(
          f'ckpt_type can only be one of (`latest`, `best`). Got {ckpt_type}')

    # Stores the path to the latest checkpoint.
    ckpt_path = os.path.join(model_path, ckpt_names[latest_checkpoint_id])
    ckpt_paths.append(ckpt_path)

  return ckpt_paths


class TrainStateEnsembleInitializer(utils.TrainStateInitializer):
  """Helper for initializing ensembled TrainState from several checkpoints."""

  def __init__(self,
               optimizer_def: Optional[optimizers.OptimizerDefType],
               init_fn: utils.InitFnCallable,
               ensemble_size: int = 1,
               **kwargs):
    """TrainStateEnsembleInitializer constructor.

    Args:
      optimizer_def: Optimizer def to be initialized, or None to create a
        `InferenceState` without an optimizer.
      init_fn: callable that initializes model variables from a PRNGKey and the
        input shapes.
      ensemble_size: The ensemble size.
      **kwargs: keyword arguments to be passed to the upstream
        TrainStateInitializer.
    """
    super().__init__(optimizer_def, init_fn, **kwargs)
    params_axes_list = [self.train_state_axes.params] * ensemble_size
    self.train_state_axes = self.train_state_axes.replace_params(
        params_axes_list)  # type: ignore

  def from_checkpoints(
      self,
      restore_cfgs: Sequence[utils.RestoreCheckpointConfig],
      ds_iter: Optional[tf.data.Iterator] = None,
      init_rng: Optional[jnp.ndarray] = None,
  ) -> Iterable[train_state_lib.TrainState]:
    params_list = []
    last_train_state = None
    for train_state in super().from_checkpoints(
        restore_cfgs=restore_cfgs, ds_iter=ds_iter, init_rng=init_rng):
      params_list.append(train_state.params)
      last_train_state = train_state

    if len(params_list) != len(self.train_state_axes.params):  # type: ignore
      raise ValueError(
          'The number of checkpoints is different from the ensemble size.')

    if params_list:
      train_state = last_train_state.replace_params(params_list)  # type: ignore
      yield train_state



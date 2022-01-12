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

"""Eval utils."""
# pylint: disable=g-bare-generic
# pylint: disable=g-doc-args
# pylint: disable=g-doc-return-or-yield
# pylint: disable=g-importing-member
# pylint: disable=g-no-space-after-docstring-summary
# pylint: disable=g-short-docstring-punctuation
# pylint: disable=logging-format-interpolation
# pylint: disable=logging-fstring-interpolation
# pylint: disable=missing-function-docstring
import collections
import time
from typing import Any, Dict
from absl import logging

import numpy as np
import robustness_metrics as rm
from sklearn.metrics import accuracy_score
from sklearn.metrics import auc
from sklearn.metrics import log_loss
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils import check_array
from sklearn.utils import check_consistent_length
import tensorflow as tf
from .metric_utils import log_epoch_metrics  # local file import from baselines.diabetic_retinopathy_detection.utils.metric_utils
from .results_storage_utils import add_joint_dicts  # local file import from baselines.diabetic_retinopathy_detection.utils.results_storage_utils
from .results_storage_utils import load_dataframe_gfile  # local file import from baselines.diabetic_retinopathy_detection.utils.results_storage_utils


@tf.function
def eval_step_tf(dataset_iterator, dataset_steps, strategy, estimator,
                 estimator_args, uncertainty_estimator_fn, is_deterministic):
  """Eval step.

  Run TensorFlow model evaluation, using an `uncertainty_estimator_fn`
  to produce predictions and decomposed uncertainty estimates for each example.

  Args:
    dataset_iterator: tf.data.Dataset, dataset on which we will evaluate the
      model.
    dataset_steps: int, number of gradient steps in the dataset.
    strategy: tf.distribute strategy, used to distribute datasets.
    estimator: model wrapped to produce a `tf.Tensor`, predictive mean, with
      shape [B].
    estimator_args: Dict, extra args for the `uncertainty_estimator_fn`, such as
      the number of MC samples, `num_samples`.
    uncertainty_estimator_fn: Callable, method to produce predictive means along
      with various metrics of uncertainty, e.g., predictive_entropy, epistemic
      uncertainty (mutual information).
    is_deterministic: bool, is the model a single deterministic network. In this
      case, we cannot capture epistemic uncertainty.

  Returns:
    Dict, contains `tf.Tensor` predictions, ground truth,
      and uncertainty estimates.
  """
  print('Tracing in `eval_utils.eval_step_tf`.')

  def step_fn(inputs):
    print('Tracing in `eval_utils.eval_step_tf.step_fn`.')
    images = inputs['features']
    labels = inputs['labels']

    # Compute prediction, total, aleatoric, and epistemic uncertainty estimates
    pred_and_uncert = uncertainty_estimator_fn(
        images, estimator, training_setting=False, **estimator_args)

    # Return a tuple
    y_true = labels
    y_pred = pred_and_uncert['prediction']
    y_pred_entropy = pred_and_uncert['predictive_entropy']

    if not is_deterministic:
      y_pred_variance = pred_and_uncert['predictive_variance']
      y_aleatoric_uncert = pred_and_uncert['aleatoric_uncertainty']
      y_epistemic_uncert = pred_and_uncert['epistemic_uncertainty']
    else:
      y_pred_variance = tf.zeros(0)
      y_aleatoric_uncert = tf.zeros(0)
      y_epistemic_uncert = tf.zeros(0)

    return (y_true, y_pred, y_pred_entropy, y_pred_variance, y_aleatoric_uncert,
            y_epistemic_uncert)

  # Containers for storage of
  # predictions, ground truth, uncertainty estimates
  # Construct tf.TensorArrays to store model results
  n_per_core_batches = dataset_steps * strategy.num_replicas_in_sync
  y_true = tf.TensorArray(tf.int32, size=n_per_core_batches)
  y_pred = tf.TensorArray(tf.float32, size=n_per_core_batches)
  y_pred_entropy = tf.TensorArray(tf.float32, size=n_per_core_batches)
  y_pred_variance = tf.TensorArray(tf.float32, size=n_per_core_batches)
  y_aleatoric_uncert = tf.TensorArray(tf.float32, size=n_per_core_batches)
  y_epistemic_uncert = tf.TensorArray(tf.float32, size=n_per_core_batches)

  for i in tf.range(dataset_steps):
    result = strategy.run(step_fn, args=(next(dataset_iterator),))

    # Parse results tuple
    (y_true_, y_pred_, y_pred_entropy_, y_pred_variance_, y_aleatoric_uncert_,
     y_epistemic_uncert_) = result

    # Convert from Per-Replica object to tuple
    if strategy.num_replicas_in_sync > 1:
      y_true_ = y_true_.values
      y_pred_ = y_pred_.values
      y_pred_entropy_ = y_pred_entropy_.values

      if not is_deterministic:
        y_pred_variance_ = y_pred_variance_.values
        y_aleatoric_uncert_ = y_aleatoric_uncert_.values
        y_epistemic_uncert_ = y_epistemic_uncert_.values

      # Iterate through per-batch results
      # This is written in a very un-Pythonic manner to have updates only
      # rely on arguments successfully passed to TPU scope
      for replica_id in tf.range(strategy.num_replicas_in_sync):
        index = (strategy.num_replicas_in_sync * i) + replica_id
        for batch_result in y_true_:
          y_true = y_true.write(index, batch_result)
        for batch_result in y_pred_:
          y_pred = y_pred.write(index, batch_result)
        for batch_result in y_pred_entropy_:
          y_pred_entropy = y_pred_entropy.write(index, batch_result)

        if not is_deterministic:
          for batch_result in y_pred_variance_:
            y_pred_variance = y_pred_variance.write(index, batch_result)
          for batch_result in y_aleatoric_uncert_:
            y_aleatoric_uncert = y_aleatoric_uncert.write(index, batch_result)
          for batch_result in y_epistemic_uncert_:
            y_epistemic_uncert = y_epistemic_uncert.write(index, batch_result)
    else:
      y_true = y_true.write(i, y_true_)
      y_pred = y_pred.write(i, y_pred_)
      y_pred_entropy = y_pred_entropy.write(i, y_pred_entropy_)

      if not is_deterministic:
        y_pred_variance = y_pred_variance.write(i, y_pred_variance_)
        y_aleatoric_uncert = y_aleatoric_uncert.write(i, y_aleatoric_uncert_)
        y_epistemic_uncert = y_epistemic_uncert.write(i, y_epistemic_uncert_)

  results_arrs = {
      'y_true': y_true.stack(),
      'y_pred': y_pred.stack(),
      'y_pred_entropy': y_pred_entropy.stack(),
  }
  if not is_deterministic:
    results_arrs['y_pred_variance'] = y_pred_variance.stack()
    results_arrs['y_aleatoric_uncert'] = y_aleatoric_uncert.stack()
    results_arrs['y_epistemic_uncert'] = y_epistemic_uncert.stack()

  return results_arrs


def evaluate_model_on_datasets(strategy,
                               datasets,
                               steps,
                               estimator,
                               estimator_args,
                               uncertainty_estimator_fn,
                               eval_batch_size,
                               call_dataset_iter,
                               is_deterministic=False,
                               backend='tf',
                               eval_step_jax=None,
                               verbose=False):
  """Eval on dataset.

  Run model evaluation on all provided datasets, using an
  `uncertainty_estimator_fn` to produce predictions and decomposed
  uncertainty estimates for each example.

  Additionally constructs joint dataset predictions, composed of predictions
    on both in-domain and OOD datasets.

  Args:
    strategy: tf.distribute strategy, used to distribute datasets.
    datasets: Dict[str, tf.data.Dataset], datasets on which we evaluate the
      model.
    steps: Dict[str, int], number of gradient steps in each dataset.
    estimator: model wrapped to produce a `tf.Tensor` (if `backend`=='tf') or
      `np.ndarray` (if `backend`=='jax'), predictive mean, with shape [B].
    estimator_args: Dict, extra args for the `uncertainty_estimator_fn`, such as
      the number of MC samples, `num_samples`.
    uncertainty_estimator_fn: Callable, method to produce predictive means along
      with various metrics of uncertainty, e.g., predictive_entropy, epistemic
      uncertainty (mutual information).
    eval_batch_size: int, size of evaluation minibatches.
    call_dataset_iter: bool, if True, should call `iter()` on each dataset. May
      not need if evaluation datasets have been repeated.
    is_deterministic: bool, is the model a single deterministic network. In this
      case, we cannot capture epistemic uncertainty.
    backend: str, in {'tf', 'jax'}, specifies the evaluation method.
    eval_step_jax: Callable, evaluation method used for Jax model.
    verbose: bool, extra logging.

  Returns:
    Dict, for each dataset, contains `np.array` predictions, ground truth,
      and uncertainty estimates.
  """
  # Need to collect these so we can form joint datasets:
  # e.g., joint_test = in_domain_test UNION ood_test
  dataset_split_to_containers = {}

  for dataset_split, dataset in datasets.items():
    # Begin iteration for this dataset split
    start_time = time.time()

    if call_dataset_iter:
      dataset_iterator = iter(dataset)
    else:
      dataset_iterator = dataset

    logging.info(f'Creating iterator took {time.time() - start_time} seconds.')

    dataset_steps = steps[dataset_split]
    logging.info(f'Evaluating split {dataset_split}.')
    if backend == 'jax':
      eval_epoch_arrs = eval_step_jax(dataset_iterator, dataset_steps,
                                      is_deterministic, **estimator_args)
    elif backend == 'tf':
      eval_epoch_arrs = eval_step_tf(dataset_iterator,
                                     tf.convert_to_tensor(dataset_steps),
                                     strategy, estimator, estimator_args,
                                     uncertainty_estimator_fn, is_deterministic)
    else:
      raise NotImplementedError(f'Backend {backend} is not supported yet.')

    # Update metadata
    time_elapsed = time.time() - start_time
    dataset_split_to_containers[dataset_split] = {}
    dataset_split_dict = dataset_split_to_containers[dataset_split]
    dataset_split_dict['total_ms_elapsed'] = time_elapsed * 1e6
    dataset_split_dict['dataset_size'] = dataset_steps * eval_batch_size

    # Use vectorized NumPy containers
    for eval_key, eval_arr in eval_epoch_arrs.items():
      tmp_eval_arr = eval_arr if backend == 'jax' else eval_arr.numpy()

      if tmp_eval_arr.ndim > 1:
        tmp_eval_arr = np.concatenate(tmp_eval_arr).flatten()

      dataset_split_dict[eval_key] = tmp_eval_arr
      if verbose:
        print(f'Concatenated {eval_key} into shape '
              f'{dataset_split_dict[eval_key].shape}')

    dataset_split_dict['y_pred'] = dataset_split_dict['y_pred'].astype(
        'float64')

  # Add Joint Dicts
  dataset_split_to_containers = add_joint_dicts(
      dataset_split_to_containers, is_deterministic=is_deterministic)

  return dataset_split_to_containers


def evaluate_model_and_compute_metrics(
    strategy,
    eval_datasets,
    steps,
    metrics,
    eval_estimator,
    uncertainty_estimator_fn,
    eval_batch_size,
    available_splits,
    estimator_args,
    call_dataset_iter,
    is_deterministic=False,
    num_bins=15,
    use_tpu=True,
    return_per_pred_results=False,
    backend='tf',
    eval_step_jax=None):
  """Main.

  Main method for evaluation and computing metrics using TF or Jax
  models. Usable for evaluation during tuning.

  Args:
    strategy: tf.distribute strategy, used to distribute datasets.
    eval_datasets: Dict[str, tf.data.Dataset], datasets on which we evaluate the
      model.
    steps: Dict[str, int], number of gradient steps in each dataset.
    metrics: metrics.
    eval_estimator: model wrapped to produce a `tf.Tensor` (if `backend`=='tf')
      or `np.ndarray` (if `backend`=='jax'), predictive mean, with shape [B].
    uncertainty_estimator_fn: Callable, method to produce predictive means along
      with various metrics of uncertainty, e.g., predictive_entropy, epistemic
      uncertainty (mutual information).
    eval_batch_size: int, size of evaluation minibatches.
    available_splits: List[str], names of the evaluation datasets provided, used
      to log results only for these splits.
    estimator_args: Dict, extra args for the `uncertainty_estimator_fn`, such as
      the number of MC samples, `num_samples`.
    call_dataset_iter: bool, if True, should call `iter()` on each dataset. May
      not need if evaluation datasets have been repeated.
    is_deterministic: bool, is the model a single deterministic network. In this
      case, we cannot capture epistemic uncertainty.
    num_bins: int, number of bins to use with expected calibration error.
    use_tpu: bool, currently exists a bug that disallows collecting ECE during
      training with TPU, this is used to avoid logging that metric.
    return_per_pred_results: bool,
    backend: str, in {'tf', 'jax'}, specifies the evaluation method.
    eval_step_jax: Callable, evaluation method used for Jax model.

  Returns:
    Union[Tuple[Dict, Dict], Dict]
      If return_per_pred_results, return two Dicts. Else, return only the
      second.
      first Dict:
        for each dataset, per-prediction results (e.g., each prediction,
        ground-truth, loss, retention arrays).
      second Dict:
        for each dataset, contains `np.array` predictions, ground truth,
        and uncertainty estimates.
  """
  # Compute predictions on all evaluation datasets
  # When we eval during training we don't need to re-iterate the
  # evaluation datasets
  eval_results = evaluate_model_on_datasets(
      strategy=strategy,
      datasets=eval_datasets,
      steps=steps,
      estimator=eval_estimator,
      estimator_args=estimator_args,
      uncertainty_estimator_fn=uncertainty_estimator_fn,
      eval_batch_size=eval_batch_size,
      call_dataset_iter=call_dataset_iter,
      is_deterministic=is_deterministic,
      backend=backend,
      eval_step_jax=eval_step_jax)

  # For each eval dataset, add NLL and accuracy for each example
  eval_results = compute_loss_and_accuracy_arrs_for_all_datasets(eval_results)

  # Compute all metrics for each dataset --
  # Robustness, Open Set Recognition, Retention AUC
  metrics_results = compute_metrics_for_all_datasets(
      eval_results,
      use_precomputed_arrs=False,
      ece_num_bins=num_bins,
      compute_retention_auc=True,
      verbose=False)

  # Log metrics
  metrics_results = log_epoch_metrics(
      metrics=metrics,
      eval_results=metrics_results,
      use_tpu=use_tpu,
      dataset_splits=available_splits)

  if return_per_pred_results:
    return eval_results, metrics_results
  else:
    return metrics_results


def evaluate_model_on_datasets_np(
    datasets,
    steps,
    estimator,
    estimator_args,
    uncertainty_estimator_fn,
    eval_batch_size,
    is_deterministic,
    np_input=False,
):
  """Main method for evaluation and computing metrics.

  Appropriate for evaluation loops with a single GPU (i.e., no
  distribution strategy), and is framework-agnostic (will work just as
  well with a TF, Jax, or PyTorch model, given an `uncertainty_estimator_fn`
  to cast to NumPy ndarrays.

  Args:
    datasets: Dict[str, tf.data.Dataset], datasets on which we evaluate the
      model.
    steps: Dict[str, int], number of gradient steps in each dataset.
    estimator: model wrapped to produce a `np.ndarray`, predictive mean, with
      shape [B].
    estimator_args: Dict, extra args for the `uncertainty_estimator_fn`, such as
      the number of MC samples, `num_samples`.
    uncertainty_estimator_fn: Callable, method to produce predictive means along
      with various metrics of uncertainty, e.g., predictive_entropy, epistemic
      uncertainty (mutual information).
    eval_batch_size: int, size of evaluation minibatches.
    is_deterministic: bool, is the model a single deterministic network. In this
      case, we cannot capture epistemic uncertainty.
    np_input: bool, True if the model expects a NumPy input; we add a cast.

  Returns:
    Dict, for each dataset, contains `np.array` predictions, ground truth,
      and uncertainty estimates.
  """
  # Need to collect these so we can form joint datasets:
  # e.g., joint_test = in_domain_test UNION ood_test
  dataset_split_to_containers = {}

  for dataset_split, dataset in datasets.items():
    # Containers for numpy storage of
    # image names, predictions, ground truth, uncertainty estimates
    names = list()
    y_true = list()
    y_pred = list()
    y_pred_entropy = list()

    if not is_deterministic:
      y_pred_variance = list()
      y_aleatoric_uncert = list()
      y_epistemic_uncert = list()

    # Begin iteration for this dataset split
    start_time = time.time()
    dataset_iterator = iter(dataset)
    dataset_steps = steps[dataset_split]
    logging.info(f'Evaluating split {dataset_split}.')
    for step in range(dataset_steps):
      if step % 10 == 0:
        logging.info('Evaluated %d/%d batches.', step, dataset_steps)

      inputs = next(dataset_iterator)  # pytype: disable=attribute-error
      images = inputs['features']
      labels = inputs['labels']

      # Compute prediction, total, aleatoric, and epistemic
      # uncertainty estimates
      pred_and_uncert = uncertainty_estimator_fn(
          images._numpy() if np_input else images,  # pylint: disable=protected-access
          estimator,
          training_setting=False,
          **estimator_args)

      # Add this batch of predictions to the containers
      names.append(inputs['name'])
      y_true.append(labels.numpy())
      y_pred.append(pred_and_uncert['prediction'])
      y_pred_entropy.append(pred_and_uncert['predictive_entropy'])
      if not is_deterministic:
        y_pred_variance.append(pred_and_uncert['predictive_variance'])
        y_aleatoric_uncert.append(pred_and_uncert['aleatoric_uncertainty'])
        y_epistemic_uncert.append(pred_and_uncert['epistemic_uncertainty'])

    # Update metadata
    time_elapsed = time.time() - start_time
    dataset_split_to_containers[dataset_split] = {}
    dataset_split_dict = dataset_split_to_containers[dataset_split]
    dataset_split_dict['total_ms_elapsed'] = time_elapsed * 1e6
    dataset_split_dict['dataset_size'] = dataset_steps * eval_batch_size

    dataset_split_dict['names'] = np.concatenate(names).flatten()
    dataset_split_dict['y_true'] = np.concatenate(y_true).flatten()
    dataset_split_dict['y_pred'] = np.concatenate(y_pred).flatten()
    dataset_split_dict['y_pred'] = dataset_split_dict['y_pred'].astype(
        'float64')

    # Use vectorized NumPy containers
    dataset_split_dict['y_pred_entropy'] = (
        np.concatenate(y_pred_entropy).flatten())

    if not is_deterministic:
      dataset_split_dict['y_pred_variance'] = (
          np.concatenate(y_pred_variance).flatten())
      dataset_split_dict['y_aleatoric_uncert'] = (
          np.concatenate(y_aleatoric_uncert).flatten())
      dataset_split_dict['y_epistemic_uncert'] = (
          np.concatenate(y_epistemic_uncert).flatten())

  # Add Joint Dicts
  dataset_split_to_containers = add_joint_dicts(
      dataset_split_to_containers, is_deterministic=is_deterministic)

  return dataset_split_to_containers


def eval_model_numpy(datasets,
                     steps,
                     estimator,
                     estimator_args,
                     uncertainty_estimator_fn,
                     eval_batch_size,
                     is_deterministic,
                     distribution_shift,
                     num_bins=15,
                     np_input=False):
  """Main method for evaluation and computing metrics.

  Appropriate for evaluation loops with a single GPU (i.e., no
  distribution strategy), and is framework-agnostic (will work just as
  well with a TF, Jax, or PyTorch model, given an `uncertainty_estimator_fn`
  to cast to NumPy ndarrays.

  Args:
    datasets: Dict[str, tf.data.Dataset], datasets on which we evaluate the
      model.
    steps: Dict[str, int], number of gradient steps in each dataset.
    estimator: model wrapped to produce a `np.ndarray`, predictive mean, with
      shape [B].
    estimator_args: Dict, extra args for the `uncertainty_estimator_fn`, such as
      the number of MC samples, `num_samples`.
    uncertainty_estimator_fn: Callable, method to produce predictive means along
      with various metrics of uncertainty, e.g., predictive_entropy, epistemic
      uncertainty (mutual information).
    eval_batch_size: int, size of evaluation minibatches.
    is_deterministic: bool, is the model a single deterministic network. In this
      case, we cannot capture epistemic uncertainty.
    distribution_shift: which distribution shift to run on.
    num_bins: the number of bins to use for ECE.
    np_input: bool, True if the model expects a NumPy input; we add a cast.

  Returns:
    Dict, for each dataset, contains `np.array` predictions, ground truth,
      and uncertainty estimates.
  """
  eval_results = evaluate_model_on_datasets_np(
      datasets,
      steps,
      estimator,
      estimator_args,
      uncertainty_estimator_fn,
      eval_batch_size,
      is_deterministic=is_deterministic,
      np_input=np_input)

  if distribution_shift == 'aptos':
    # TODO(nband): generalize
    aptos_metadata_path = 'gs://ub-data/aptos/metadata.csv'
    eval_results['ood_test_balanced'] = compute_rebalanced_aptos_dataset(
        aptos_dataset=eval_results['ood_test'],
        aptos_metadata_path=aptos_metadata_path,
        new_aptos_size=10000)

  # For each eval dataset, add NLL and accuracy for each example
  eval_results = compute_loss_and_accuracy_arrs_for_all_datasets(eval_results)

  # Precompute ROC/PR curves, retention and balanced retention curves
  logging.info('Precomputing ROC/PR curves, retention and balanced'
               ' retention curves.')
  eval_results = precompute_arrs_for_all_datasets(eval_results=eval_results)

  logging.info('Computing metrics with precomputed arrs.')
  metrics_results = compute_metrics_for_all_datasets(
      eval_results,
      use_precomputed_arrs=True,
      ece_num_bins=num_bins,
      compute_retention_auc=True,
      verbose=False)

  # Log metrics
  available_splits = [split for split in eval_results.keys()]
  metrics_results = log_epoch_metrics(
      metrics=None,
      eval_results=metrics_results,
      use_tpu=False,
      dataset_splits=available_splits)

  return eval_results, metrics_results


def compute_metrics_for_all_datasets(eval_results,
                                     use_precomputed_arrs,
                                     ece_num_bins=15,
                                     compute_retention_auc=False,
                                     verbose=False):
  """Computes scalar metrics for all datasets.

  Args:
    eval_results: Dict[str, Dict], evaluation results for each dataset.
    use_precomputed_arrs: selects which eval function to use.
    ece_num_bins: int, used to compute expected calibration error metric.
    compute_retention_auc: bool, should compute retention metrics.
    verbose: bool, extra logging.

  Returns:
    Dict[str, Dict], scalar metric results for each dataset.
  """
  dataset_eval_fn = (
      compute_dataset_eval_metrics_with_precomputed_arrs
      if use_precomputed_arrs else compute_dataset_eval_metrics)
  metric_results = {}
  for dataset_key, results_dict in eval_results.items():
    if verbose:
      logging.info(f'Computing metrics for dataset split {dataset_key}.')

    compute_open_set_recognition = 'joint' in dataset_key
    metric_results[dataset_key] = dataset_eval_fn(
        dataset_key,
        results_dict,
        ece_num_bins=ece_num_bins,
        compute_open_set_recognition=compute_open_set_recognition,
        compute_retention_auc=compute_retention_auc)

  return metric_results


def precompute_arrs_for_all_datasets(eval_results, verbose=False):
  """Precompute metric arrays for all datasets, e.g., log loss, retention

  arrays, etc.

  Args:
    eval_results: Dict[str, Dict], evaluation results for each dataset.
    verbose: bool, extra logging.

  Returns:
    Dict[str, Dict], metric arrays for each dataset.
  """
  for dataset_key, results_dict in eval_results.items():
    if verbose:
      logging.info(f'Computing metrics for dataset split {dataset_key}.')

    compute_open_set_recognition = 'joint' in dataset_key
    eval_results[dataset_key] = precompute_metric_arrs(
        results=results_dict,
        compute_open_set_recognition=compute_open_set_recognition)

  return eval_results


def compute_loss_and_accuracy_arrs_for_all_datasets(eval_results):
  """Compute loss and accuracy arrays for each dataset.

  Args:
    eval_results: Dict[str, Dict], evaluation results for each dataset.

  Returns:
    Dict[str, Dict], loss and accuracy arrays for each dataset.
  """
  for dataset_key, results_dict in eval_results.items():
    eval_results[dataset_key] = compute_loss_and_accuracy_arrs(results_dict)

  return eval_results


def compute_loss_and_accuracy_arrs(results):
  """Compute loss and accuracy arrays for a particular dataset results dict.

  Args:
    results: Dict, evaluation results for a single dataset.

  Returns:
    Dict, loss and accuracy arrays for a single dataset.
  """
  results = compute_log_loss_arr(results)
  y_pred, y_true = results['y_pred'], results['y_true']
  results['accuracy_arr'] = y_true == (y_pred > 0.5)
  return results


def compute_log_loss_arr(results, labels=np.asarray([0, 1]), eps=1e-15):
  """Based on sklearn.preprocessing.log_loss, no aggregation.

  Args:
    results: Dict, evaluation results for a single dataset.
    labels: np.ndarray, binary classification task labels.
    eps: float, for numerical stability.

  Returns:
    Dict containing unaggregated log loss `np.ndarray`.
  """
  y_pred, y_true = results['y_pred'], results['y_true']
  y_pred = check_array(y_pred, ensure_2d=False)
  check_consistent_length(y_pred, y_true, None)

  lb = LabelBinarizer()

  if labels is not None:
    lb.fit(labels)
  else:
    lb.fit(y_true)

  if len(lb.classes_) == 1:
    if labels is None:
      raise ValueError('y_true contains only one label ({0}). Please '
                       'provide the true labels explicitly through the '
                       'labels argument.'.format(lb.classes_[0]))
    else:
      raise ValueError('The labels array needs to contain at least two '
                       'labels for log_loss, '
                       'got {0}.'.format(lb.classes_))

  transformed_labels = lb.transform(y_true)

  if transformed_labels.shape[1] == 1:
    transformed_labels = np.append(
        1 - transformed_labels, transformed_labels, axis=1)

  # Clipping
  y_pred = np.clip(y_pred, eps, 1 - eps)

  # If y_pred is of single dimension, assume y_true to be binary
  # and then check.
  if y_pred.ndim == 1:
    y_pred = y_pred[:, np.newaxis]
  if y_pred.shape[1] == 1:
    y_pred = np.append(1 - y_pred, y_pred, axis=1)

  # Check if dimensions are consistent.
  transformed_labels = check_array(transformed_labels)
  if len(lb.classes_) != y_pred.shape[1]:
    if labels is None:
      raise ValueError('y_true and y_pred contain different number of '
                       'classes {0}, {1}. Please provide the true '
                       'labels explicitly through the labels argument. '
                       'Classes found in '
                       'y_true: {2}'.format(transformed_labels.shape[1],
                                            y_pred.shape[1], lb.classes_))
    else:
      raise ValueError('The number of classes in labels is different '
                       'from that in y_pred. Classes found in '
                       'labels: {0}'.format(lb.classes_))

  # Renormalize
  y_pred /= y_pred.sum(axis=1)[:, np.newaxis]
  loss = -(transformed_labels * np.log(y_pred)).sum(axis=1)
  results['nll_arr'] = loss
  return results


def compute_rebalanced_aptos_dataset(aptos_dataset,
                                     aptos_metadata_path,
                                     new_aptos_size=10000):
  """The APTOS dataset has a significantly different underlying distribution

  of clinical severity (e.g., there are far more severe examples) versus
  the EyePACS dataset.

  This might tend to inflate the performance of a model (by providing more
  easily classified "severe" examples) without rebalancing of the dataset
  and metrics.

  Here we match the test empirical distribution of EyePACS by sampling
  from each underlying category with replacement, bringing the total size
  of the dataset to `new_aptos_size`.

  Args:
    aptos_dataset: Dict, predictions and ground truths on the APTOS dataset.
    aptos_metadata_path: str, location of APTOS metadata (names and clinical
      labels of each example).
    new_aptos_size: int, target size of the new rebalanced APTOS dataset.

  Returns:
    Dict, rebalanced APTOS dataset predictions, ground truths, etc.
  """
  # EyePACS Test Data: clinical severity label to proportion of dataset
  label_to_proportion = {
      0: 0.7359503164,
      1: 0.07129130537,
      2: 0.1472228732,
      3: 0.0228966487,
      4: 0.02263885634
  }

  # Load in APTOS metadata
  aptos_metadata_df = load_dataframe_gfile(
      file_path=aptos_metadata_path, sep=',')
  name_to_diagnosis = dict(
      zip(aptos_metadata_df['id_code'], aptos_metadata_df['diagnosis']))

  # Determine location of indices corresponding to each diagnosis
  diagnosis_to_indices = collections.defaultdict(list)
  names = aptos_dataset['names']
  for i, name in enumerate(names):
    try:
      name = name.decode('utf-8')
    except UnicodeDecodeError:
      name = str(name)
    diagnosis = int(name_to_diagnosis[name])
    diagnosis_to_indices[diagnosis].append(i)

  # Uniformly sample without replacement to form a new dataset,
  # with approximately same proportions as EyePACS Test Data
  # and total size ~10000
  new_indices = []
  for diagnosis, indices in diagnosis_to_indices.items():
    total_count_in_new_dataset = int(label_to_proportion[diagnosis] *
                                     new_aptos_size)
    new_indices.append(
        np.random.choice(
            indices, size=total_count_in_new_dataset, replace=True))

  new_indices = np.concatenate(new_indices)
  new_aptos_dataset = {}
  for key, value in aptos_dataset.items():
    try:
      new_aptos_dataset[key] = value[new_indices]
    except IndexError:
      new_aptos_dataset[key] = value

  return new_aptos_dataset


def compute_rebalanced_retention_curves(results: Dict[str, Any]):
  """Compute rebalanced retention curves.

  Compute rebalanced retention curves, which are used for joint (ID + OOD)
  tuning. This is done by repeating the OOD indices many times, and then
  bringing the number of OOD indices to the number of ID indices by sampling
  from the OOD indices without replacement.

  Args:
    results: Dict, results for a particular dataset (must be joint to have the
      `is_ood` key, for a binary `np.ndarray`.

  Returns:
    Dict, contains rebalanced retention curves.
  """
  y_pred_entropy, is_ood = results['y_pred_entropy'], results['is_ood']

  # Convert the boolean list to numpy array
  is_ood = np.array(is_ood)

  in_domain_indices = np.where(is_ood == 0)[0]

  n_in_domain = in_domain_indices.shape[0]
  ood_indices = np.where(is_ood == 1)[0]
  n_ood = ood_indices.shape[0]

  # We first tile the OOD indices this many times
  n_ood_repeats = int(n_in_domain / n_ood)

  # To bring the number of OOD indices = the number of ID indices, we sample
  # the necessary indices without replacement
  remaining_n_ood_indices = n_in_domain - (n_ood_repeats * n_ood)

  remaining_ood_indices = np.random.choice(
      ood_indices, size=remaining_n_ood_indices, replace=False)

  # Construct a list of all of the indices to retrieve
  all_indices = (
      in_domain_indices.tolist() + (n_ood_repeats * ood_indices.tolist()) +
      remaining_ood_indices.tolist())

  # Construct predictive entropy, accuracy, NLL arrays with these indices
  rebalanced_y_pred_entropy = y_pred_entropy[all_indices]
  rebalanced_accuracy = results['accuracy_arr'][all_indices]
  rebalanced_nll = results['nll_arr'][all_indices]

  rebalanced_accuracy_retention_curve = compute_retention_curve_on_accuracies(
      accuracies=rebalanced_accuracy, uncertainty=rebalanced_y_pred_entropy)
  rebalanced_nll_retention_curve = compute_retention_curve_on_losses(
      losses=rebalanced_nll, uncertainty=rebalanced_y_pred_entropy)

  y_pred = results['y_pred']
  y_true = results['y_true']
  rebalanced_auroc_retention_curve = compute_auc_retention_curve(
      y_pred=y_pred, y_true=y_true, uncertainty=y_pred_entropy, auc_str='roc')
  rebalanced_auprc_retention_curve = compute_auc_retention_curve(
      y_pred=y_pred, y_true=y_true, uncertainty=y_pred_entropy, auc_str='prc')

  return {
      'accuracy': rebalanced_accuracy_retention_curve,
      'nll': rebalanced_nll_retention_curve,
      'auroc': rebalanced_auroc_retention_curve,
      'auprc': rebalanced_auprc_retention_curve
  }


def compute_rebalanced_retention_scores(results: Dict[str, Any]):
  """Computes rebalanced retention curves, then the mean to get an AUC.

  Args:
    results: Dict, results for a particular dataset (must be joint to have the
      `is_ood` key, for a binary `np.ndarray`.

  Returns:
    Dict, contains the AUCs of the retention curves built on various
      base metrics.
  """
  rebalanced_curves = compute_rebalanced_retention_curves(results)
  return {
      'accuracy': np.mean(rebalanced_curves['accuracy']),
      'nll': np.mean(rebalanced_curves['nll']),
      'auroc': np.mean(rebalanced_curves['auroc']),
      'auprc': np.mean(rebalanced_curves['auprc'])
  }


def precompute_metric_arrs(results, compute_open_set_recognition=False):
  """Compute retention arrays and ROC/PR curves.

  Used for caching to do downstream plots and scalar metrics quickly.

  Args:
    results: dict, results for a particular dataset split.
    compute_open_set_recognition: bool, if True, compute OOD detection PR and
      AUROC metrics.

  Returns:
      Dict, contains retention arrays and ROC/PR curves.
  """
  y_true = results['y_true']
  y_pred = results['y_pred']
  y_pred_entropy = results['y_pred_entropy']

  # Compute ROC curve
  # try:
  results['fpr_arr'], results['tpr_arr'], _ = roc_curve(
      y_true=y_true, y_score=y_pred)
  # except:
  #   pass

  # Compute PR curve
  # try:
  results['precision_arr'], results['recall_arr'], _ = precision_recall_curve(
      y_true=y_true, probas_pred=y_pred)
  # except:
  #   pass

  if compute_open_set_recognition:
    is_ood = results['is_ood']
    results['ood_detection_fpr_arr'], results['ood_detection_tpr_arr'], _ = (
        roc_curve(y_true=is_ood, y_score=y_pred_entropy))
    (results['ood_detection_precision_arr'],
     results['ood_detection_recall_arr'], _) = (
         precision_recall_curve(y_true=is_ood, probas_pred=y_pred_entropy))

    # For the joint datasets, we also compute a rebalanced retention metric,
    # in which we duplicate the OOD dataset to match the size of the in-domain
    # dataset, and then compute the retention metrics.
    ret_curves = compute_rebalanced_retention_curves(results)
    results['balanced_retention_accuracy_arr'] = ret_curves['accuracy']
    results['balanced_retention_nll_arr'] = ret_curves['nll']
    results['balanced_retention_auroc_arr'] = ret_curves['auroc']
    results['balanced_retention_auprc_arr'] = ret_curves['auprc']

  # Retention curves
  assert 'accuracy_arr' in results
  assert 'nll_arr' in results
  results['retention_accuracy_arr'] = compute_retention_curve_on_accuracies(
      accuracies=results['accuracy_arr'], uncertainty=y_pred_entropy)
  results['retention_nll_arr'] = compute_retention_curve_on_losses(
      losses=results['nll_arr'], uncertainty=y_pred_entropy)
  results['retention_auroc_arr'] = compute_auc_retention_curve(
      y_pred=y_pred, y_true=y_true, uncertainty=y_pred_entropy, auc_str='roc')
  results['retention_auprc_arr'] = compute_auc_retention_curve(
      y_pred=y_pred, y_true=y_true, uncertainty=y_pred_entropy, auc_str='prc')

  return results


def compute_dataset_eval_metrics_with_precomputed_arrs(
    dataset_key,
    results,
    ece_num_bins=15,
    compute_open_set_recognition=False,
    compute_retention_auc=False):
  """Compute scalar metrics using cached retention and ROC/PR curves for

  efficiency.

  Args:
    dataset_key: str, name of dataset (prepends each metric in returned Dict).
    results: Dict, results for a particular dataset split including precomputed
      arrays.
    ece_num_bins: int, number of bins used in computing expected calibration
      error.
    compute_open_set_recognition: bool, if True, compute OOD detection PR and
      AUROC metrics.
    compute_retention_auc: bool, if True, compute retention AUC metrics by
      taking the mean of the retention arrays.

  Returns:
      Dict, scalar metrics.
  """
  y_pred = results['y_pred']
  y_true = results['y_true']

  eval_metrics = dict()

  # Standard predictive metrics
  eval_metrics[f'{dataset_key}/negative_log_likelihood'] = log_loss(
      y_pred=y_pred, y_true=y_true, labels=np.asarray([0, 1]))
  if 'fpr_arr' in results.keys() and 'tpr_arr' in results.keys():
    eval_metrics[f'{dataset_key}/auroc'] = auc(results['fpr_arr'],
                                               results['tpr_arr'])
  else:
    eval_metrics[f'{dataset_key}/auroc'] = None

  if 'precision_arr' in results.keys() and 'recall_arr' in results.keys():
    recall = results['recall_arr']
    precision = results['precision_arr']
    eval_metrics[f'{dataset_key}/auprc'] = auc(recall, precision)

  eval_metrics[f'{dataset_key}/accuracy'] = (
      accuracy_score(y_true=y_true, y_pred=(y_pred > 0.5)))

  # Uncertainty metrics
  ece = rm.metrics.ExpectedCalibrationError(num_bins=ece_num_bins)
  ece.add_batch(y_pred, label=y_true)
  eval_metrics[f'{dataset_key}/ece'] = ece.result()['ece']

  if compute_open_set_recognition:
    eval_metrics[f'{dataset_key}/ood_detection_auroc'] = auc(
        results['ood_detection_fpr_arr'], results['ood_detection_tpr_arr'])
    eval_metrics[f'{dataset_key}/ood_detection_auprc'] = auc(
        results['ood_detection_recall_arr'],
        results['ood_detection_precision_arr'])

    eval_metrics[f'{dataset_key}/balanced_retention_accuracy_auc'] = np.mean(
        results['balanced_retention_accuracy_arr'])
    eval_metrics[f'{dataset_key}/balanced_retention_nll_auc'] = np.mean(
        results['balanced_retention_nll_arr'])
    eval_metrics[f'{dataset_key}/balanced_retention_auroc_auc'] = np.mean(
        results['balanced_retention_auroc_arr'])
    eval_metrics[f'{dataset_key}/balanced_retention_auprc_auc'] = np.mean(
        results['balanced_retention_auprc_arr'])
  else:
    # This is added for convenience when logging (so the entry exists
    # in tabular format)
    eval_metrics[f'{dataset_key}/ood_detection_auroc'] = None
    eval_metrics[f'{dataset_key}/ood_detection_auprc'] = None
    eval_metrics[f'{dataset_key}/balanced_retention_accuracy_auc'] = None
    eval_metrics[f'{dataset_key}/balanced_retention_nll_auc'] = None
    eval_metrics[f'{dataset_key}/balanced_retention_auroc_auc'] = None
    eval_metrics[f'{dataset_key}/balanced_retention_auprc_auc'] = None

  if compute_retention_auc:
    assert 'accuracy_arr' in results
    assert 'nll_arr' in results

    eval_metrics[f'{dataset_key}/retention_accuracy_auc'] = np.mean(
        results['retention_accuracy_arr'])
    eval_metrics[f'{dataset_key}/retention_nll_auc'] = np.mean(
        results['retention_nll_arr'])
    eval_metrics[f'{dataset_key}/retention_auroc_auc'] = np.mean(
        results['retention_auroc_arr'])
    eval_metrics[f'{dataset_key}/retention_auprc_auc'] = np.mean(
        results['retention_auprc_arr'])

  return eval_metrics


def compute_dataset_eval_metrics(
    dataset_key,
    results,
    ece_num_bins=15,
    compute_open_set_recognition=False,
    compute_retention_auc=False,
):
  """Compute scalar metrics.

  Args:
    dataset_key: str, name of dataset (prepends each metric in returned Dict).
    results: Dict, results for a particular dataset split.
    ece_num_bins: int, number of bins used in computing expected calibration
      error.
    compute_open_set_recognition: bool, if True, compute OOD detection PR and
      AUROC metrics.
    compute_retention_auc: bool, if True, compute retention AUC metrics by
      taking the mean of the retention arrays.

  Returns:
      Dict, scalar metrics.
  """
  y_pred, y_true, y_pred_entropy = (results['y_pred'], results['y_true'],
                                    results['y_pred_entropy'])

  eval_metrics = dict()

  # Standard predictive metrics
  eval_metrics[f'{dataset_key}/negative_log_likelihood'] = log_loss(
      y_pred=y_pred, y_true=y_true, labels=np.asarray([0, 1]))
  try:
    eval_metrics[f'{dataset_key}/auroc'] = roc_auc_score(
        y_true=y_true, y_score=y_pred, labels=np.asarray([0, 1]))
  except ValueError:
    eval_metrics[f'{dataset_key}/auroc'] = None
  precision, recall, _ = precision_recall_curve(
      y_true=y_true, probas_pred=y_pred)
  eval_metrics[f'{dataset_key}/auprc'] = auc(recall, precision)
  eval_metrics[f'{dataset_key}/accuracy'] = (
      accuracy_score(y_true=y_true, y_pred=(y_pred > 0.5)))

  # Uncertainty metrics
  ece = rm.metrics.ExpectedCalibrationError(num_bins=ece_num_bins)
  ece.add_batch(y_pred, label=y_true)
  eval_metrics[f'{dataset_key}/ece'] = ece.result()['ece']

  if compute_open_set_recognition:
    is_ood = results['is_ood']
    eval_metrics[f'{dataset_key}/ood_detection_auroc'] = roc_auc_score(
        y_true=is_ood, y_score=y_pred_entropy)
    precision, recall, _ = precision_recall_curve(
        y_true=is_ood, probas_pred=y_pred_entropy)
    eval_metrics[f'{dataset_key}/ood_detection_auprc'] = auc(recall, precision)

    # For the joint datasets, we also compute a rebalanced retention metric,
    # in which we duplicate the OOD dataset to match the size of the in-domain
    # dataset, and then compute the retention metrics.
    rebal_ret_scores = compute_rebalanced_retention_scores(results)
    eval_metrics[f'{dataset_key}/balanced_retention_accuracy_auc'] = (
        rebal_ret_scores['accuracy'])
    eval_metrics[f'{dataset_key}/balanced_retention_nll_auc'] = (
        rebal_ret_scores['nll'])
    eval_metrics[f'{dataset_key}/balanced_retention_auroc_auc'] = (
        rebal_ret_scores['auroc'])
    eval_metrics[f'{dataset_key}/balanced_retention_auprc_auc'] = (
        rebal_ret_scores['auprc'])
  else:
    # This is added for convenience when logging (so the entry exists
    # in tabular format)
    eval_metrics[f'{dataset_key}/ood_detection_auroc'] = None
    eval_metrics[f'{dataset_key}/ood_detection_auprc'] = None
    eval_metrics[f'{dataset_key}/balanced_retention_accuracy_auc'] = None
    eval_metrics[f'{dataset_key}/balanced_retention_nll_auc'] = None
    eval_metrics[f'{dataset_key}/balanced_retention_auroc_auc'] = None
    eval_metrics[f'{dataset_key}/balanced_retention_auprc_auc'] = None

  if compute_retention_auc:
    assert 'accuracy_arr' in results
    assert 'nll_arr' in results
    eval_metrics[f'{dataset_key}/retention_accuracy_auc'] = np.mean(
        compute_retention_curve_on_accuracies(
            accuracies=results['accuracy_arr'], uncertainty=y_pred_entropy))
    eval_metrics[f'{dataset_key}/retention_nll_auc'] = np.mean(
        compute_retention_curve_on_losses(
            losses=results['nll_arr'], uncertainty=y_pred_entropy))
    eval_metrics[f'{dataset_key}/retention_auroc_auc'] = np.mean(
        compute_auc_retention_curve(
            y_pred=y_pred,
            y_true=y_true,
            uncertainty=y_pred_entropy,
            auc_str='roc'))
    eval_metrics[f'{dataset_key}/retention_auprc_auc'] = np.mean(
        compute_auc_retention_curve(
            y_pred=y_pred,
            y_true=y_true,
            uncertainty=y_pred_entropy,
            auc_str='prc'))

  return eval_metrics


def compute_roc_curve(y_uncertainty: np.ndarray, is_ood: np.ndarray):
  """Compute OOD detection ROC curve using sklearn methods.

  Args:
    y_uncertainty: np.ndarray, uncertainty scores for each example.
    is_ood: np.ndarray, Boolean array indicating if an example is from the OOD
      dataset (True) or in-domain (False).

  Returns:
      Tuple[np.ndarray, np.ndarray, np.float]:
        FPR curve, TPR curve, and ROC-AUC value.
  """
  fpr, tpr, _ = roc_curve(y_true=is_ood, y_score=y_uncertainty)
  roc_auc = auc(x=fpr, y=tpr)
  return fpr, tpr, roc_auc


def get_retention_curve_normalizer(use_oracle, n_objects):
  """Obtain normalization constants for each entry of the unnormalized
  retention curve.

  When using an oracle, we divide by the total number of objects.
  When not, we divide by the object index (i.e., the number of objects used
    to compute the model metric at each referral rate).

  Args:
    use_oracle: Bool, if True, evaluate the combined predictive performance
      of the model and an oracle that is correct on all referred datapoints.
    n_objects: int, number of objects used to create the retention curve.
  """
  if use_oracle:
    return n_objects
  else:
    # e.g., for 5 objects, returns [5, 4, 3, 2, 1, 1], where the extra
    # element at the end divides the term corresponding to referring
    # all examples.
      normalizer = np.arange(n_objects + 1)
      normalizer[0] = 1
      return normalizer[::-1]


def compute_retention_curve_on_losses(losses, uncertainty, use_oracle=False):
  """Computes a retention curve on a loss (where lower loss is better)
  and corresponding per-example uncertainty values.

  Based on utils by Andrey Malinin, Yandex Research.
  https://github.com/yandex-research/shifts/blob/main/weather/assessment.py

  Args:
    losses: np.ndarray, losses from a particular dataset.
    uncertainty: np.ndarray, per-example uncertainties for the same dataset,
      should follow the order of the losses.
    use_oracle: Bool (default: False), if True, evaluate the combined predictive
      performance of the model and an oracle that is correct on all referred
      datapoints.

  Returns:
    np.ndarray, retention curve at all possible retention thresholds,
      including all examples retained (i.e., no referral) and no examples
      retained (i.e., all points referred to an expert).
  """
  n_objects = losses.shape[0]
  uncertainty_order = uncertainty.argsort()

  # Losses in order of increasing uncertainty
  losses = losses[uncertainty_order]
  error_rates = np.zeros(n_objects + 1)
  error_rates[:-1] = np.cumsum(losses)[::-1]

  # With oracle:
  # * Divide by total number of predictions
  # Without oracle:
  # * Divide by only the number of predictions the model must make at each
  # * referral rate
  normalizer = get_retention_curve_normalizer(use_oracle, n_objects)
  error_rates = error_rates / normalizer

  return error_rates


def compute_retention_curve_on_accuracies(
    accuracies,
    uncertainty,
    use_oracle=False
):
  """Computes a retention curve on an accuracy (where higher accuracy is better)
  and corresponding per-example uncertainty values.

  Based on utils by Andrey Malinin, Yandex Research.
  https://github.com/yandex-research/shifts/blob/main/weather/assessment.py

  Args:
    accuracies: np.ndarray, accuracies from a particular dataset.
    uncertainty: np.ndarray, per-example uncertainties for the same dataset,
      should follow the order of the accuracies.
    use_oracle: Bool (default: False), if True, evaluate the combined predictive
      performance of the model and an oracle that is correct on all referred
      datapoints.

  Returns:
    np.ndarray, retention curve at all possible retention thresholds,
      including all examples retained (i.e., no referral) and no examples
      retained (i.e., all points referred to an expert).
  """
  n_objects = accuracies.shape[0]
  uncertainty_order = uncertainty.argsort()

  # Per-point accuracy (binary) in order of increasing uncertainty
  accuracies = accuracies[uncertainty_order]
  retention_arr = np.zeros(n_objects + 1)

  for i in range(1, n_objects):
    accuracy_i = accuracies[:i].sum()

    if use_oracle:
      j = n_objects - i
      accuracy_i += j

    retention_arr[i] = accuracy_i

  # With oracle:
  # * Divide by total number of predictions
  # Without oracle:
  # * Divide by only the number of predictions the model must make at each
  # * referral rate
  normalizer = get_retention_curve_normalizer(use_oracle, n_objects)

  # Assume perfect performance when all examples have been referred.
  retention_arr[0] = n_objects if use_oracle else 1
  retention_arr[-1] = accuracies.sum()

  acc_rates = retention_arr[::-1] / normalizer
  return acc_rates


def compute_auc_retention_curve(y_pred,
                                y_true,
                                uncertainty,
                                auc_str,
                                n_buckets=100,
                                use_oracle=False):
  """Computes a retention curve for AUC or AUPRC using predictions, ground
  truths, and corresponding per-example uncertainty values.

  Based on utils by Andrey Malinin, Yandex Research.
  https://github.com/yandex-research/shifts/blob/main/weather/assessment.py

  Args:
    y_pred: np.ndarray, predicted sigmoid probabilities.
    y_true: np.ndarray, ground truth values.
    uncertainty: np.ndarray, per-example uncertainties for the same dataset,
      should follow the order of the accuracies.
    auc_str: str, determines if we evaluate the retention AUC or AUPRC.
    n_buckets: int, number of retention thresholds to evaluate (AUC can be
      costly to evaluate for thousands of possible thresholds.
    use_oracle: Bool (default: False), if True, evaluate the combined predictive
      performance of the model and an oracle that is correct on all referred
      datapoints.

  Returns:
    np.ndarray, AUC or AUPRC retention curve at specified number of thresholds.
  """

  def compute_auroc(true, pred):
    return roc_auc_score(y_true=true, y_score=pred)

  def compute_auprc(true, pred):
    precision, recall, _ = precision_recall_curve(y_true=true, probas_pred=pred)
    return auc(recall, precision)

  if auc_str == 'roc':
    auc_fn = compute_auroc
  elif auc_str == 'prc':
    auc_fn = compute_auprc
  else:
    raise NotImplementedError

  n_objects = y_pred.shape[0]
  bucket_indices = (np.arange(n_buckets) / n_buckets) * n_objects
  bucket_indices = bucket_indices.astype(int)

  uncertainty_order = uncertainty.argsort()
  y_pred = y_pred[uncertainty_order]
  y_true = y_true[uncertainty_order]

  retention_arr = np.zeros(n_buckets + 1)

  if use_oracle:
    # We will later divide by n_objects
    perfect_performance = n_objects
  else:
    perfect_performance = 1

  # Assume perfect performance when all examples have been referred.
  retention_arr[0] = perfect_performance

  check_n_unique = True

  for i_buckets in range(1, n_buckets):
    i_objects = bucket_indices[i_buckets]
    j_objects = n_objects - i_objects
    y_pred_curr = y_pred[:i_objects]
    y_true_curr = y_true[:i_objects]

    # For the first few low uncertainty points, we may be predicting the same
    # (correct) label, which would break AUC. We default to assigning
    # perfect AUC until this condition is broken.
    if (check_n_unique and
        len(np.unique(y_true_curr)) == 1 and
            np.array_equal(y_pred_curr > 0.5, y_true_curr)):
      retention_arr[i_buckets] = perfect_performance
    else:
      try:
        auc_val = auc_fn(true=y_true_curr, pred=y_pred_curr)
        if use_oracle:
          # Weight current AUC/AUPRC by number of objects, and
          # add weight of oracle's perfect prediction.
          retention_arr[i_buckets] = (i_objects * auc_val) + j_objects
        else:
          retention_arr[i_buckets] = auc_val
      except ValueError:  # All of the preds are the same
        if use_oracle:
          retention_arr[i_buckets] = j_objects
        else:
          retention_arr[i_buckets] = 0
      check_n_unique = False

  # Handle the case when no examples have been referred.
  try:
    auc_val = auc_fn(true=y_true, pred=y_pred)
    if use_oracle:
      retention_arr[-1] = n_objects * auc_val
    else:
      retention_arr[-1] = auc_val
  except ValueError:
    retention_arr[-1] = 0

  if use_oracle:
    auc_retention = retention_arr[::-1] / n_objects
  else:
    auc_retention = retention_arr[::-1]

  return auc_retention


def compute_ood_calibration_curve(y_pred: np.ndarray):
  """OOO calibration curve.

  Form a curve by sweeping over confidences in [0, 1], and counting
  the number of predictions with confidence over the threshold.
  On an OOD dataset, we should expect low confidence.
  Args:
    y_pred: np.ndarray, predictions on the OOD dataset

  Returns:
    Tuple[np.ndarray, np.ndarray]
      First np.ndarray:
        sorted probabilities for the predicted class
      Second np.ndarray:
        number of predictions with confidence greater than or equal to
        the confidence at a given threshold
  """
  # TODO(nband): Debug OOD calibration curve metric.
  raise NotImplementedError('Implementation in progress.')

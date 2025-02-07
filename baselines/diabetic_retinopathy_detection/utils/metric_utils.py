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

"""Utils for initializing, updating, and logging metrics."""
# pylint: disable=g-bare-generic
# pylint: disable=g-doc-args
# pylint: disable=g-doc-return-or-yield
# pylint: disable=g-importing-member
# pylint: disable=g-no-space-after-docstring-summary
# pylint: disable=g-short-docstring-punctuation
# pylint: disable=logging-format-interpolation
# pylint: disable=logging-fstring-interpolation
# pylint: disable=missing-function-docstring
from typing import List

import robustness_metrics as rm
from tabulate import tabulate
import tensorflow as tf


def get_diabetic_retinopathy_base_metrics(
    use_tpu: bool,
    num_bins: int,
    available_splits: List[str],
    use_validation: bool = True,
):
  """Initialize base metrics for non-ensemble Diabetic Retinopathy predictors.

  Should be called within the distribution strategy scope (e.g. see
  deterministic.py script).

  Note:
    We disclude AUCs in non-TPU case, which must be defined and added to this
    dict outside the strategy scope. See the
    `get_diabetic_retinopathy_cpu_eval_metrics` method for initializing AUCs.
    We disclude ECE in TPU case, which currently throws an XLA error on TPU.

  Args:
    use_tpu: bool, is run using TPU.
    num_bins: int, number of ECE bins.
    available_splits: List[str], evaluation dataset splits available.
    use_validation: whether to use a validation split.

  Returns:
    dict, metrics
  """
  validation_datasets = [key for key in available_splits if 'validation' in key]
  test_datasets = [key for key in available_splits if 'test' in key]
  eval_datasets = test_datasets
  if use_validation:
    eval_datasets = validation_datasets + eval_datasets

  metrics = {
      'train/negative_log_likelihood': tf.keras.metrics.Mean(),
      'train/accuracy': tf.keras.metrics.BinaryAccuracy(),
      'train/loss': tf.keras.metrics.Mean(),  # NLL + L2
  }
  for eval_split in eval_datasets:
    metrics.update({
        f'{eval_split}/negative_log_likelihood': tf.keras.metrics.Mean(),
        f'{eval_split}/accuracy': tf.keras.metrics.BinaryAccuracy(),
    })

  if use_tpu:
    # AUC does not yet work within GPU strategy scope, but does for TPU
    metrics.update({
        'train/auprc': tf.keras.metrics.AUC(curve='PR'),
        'train/auroc': tf.keras.metrics.AUC(curve='ROC'),
    })
    for eval_split in eval_datasets:
      metrics.update({
          f'{eval_split}/auprc': tf.keras.metrics.AUC(curve='PR'),
          f'{eval_split}/auroc': tf.keras.metrics.AUC(curve='ROC')
      })
  else:
    # ECE does not yet work on TPU
    metrics.update({
        'train/ece': rm.metrics.ExpectedCalibrationError(num_bins=num_bins),
    })
    for eval_split in eval_datasets:
      metrics.update({
          f'{eval_split}/ece':
              rm.metrics.ExpectedCalibrationError(num_bins=num_bins)
      })

  return metrics


def get_diabetic_retinopathy_cpu_metrics(
    available_splits: List[str],
    use_validation: bool = True,
):
  """Initialize metrics for non-ensemble Diabetic Retinopathy predictors.

  that must be initialized outside of the accelerator scope for CPU evaluation.

  Note that this method will cause an error on TPU.

  Args:
    available_splits: List[str], evaluation dataset splits available.
    use_validation: bool, whether to use a validation split.

  Returns:
    dict, metrics
  """
  metrics = {
      'train/auprc': tf.keras.metrics.AUC(curve='PR'),
      'train/auroc': tf.keras.metrics.AUC(curve='ROC')
  }

  validation_datasets = [key for key in available_splits if 'validation' in key]
  test_datasets = [key for key in available_splits if 'test' in key]
  eval_datasets = test_datasets
  if use_validation:
    eval_datasets = validation_datasets + eval_datasets

  for eval_split in eval_datasets:
    metrics.update({
        f'{eval_split}/auprc': tf.keras.metrics.AUC(curve='PR'),
        f'{eval_split}/auroc': tf.keras.metrics.AUC(curve='ROC')
    })

  return metrics


def log_vit_validation_metrics(eval_results):
  """Logs ViT fine-tuning evaluation metrics.

  Args:
    eval_results: Dict, contains eval scalar metric results with key formatted
      as {split}/{metric}.

  Returns:
    Dict, metrics for TensorBoard logging.
  """
  metrics_to_return = {}

  # Standard evaluation, robustness, and uncertainty quantification metrics
  eval_columns = [
      'Eval Dataset',
      'NLL', 'Accuracy', 'AUPRC', 'AUROC', 'ECE',
      'OOD AUROC', 'OOD AUPRC',
      'R-Accuracy AUC', 'R-NLL AUC', 'R-AUROC AUC', 'R-AUPRC AUC',
      'Balanced R-Accuracy AUC', 'Balanced R-NLL AUC',
      'Balanced R-AUROC AUC', 'Balanced R-AUPRC AUC']
  eval_metrics = [
      'negative_log_likelihood', 'accuracy', 'auprc', 'auroc', 'ece',
      'ood_detection_auroc', 'ood_detection_auprc',
      'retention_accuracy_auc', 'retention_nll_auc',
      'retention_auroc_auc', 'retention_auprc_auc',
      'balanced_retention_accuracy_auc', 'balanced_retention_nll_auc',
      'balanced_retention_auroc_auc', 'balanced_retention_auprc_auc']

  eval_values = list()
  for dataset_key, results_dict in eval_results.items():
    dataset_values = list()
    dataset_values.append(dataset_key)

    # Add all the relevant metrics from the per-dataset split results dict
    for eval_metric in eval_metrics:
      dataset_key_and_metric = f'{dataset_key}/{eval_metric}'
      eval_value = results_dict[dataset_key_and_metric]
      dataset_values.append(eval_value)

      # Add to the metrics dict which we will return (for TensorBoard logging)
      metrics_to_return[dataset_key_and_metric] = eval_value

    eval_values.append(dataset_values)

  eval_table = tabulate(eval_values, eval_columns,
                        tablefmt='simple', floatfmt='8.4f')
  print('\n')
  print(eval_table)
  return metrics_to_return


def log_epoch_metrics(metrics, eval_results, use_tpu, dataset_splits):
  """Logs train, validation, and test epoch metrics.

  Args:
    metrics: Dict, contains train scalar metric results with key formatted as
      {split}/{metric}.
    eval_results: Dict, contains eval scalar metric results with key formatted
      as {split}/{metric}.
    use_tpu: bool, if True, using the TPU, which means that ECE cannot be
      collected at train time (current bug).
    dataset_splits: List[str], available dataset splits.

  Returns:
    Dict, metrics for TensorBoard logging.
  """
  metrics_to_return = {}

  if 'train' in dataset_splits:
    train_columns = ['Train Loss (NLL+L2)', 'Accuracy', 'AUPRC', 'AUROC']
    train_metrics = ['loss', 'accuracy', 'auprc', 'auroc']
    train_values = [
        metrics['train/loss'].result().numpy(),
        metrics['train/accuracy'].result().numpy() * 100,
        metrics['train/auprc'].result().numpy() * 100,
        metrics['train/auroc'].result().numpy() * 100
    ]
    if not use_tpu:
      train_columns.append('ECE')
      train_values.append(metrics['train/ece'].result()['ece'] * 100)
      train_metrics.append('ece')

    train_table = tabulate([train_values],
                           train_columns,
                           tablefmt='simple',
                           floatfmt='8.4f')
    print(train_table)

    # Log to the metrics dict which we will return (for TensorBoard)
    for train_metric, train_value in zip(train_metrics, train_values):
      metrics_to_return[f'train/{train_metric}'] = train_value

  # Standard evaluation, robustness, and uncertainty quantification metrics
  eval_columns = [
      'Eval Dataset', 'NLL', 'Accuracy', 'AUPRC', 'AUROC', 'ECE', 'OOD AUROC',
      'OOD AUPRC', 'R-Accuracy AUC', 'R-NLL AUC', 'R-AUROC AUC', 'R-AUPRC AUC',
      'Balanced R-Accuracy AUC', 'Balanced R-NLL AUC', 'Balanced R-AUROC AUC',
      'Balanced R-AUPRC AUC'
  ]
  eval_metrics = [
      'negative_log_likelihood', 'accuracy', 'auprc', 'auroc', 'ece',
      'ood_detection_auroc', 'ood_detection_auprc', 'retention_accuracy_auc',
      'retention_nll_auc', 'retention_auroc_auc', 'retention_auprc_auc',
      'balanced_retention_accuracy_auc', 'balanced_retention_nll_auc',
      'balanced_retention_auroc_auc', 'balanced_retention_auprc_auc'
  ]

  eval_values = list()
  for dataset_key, results_dict in eval_results.items():
    dataset_values = list()
    dataset_values.append(dataset_key)

    # Add all the relevant metrics from the per-dataset split results dict
    for eval_metric in eval_metrics:
      dataset_key_and_metric = f'{dataset_key}/{eval_metric}'
      eval_value = results_dict[dataset_key_and_metric]
      dataset_values.append(eval_value)

      # Add to the metrics dict which we will return (for TensorBoard logging)
      metrics_to_return[dataset_key_and_metric] = eval_value

    eval_values.append(dataset_values)

  eval_table = tabulate(
      eval_values, eval_columns, tablefmt='simple', floatfmt='8.4f')
  print('\n')
  print(eval_table)
  return metrics_to_return


def flatten_dictionary(x):
  """Flattens a dictionary where elements may itself be a dictionary.

  This function is helpful when using a collection of metrics, some of which
  include Robustness Metrics' metrics. Each metric in Robustness Metrics
  returns a dictionary with potentially multiple elements. This function
  flattens the dictionary of dictionaries.

  Args:
    x: Dictionary where keys are strings such as the name of each metric.

  Returns:
    Flattened dictionary.
  """
  outputs = {}
  for k, v in x.items():
    if isinstance(v, dict):
      if len(v.values()) == 1:
        # Collapse metric results like ECE's with dicts of len 1 into the
        # original key.
        outputs[k] = list(v.values())[0]
      else:
        # Flatten metric results like diversity's.
        for v_k, v_v in v.items():
          outputs[f'{k}/{v_k}'] = v_v
    else:
      outputs[k] = v
  return outputs
